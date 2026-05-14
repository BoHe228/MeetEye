# 鱼眼全景 YOLO GPU WebUI — 全链路代码说明

> 覆盖文件：`main_GPU_webui.py` / `camera_client.py` / `webui/routes.py` / `webui/processor.py` / `webui/state.py` / `webui/gpu_info.py` / `webui/html_page.py` / `webui/webrtc_track.py`

---

## 一、全链路总述

系统实现了一条完整的"本地视频 → 网络传输 → GPU 推理 → 网页渲染"流水线，整体分为三段：

```
本地摄像头（camera_client.py）
        │  WebSocket 二进制帧（JPEG，限速，禁止用 WebP 见§2.2）
        ▼
GPU 服务器 /ws/camera（routes.py）
        │  单线程推理池（ThreadPoolExecutor max_workers=1）
        ▼
推理流水线（processor.py）
  ① CPU→GPU 上传  ② GPU 鱼眼展开  ③ GPU→CPU 下载
  ④ CPU 裁剪切片  ⑤ GPU YOLO 批量推理（FP16）
  ⑥ CPU 合并过滤  ⑦ GPU OSNet ReID 特征提取
  ⑧ CPU BoT-SORT 跟踪  ⑨ CPU 角度计算
        │
        ├─── 背景编码线程（_encode_worker）
        │         缩放 → WebP 编码（质量 95）→ 追加 8 字节时间戳头
        │         → 供 /ws/video WebSocket 备用 + /video/infer MJPEG 备用
        │
        ├─── WebRTC Track（InferenceVideoTrack，主路）
        │         直接读 latest_annotated_frame（numpy）
        │         → 缩放至 _WEBRTC_WIDTH → aiortc H.264 → UDP SRTP
        ▼
浏览器（html_page.py）
  /ws/webrtc  WebRTC H.264（主路，UDP，低延迟）
  /ws/video   WebSocket WebP（备用，TCP）
  /video/infer  MJPEG WebP（备用，HTTP 轮询）
```

**关键设计原则**：

| 原则 | 实现方式 |
|------|---------|
| 推理串行化 | `ThreadPoolExecutor(max_workers=1)` 确保 CUDA 调用不并发 |
| 编码与推理解耦 | `_encode_worker` 独立线程，`threading.Event` 事件驱动 |
| WebRTC 主路下行 | `InferenceVideoTrack` 直接读 numpy，aiortc 按固定节拍编码 H.264 |
| 单向上行 | `/ws/camera` 服务端完全不回发，规避 WebSocket 并发写竞态 |

---

## 二、上行链路：本地视频的捕获、编码与推送

### 2.1 摄像头捕获与帧率控制

`camera_client.py` 在本地（Windows/Linux）运行，通过 OpenCV 打开摄像头：

```python
cap = cv2.VideoCapture(cam_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)   # 默认 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height) # 默认 720
```

帧率控制采用**软件限速**：

```python
interval = 1.0 / fps          # fps 默认 15
used = time.time() - t0       # 本帧实际耗时（含编码）
if interval - used > 0:
    await asyncio.sleep(interval - used)
```

### 2.2 帧编码：必须用 JPEG，禁止上行 WebP

`--format` 参数支持 `jpeg`（默认）和 `webp`，**上行必须保持 JPEG**：

```python
# camera_client.py
if fmt == 'webp':
    enc_ext    = '.webp'
    enc_params = [cv2.IMWRITE_WEBP_QUALITY, quality]
else:                           # 默认且推荐
    enc_ext    = '.jpg'
    enc_params = [cv2.IMWRITE_JPEG_QUALITY, quality]   # quality 默认 75
```

**为何禁止上行 WebP**：

| 编码 | 1920×1080 单帧耗时 | 上传帧率上限 | 结论 |
|------|-----------------|------------|------|
| JPEG Q75 | ~15 ms | ~50 fps | 轻松跑满 --fps 15 |
| WebP Q75 | ~150-220 ms | ~4-5 fps | 编码本身超过帧间隔，限速失效 |

WebP 编码耗时超过帧间隔（67ms@15fps）后，`asyncio.sleep` 中的限速永远不会触发，帧率完全由编码速度决定，导致上传只有 4-5 fps。

服务端 `cv2.imdecode` 对 JPEG 和 WebP 均透明解码，上行格式对推理流水线无影响。

### 2.3 WebSocket 上行连接

```python
async with websockets.connect(
    ws_url,
    max_size=30 * 1024 * 1024,  # 支持最大 30 MB 单帧
    open_timeout=10,
    ping_interval=None,          # 禁用 ping，避免与帧发送竞态
) as ws:
    await ws.send(buf.tobytes())
```

连接前先 HTTP 预检（`check_server_http`），断线后自动重连（`--retry` 默认 3s）。

### 2.4 服务端接收：`/ws/camera`

双协程分离设计：

```
drain_recv 协程（以摄像头速率持续接收）
  → latest_original_jpeg = data   # 原始帧，直接存供 /video/original MJPEG
  → latest_frame[0] = data        # 待推理帧
  → frame_event.set()             # 唤醒主调度协程

主调度协程
  → await frame_event.wait()
  → await run_in_executor(inference_executor, inference_fn, frame_data)
```

- 推理慢时新帧覆盖旧帧（帧丢弃），不累积队列
- 20 秒无数据超时主动断连

---

## 三、计算链路：GPU 推理流水线

核心类 `FisheyePanoramaYOLOPose`（`processor.py`）实现九步流水线，在 `process_panorama_slices()` 中顺序执行：

```
numpy BGR 帧（CPU）
  ① CPU→GPU  ──────── PCIe 上传，归一化为 float [0,1]
  ② 鱼眼展开 ──────── GPU 等距圆柱投影（预计算映射矩阵）
  ③ GPU→CPU  ──────── GPU 先转 uint8（12 MB），再 PCIe 下载（原 float 为 50 MB）
  ④ 裁剪切片 ──────── CPU：裁底部噪声 + 横向分 num_slices 片
  ⑤ YOLO 推理 ─────── GPU：FP16，batch 批量，torch.no_grad()
  ⑥ 合并过滤 ──────── CPU NMS + GPU OSNet 特征提取（GPU crop 路径）
  ⑦ 特征复用 ──────── 特征已在⑥内挂在 det['feature']，直接传跟踪器，省 ~15ms
  ⑧ 跟踪绘制 ──────── CPU：BoT-SORT 卡尔曼 + ReID 关联，draw_detections 绘框
  ⑨ 角度计算 ──────── CPU：关键点 → 水平方位角，叠加绘制到标注帧
numpy BGR 标注帧（CPU） → latest_annotated_frame
```

### 3.1 各步详解

#### ① CPU → GPU（PCIe 上传）

```python
frame_tensor = torch.from_numpy(frame).cuda().float() / 255.0
frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC → CHW，匹配 torchvision 约定
```

`frame` 是 `cv2.imdecode` 输出的 BGR uint8 numpy 数组，直接 `from_numpy` 零拷贝共享内存，`.cuda()` 触发一次 PCIe 传输。

#### ② 鱼眼展开（GPU）

```python
panorama_tensor = self.panorama_processor.apply_panorama_gpu(frame_tensor)
```

`FisheyePanoramaGPU` 在首帧懒初始化时预计算鱼眼→等距圆柱的像素映射表（双线性插值权重），之后每帧仅做一次 GPU gather 操作，无额外 CPU 参与。输出尺寸由 `--output-width / --output-height`（默认 3840×1080）控制。

#### ③ GPU → CPU（PCIe 下载，已提前转 uint8）

```python
panorama = (
    (panorama_tensor * 255.0)
    .clamp_(0, 255)
    .to(torch.uint8)           # float32 50MB → uint8 12MB，减少传输量 75%
    .permute(1, 2, 0)          # CHW → HWC
    .contiguous()
    .cpu()
    .numpy()
)
```

此时 `panorama_tensor`（GPU float32）仍保留在显存，供步骤⑥的 OSNet GPU crop 路径直接使用，无需二次上传。

#### ④ 裁剪 + 切片（CPU）

```python
crop_h = original_h // self.args.crop_divisor   # 裁掉底部噪声行（crop_divisor 默认 4 → 裁 1/4）
panorama = panorama[crop_h:, :]
slices, slice_infos = self.slicer.slice_panorama(panorama, num_slices=self.num_slices)
```

`PanoramaSlicer` 将全景横向均分为 `num_slices`（默认 3）片，相邻片有 `slice_overlap`（默认 5%）重叠，供后续 NMS 去重。切片坐标 `slice_infos` 同时记录是否跨越左右边界（`wrap_around`），支持等距圆柱的循环拼接。

**GPU 切片预备**：同步从 `panorama_tensor` 直接裁出对应 RGB float 切片张量 `slice_tensors_gpu`，供步骤⑥ OSNet 直接在 GPU 上提特征，跳过 numpy → PIL → transform 的 CPU 路径（省 ~5ms）：

```python
pano_rgb = panorama_tensor[[2, 1, 0], crop_h:, :]   # BGR→RGB，裁掉相同 crop_h 行
for info in slice_infos:
    st = pano_rgb[:, :, info['start_x']:info['end_x']]   # wrap_around 情形用 torch.cat
    slice_tensors_gpu.append(st)
```

#### ⑤ 批量 YOLO 推理（GPU，FP16）

```python
with torch.no_grad():
    all_yolo_results = self.yolo_detector.detect_batch(slices)   # slices: List[np.ndarray]
```

`YOLOPoseDetector.detect_batch` 内部以 `half=True` 调用 Ultralytics `predict()`，自动完成 FP16 转换、NMS 及关键点解码。YOLO 模型在 `initialize()` 时移至 `cuda`：

```python
self.yolo_detector.model.to('cuda')
# FP16 不在此静态转换，由每次 predict(half=True) 动态管理
```

> **为何 YOLO 用 FP16、OSNet 用 FP32**：YOLO 的卷积层对精度不敏感，FP16 可降低显存并提升吞吐；OSNet 含 BatchNorm，BN 的均值/方差在 FP16 下数值范围易溢出，需保留 FP32。

#### ⑥ 合并过滤（CPU NMS + GPU OSNet 特征提取）

```python
merged = self.slicer.merge_detections(
    all_yolo_results, slice_infos,
    slice_images=slices,
    slice_tensors=slice_tensors_gpu,    # 有则走 GPU crop 路径
    feature_extractor=self.feature_extractor,
)
filtered = filter_cross_boundary_detections(merged, panorama.shape)
filtered = self.slicer.filter_wide_detections(filtered, panorama.shape[1])
```

`merge_detections` 内部：
1. 将各切片局部坐标映射回全景坐标
2. IoU-NMS 去除重叠框（含跨切片重叠）
3. 对保留框逐人调用 OSNet 提取 ReID 特征（`slice_tensors_gpu` 存在时直接在 GPU 上 crop + resize，否则走 PIL 路径）
4. 特征向量挂在 `det['feature']` 上供跟踪器使用

#### ⑦ OSNet 特征复用

步骤⑥已完成特征提取并附加在检测结果上，此处仅是语义分层，无额外计算：

```python
dets_with_feat = filtered   # det['feature']: numpy [1, feat_dim]（或 None）
```

与"先 YOLO 后单独一次 OSNet 全图特征"方案相比，切片级 crop 特征提取节省约 15ms（避免全景图 resize 和冗余推理）。

#### ⑧ 跟踪 + 绘制（CPU）

```python
tracked = self.tracker.update(dets_with_feat)   # BoT-SORT：卡尔曼预测 + ReID 关联
annotated = draw_detections(panorama, tracked, self.tracker)
```

`BoT_SORTTracker` 使用卡尔曼滤波做运动预测，结合 ReID cosine 距离做外观匹配，支持左右边界 ID 保持（`enable_left_boundary / enable_right_boundary = True`，覆盖等距圆柱首尾循环场景）。`track_buffer=500` 表示目标消失后最多保留 500 帧再宣告丢失。

#### ⑨ 角度计算（CPU）

```python
kpts_list = [np.array(d['keypoints']) for d in tracked if d.get('keypoints')]
angle_info = self.angle_calculator.calculate_angles_from_keypoints(np.array(kpts_list))
annotated = self.angle_calculator.draw_angles_on_image(annotated, angle_info)
```

`AngleCalculator` 根据关键点在等距圆柱图中的 x 坐标，结合 `fisheye_calib.yaml` 标定参数（或多项式拟合 `fit_degree=5`），计算每个检测目标相对摄像头光轴的水平方位角，叠加绘制到标注帧。

### 3.2 懒初始化：全景处理器从第一帧初始化

`FisheyePanoramaGPU` 需要输入分辨率才能预计算映射矩阵，因此延迟到第一帧到来时初始化（`_init_panorama_from_frame`），加锁防止并发重复初始化：

```python
def process_panorama_slices(self, frame):
    if not self._panorama_ready:
        if not self._init_panorama_from_frame(frame):
            return None, None, None, None, None
    ...
```

`initialize()` 在启动时只加载 YOLO 和 OSNet 权重，不需要分辨率信息，可立即完成。

### 3.3 逐步耗时打印（每 30 帧一次）

```python
[总耗时 42.3ms | 检测 3 人]  ①CPU→GPU[CPU→GPU]=0.8ms  ②鱼眼展开[GPU]=1.2ms  ③GPU→CPU[GPU→CPU]=3.1ms
  ④裁剪切片[CPU]=0.4ms  ⑤YOLO推理[GPU]=18.6ms  ⑥合并过滤[CPU]=12.4ms
  ⑦特征复用[CPU]=0.0ms  ⑧跟踪绘制[CPU]=5.2ms  ⑨角度计算[CPU]=0.6ms
```

耗时最长的步骤通常是 **⑤ YOLO 推理**（GPU 批量）和 **⑥ 合并过滤**（含 OSNet 特征提取）。

### 3.4 性能指标计算（修正）

`update_perf` 传入 `total_ms`（JPEG 解码 + 九步流水线全过程），而非仅 `pipeline_ms`：

```python
# main_GPU_webui.py — inference_and_encode()
decode_ms   = (t_decoded - t_start) * 1000    # cv2.imdecode 耗时
pipeline_ms = (t_infer - t_decoded) * 1000    # 九步流水线耗时
total_ms    = (t_infer - t_start) * 1000      # 解码 + 流水线

update_perf(total_ms, detected_persons, tracking_ids)
#           ↑ 确保"帧处理耗时"与"理论 FPS = 1000/total_ms"分母一致
```

**FPS 两层含义**：

```
理论最高 FPS = 1000 / total_ms       # GPU 推理吞吐上限（performance_data["theoretical_fps"]）
实际处理 FPS = frame_count / dt      # 真实到达帧数/秒（受摄像头/网络速率限制）
```

| 实际 FPS vs 理论 FPS | 瓶颈位置 |
|---------------------|---------|
| 实际 << 理论（如 4 vs 33）| 摄像头/网络发送太慢，GPU 处于等待 |
| 实际 ≈ 理论 | GPU 推理满载，每帧都在全速处理 |

两个 FPS 均在页面实时展示，帮助快速判断瓶颈。

### 3.1 性能指标计算（修正）

`update_perf` 传入 `total_ms`（解码 + 推理全过程），而非仅 `pipeline_ms`：

```python
# main_GPU_webui.py
update_perf(total_ms, detected_persons, tracking_ids)
#           ↑ total_ms = decode_ms + pipeline_ms
```

**FPS 两层含义**：

```
理论最高 FPS = 1000 / total_ms       # GPU 推理吞吐上限，存入 performance_data["theoretical_fps"]
实际处理 FPS = frame_count / dt      # 真实到达帧数/秒，受摄像头发送速率限制
```

| 实际 FPS vs 理论 FPS | 瓶颈位置 |
|---------------------|---------|
| 实际 << 理论（如 4 vs 33）| 摄像头/网络发送太慢 |
| 实际 ≈ 理论 | GPU 推理满载 |

两个 FPS 均在页面实时展示，帮助快速判断瓶颈。

---

## 四、下行链路：处理结果的编码与网页渲染

### 4.1 两条并行下行路径

推理完成后，标注帧（`latest_annotated_frame`，numpy）通过两条独立路径到达浏览器：

```
latest_annotated_frame（numpy，3840×H BGR）
        │
        ├─── [路径 A] InferenceVideoTrack.recv()  ← WebRTC 主路
        │         每 1/fps 秒读一次（非阻塞，推理慢则复用上帧）
        │         → cv2.resize 到 _WEBRTC_WIDTH
        │         → av.VideoFrame → aiortc H.264 → UDP SRTP → 浏览器 <video>
        │
        └─── [路径 B] _encode_worker 线程  ← WebSocket/MJPEG 备用
                  _new_annotated_event 事件驱动（推理完立即唤醒）
                  → cv2.resize（若宽度 > _ENCODE_WIDTH=3840 则缩放）
                  → cv2.imencode(".webp", quality=95)
                  → [8字节时间戳] + WebP 字节
                  → latest_annotated_ws_frame（供 /ws/video）
                  → latest_annotated_jpeg（供 /video/infer MJPEG）
```

### 4.2 路径 A：WebRTC 主路（`webui/webrtc_track.py`）

**画质关键参数：`_WEBRTC_WIDTH`**

```python
_WEBRTC_WIDTH = 1280   # 调大此值可提升画质，代价是 CPU H.264 编码耗时增加
```

| `_WEBRTC_WIDTH` | 对 3840px 原图的缩放比 | H.264 编码耗时估算 | 适用场景 |
|----------------|--------------------|-----------------|----|
| 1280 | 3× 降采样 | ~15 ms | 低延迟优先 |
| **1920** | **2× 降采样** | **~30 ms** | **画质/延迟均衡（推荐）** |
| 2560 | 1.5× 降采样 | ~60 ms | 高画质 |
| 3840 | 不缩放 | ~120-200 ms | 最高画质，可能掉帧 |

`InferenceVideoTrack.recv()` 实现：

```python
async def recv(self) -> av.VideoFrame:
    # 按 fps 节拍调度（sleep 到下一帧时刻）
    if wait > 0:
        await asyncio.sleep(wait)
    self._next_time += 1.0 / self._fps

    # 非阻塞读最新标注帧（推理慢则复用上帧，不卡 H.264 编码器）
    with ws.frame_lock:
        frame = ws.latest_annotated_frame
    if frame is not None:
        self._last_ndarray = frame

    # 缩放 + 转 av.VideoFrame
    ndarray = cv2.resize(self._last_ndarray, (_WEBRTC_WIDTH, ...))
    vf = av.VideoFrame.from_ndarray(ndarray, format='bgr24')
    vf.pts = self._pts
    vf.time_base = fractions.Fraction(1, 90000)
    self._pts += 90000 // self._fps
    return vf
```

**WebRTC 信令流程**（Vanilla ICE，一次往返）：

```
浏览器                           服务端 /ws/webrtc
  创建 Offer
  等本端 ICE 收集完成（局域网 <200ms）
  ──── SDP Offer ─────────────▶
                                  setRemoteDescription
                                  createAnswer
                                  等服务端 ICE 收集完成
  ◀─── SDP Answer（含全部 ICE）──
  setRemoteDescription
  WebSocket 关闭（信令结束）
  ══════ UDP SRTP H.264 媒体流 ══════
```

服务端在 `@app.on_event("startup")` 时注册 asyncio 事件循环，确保 WebRTC 连接到来前事件机制已就绪。

### 4.3 路径 B：WebSocket 备用（`/ws/video`）

WebSocket 滑动窗口 ACK 协议，`_MAX_IN_FLIGHT=2`，格式为 `[8字节时间戳] + WebP`。当前主路已切换为 WebRTC，此路径作为回退保留。

### 4.4 路径 C：MJPEG 备用（`/video/infer`）

HTTP `multipart/x-mixed-replace`，内容类型为 `image/webp`（服务 `latest_annotated_jpeg` 中的 WebP 字节）。10ms 轮询，兼容旧浏览器及 VLC 等播放器。

### 4.5 浏览器端渲染（WebRTC 路径）

```javascript
// html_page.py — RTCPeerConnection 主流程
pc.addTransceiver('video', { direction: 'recvonly' });
pc.ontrack = (e) => { video.srcObject = e.streams[0]; };  // <video> 直接播放

// Vanilla ICE：等本端 ICE 收集完再发 Offer
await new Promise(resolve => {
    pc.addEventListener('icegatheringstatechange', () => {
        if (pc.iceGatheringState === 'complete') resolve();
    });
    setTimeout(resolve, 3000);  // 超时保底
});
sigWs.send(JSON.stringify({ sdp: pc.localDescription.sdp, type: 'offer' }));
```

**三段延迟拆解**：

| 段 | 测量方式 | 覆盖范围 | 精度 |
|----|---------|---------|------|
| ① WebP 编码耗时 | `perf_counter` 包围 `imencode` | 备用路径编码时间 | 精确 |
| ② 网络耗时·参考值 | `Date.now() - T0`（备用路径） | 时间戳→浏览器收到 | 受时钟漂移影响 |
| ③ 浏览器渲染耗时 | `requestVideoFrameCallback`（WebRTC路径） | 解码+绘制 | 精确 |

WebRTC 路径无法嵌入时间戳，② 仅在备用 WebSocket 路径下有意义。

---

## 五、线程与并发模型总览

```
线程/协程                       职责                          通信方式
────────────────────────────────────────────────────────────────────────
asyncio 事件循环（uvicorn）     HTTP/WebSocket/WebRTC 协议处理  asyncio.Event / call_soon_threadsafe
  └─ camera_ws 协程             接收上行帧，提交推理任务        frame_event（asyncio.Event）
  └─ drain_recv 协程            持续接收 WebSocket 数据         latest_frame[0]（共享列表）
  └─ video_ws 协程（×N）        备用 WebSocket WebP 推流        _frame_ready_waiters（asyncio.Event 集合）
  └─ webrtc_signaling 协程      WebRTC SDP/ICE 信令             WebSocket JSON
  └─ InferenceVideoTrack.recv() aiortc 按节拍拉帧               ws.frame_lock（直接读）

inference_executor（单线程）    串行 GPU 推理                   ThreadPoolExecutor（max_workers=1）
  └─ inference_and_encode       解码 + 9 步推理流水线            frame_lock（threading.Lock）

encode-worker 线程              WebP 编码 + 时间戳拼接          _new_annotated_event（threading.Event）
                                + 通知备用下行协程               call_soon_threadsafe

gpu-monitor 线程                每秒查询 GPU/CPU 指标           perf_lock（threading.RLock）
```

**锁使用原则**：
- `camera_lock`：保护 GPU 推理调用（防止并发 CUDA）
- `frame_lock`：保护帧缓冲读写（推理线程写，encode-worker/InferenceVideoTrack 读）
- `perf_lock`（RLock）：保护性能数据
- `_frame_waiters_lock`：保护备用 WebSocket 协程等待集合的增删

WebRTC 路径（`InferenceVideoTrack`）直接在 asyncio 事件循环中读 `frame_lock` 保护的帧缓冲，不使用 `_frame_ready_waiters` 通知机制，以固定节拍（15fps）轮询取帧，推理慢时复用上帧保持视频连续性。
