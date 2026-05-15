"""
FastAPI 应用 + 所有路由
  - WebSocket /ws/camera  : 接收摄像头推流（单向，不回发）
  - WebSocket /ws/webrtc  : WebRTC 信令（SDP Offer/Answer，vanilla ICE）
  - GET /video/original   : 原始画面 MJPEG 流（备用）
  - GET /video/infer      : 推理结果 MJPEG 流（备用）
  - GET /performance      : 性能指标 JSON
  - GET /snapshot         : 截图并保存
  - GET /                 : WebUI 首页
"""
import asyncio
import os
import time
import cv2

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from . import state
from .html_page import HTML

# aiortc 可选依赖：未安装时 WebRTC 端点返回 1011，其余功能不受影响
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from .webrtc_track import InferenceVideoTrack
    _WEBRTC_OK = True
except ImportError:
    _WEBRTC_OK = False

_webrtc_pcs: set = set()   # 管理所有 RTCPeerConnection 生命周期

app = FastAPI(title="Fisheye YOLO GPU WebUI")


@app.on_event("startup")
async def _startup() -> None:
    state._frame_event_loop = asyncio.get_running_loop()
    if state.upload_mode == 'udp':
        from . import udp_receiver
        asyncio.create_task(udp_receiver.recv_loop())


# ── WebSocket：接收摄像头帧（单向）─────────────────────────────────────
@app.websocket("/ws/camera")
async def camera_ws(websocket: WebSocket) -> None:
    """
    完全单向：client → server，服务器不回发任何数据。
    推理结果通过 /video/infer MJPEG 流展示给浏览器。
    单向设计彻底规避 websockets 并发写（PONG vs data）导致的 AssertionError。
    """
    await websocket.accept()
    loop = asyncio.get_event_loop()

    with state.perf_lock:
        state.performance_data["connected_clients"] += 1
    print(f"📡 新客户端连接，当前在线: {state.performance_data['connected_clients']}")

    latest_frame: list = [None]
    alive: list = [True]
    frame_event = asyncio.Event()  # drain_recv 收到帧后立即 set，消除轮询延迟

    async def drain_recv() -> None:
        try:
            while alive[0]:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_bytes(), timeout=20.0
                    )
                    # 原始 JPEG 直接在摄像头速率更新，不等推理完成
                    with state.frame_lock:
                        state.latest_original_jpeg = data
                    latest_frame[0] = data
                    frame_event.set()  # 立即唤醒推理调度循环，不再等 5ms 轮询
                except asyncio.TimeoutError:
                    print("⚠️  WebSocket 20s 无数据，关闭连接")
                    alive[0] = False
        except WebSocketDisconnect:
            alive[0] = False
        except Exception as e:
            print(f"⚠️  drain_recv 异常: {type(e).__name__}: {e}")
            alive[0] = False
        finally:
            frame_event.set()  # 唤醒主循环，让它检查 alive[0] 并退出

    recv_task = asyncio.create_task(drain_recv())
    try:
        while alive[0]:
            frame_data = latest_frame[0]
            if frame_data is None:
                frame_event.clear()
                await frame_event.wait()  # drain_recv 收到帧立即唤醒
                continue
            latest_frame[0] = None
            try:
                await loop.run_in_executor(
                    state.inference_executor, state.inference_fn, frame_data
                )
            except Exception as e:
                print(f"⚠️  推理异常 [{type(e).__name__}]: {e}")
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as e:
        print(f"⚠️  WebSocket 主循环异常: {type(e).__name__}: {e}")
    finally:
        alive[0] = False
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass
        with state.perf_lock:
            state.performance_data["connected_clients"] -= 1
        print(f"📡 客户端断开，当前在线: {state.performance_data['connected_clients']}")


# ── MJPEG 流（原始画面，浏览器原生支持，延迟可接受）──────────────────
async def _mjpeg_gen_original():
    last_buf = None
    while True:
        with state.frame_lock:
            buf = state.latest_original_jpeg
        if buf is not None and buf is not last_buf:
            last_buf = buf
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf + b'\r\n'
        await asyncio.sleep(0.010)


@app.get("/video/original")
async def video_original():
    return StreamingResponse(_mjpeg_gen_original(),
                              media_type="multipart/x-mixed-replace; boundary=frame")


# ── REST API ───────────────────────────────────────────────────────────
@app.get("/performance")
def get_performance():
    with state.perf_lock:
        return {k: v for k, v in state.performance_data.items()
                if not k.startswith('_')}


@app.get("/inference/latest")
def inference_latest():
    """
    返回最新一帧的推理结果 JSON（轮询用）。
    尚无数据时返回 503；有数据后 content-type 为 application/json。
    """
    data = state.latest_inference_result
    if data is None:
        return JSONResponse({'error': 'no inference result yet'}, status_code=503)
    return Response(content=data, media_type="application/json")


@app.websocket("/ws/inference")
async def inference_ws(websocket: WebSocket) -> None:
    """
    推理结果实时推送（事件驱动）。
    每帧推理完成后立即推送 JSON bytes，无需客户端轮询或 ACK。
    JSON 格式：{"timestamp": float, "frame_id": int, "targets": {"<id>": {...}}}
    """
    await websocket.accept()

    ev = asyncio.Event()
    with state._inference_waiters_lock:
        state._inference_ready_waiters.add(ev)

    last_data = None
    try:
        while True:
            await ev.wait()
            ev.clear()
            data = state.latest_inference_result
            if data is None or data is last_data:
                continue
            last_data = data
            await websocket.send_bytes(data)
    except Exception:
        pass
    finally:
        with state._inference_waiters_lock:
            state._inference_ready_waiters.discard(ev)


@app.get("/snapshot")
def snapshot():
    with state.frame_lock:
        f = state.latest_annotated_frame
    if f is None:
        return {"msg": "no frame available", "filename": None}
    os.makedirs("screenshots", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f"screenshots/snapshot_{ts}.jpg"
    cv2.imwrite(path, f)
    print(f"📸 截图已保存: {path}")
    return {"msg": "saved", "filename": path}


@app.websocket("/ws/webrtc")
async def webrtc_signaling(websocket: WebSocket) -> None:
    """
    WebRTC 信令通道（Vanilla ICE）：
      1. 浏览器发送 SDP Offer
      2. 服务端返回完整 SDP Answer（含所有 ICE Candidate）
      3. 信令完成后 WebSocket 关闭，媒体流走独立 UDP SRTP
    aiortc 使用 CPU openh264 编码，分辨率已限制在 1280px 以内。
    """
    if not _WEBRTC_OK:
        await websocket.close(code=1011, reason="aiortc not installed: pip install aiortc av")
        return

    await websocket.accept()
    pc = RTCPeerConnection()
    _webrtc_pcs.add(pc)
    track = InferenceVideoTrack(fps=30)

    @pc.on("connectionstatechange")
    async def _on_state() -> None:
        if pc.connectionState in ("failed", "closed", "disconnected"):
            track.stop()
            await pc.close()
            _webrtc_pcs.discard(pc)
            print(f"[webrtc] 连接关闭 ({pc.connectionState})，剩余 {len(_webrtc_pcs)} 路")

    pc.addTrack(track)

    try:
        # 接收浏览器 Offer（15s 超时）
        msg = await asyncio.wait_for(websocket.receive_json(), timeout=15.0)
        offer = RTCSessionDescription(sdp=msg["sdp"], type=msg["type"])
        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # 等待 ICE 收集完成（局域网通常 <200ms，设 5s 上限保底）
        deadline = time.time() + 5.0
        while pc.iceGatheringState != "complete" and time.time() < deadline:
            await asyncio.sleep(0.05)

        await websocket.send_json({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        })
        print(f"[webrtc] 信令完成，当前 {len(_webrtc_pcs)} 路连接")
    except asyncio.TimeoutError:
        print("[webrtc] 等待 Offer 超时")
    except Exception as e:
        print(f"[webrtc] 信令异常: {type(e).__name__}: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/")
def index():
    return HTMLResponse(HTML)
