"""
WebUI 前端页面 HTML
"""

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>鱼眼全景 YOLO GPU WebUI</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{
    background:#0f172a;color:#e5e7eb;
    font-family:"Segoe UI",system-ui,sans-serif;
    min-height:100vh;padding:20px 20px 72px;
}
h1{text-align:center;font-size:1.5rem;margin-bottom:4px;}
.subtitle{text-align:center;color:#64748b;font-size:.85rem;margin-bottom:14px;}
.howto{
    max-width:780px;margin:0 auto 18px;
    background:rgba(30,41,59,.9);border:1px solid #334155;
    border-left:4px solid #f59e0b;border-radius:8px;padding:12px 16px;
    font-size:13px;line-height:1.7;
}
.howto strong{color:#fbbf24;}
code{
    background:rgba(51,65,85,.7);padding:1px 6px;
    border-radius:4px;font-family:monospace;font-size:12px;color:#7dd3fc;
}
.howto ol{padding-left:18px;}
.controls{display:flex;flex-wrap:wrap;justify-content:center;gap:10px;margin-bottom:18px;}
button{
    padding:10px 20px;border-radius:8px;border:none;
    font-size:13px;font-weight:600;cursor:pointer;
    transition:opacity .15s,transform .1s;
}
button:hover{opacity:.85;transform:translateY(-1px);}
.btn-record{background:#3b82f6;color:#fff;}
.btn-record-stop{background:#ef4444;color:#fff;}
.record-badge{
    display:inline-flex;align-items:center;gap:6px;
    background:rgba(239,68,68,.15);border:1px solid #ef4444;
    color:#ef4444;border-radius:6px;padding:6px 14px;font-size:13px;font-weight:600;
}
.record-dot{width:8px;height:8px;border-radius:50%;background:#ef4444;
    animation:blink 1s step-start infinite;}
@keyframes blink{50%{opacity:0;}}
.record-dest{
    display:flex;gap:0;border:1px solid #334155;border-radius:8px;overflow:hidden;
}
.record-dest label{
    flex:1;text-align:center;padding:8px 16px;font-size:13px;font-weight:600;
    cursor:pointer;color:#94a3b8;background:rgba(30,41,59,.8);
    transition:background .15s,color .15s;user-select:none;
}
.record-dest input[type=radio]{display:none;}
.record-dest input[type=radio]:checked + span{
    background:#1d4ed8;color:#fff;
}
.record-dest label:first-child{border-right:1px solid #334155;}
.record-dest.disabled label{opacity:.45;cursor:not-allowed;}
.video-grid{display:flex;flex-wrap:wrap;justify-content:center;gap:16px;margin-bottom:22px;}
.vcard{
    background:#1e293b;border:1px solid #334155;
    border-radius:12px;overflow:hidden;max-width:640px;width:100%;
}
.vcard-title{padding:7px 14px;font-size:12px;font-weight:600;color:#94a3b8;background:rgba(51,65,85,.4);}
.vcard img{display:block;width:100%;min-height:200px;object-fit:contain;background:#000;}
.metrics{
    display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:10px;max-width:1100px;margin:0 auto 20px;
}
.metric{
    background:rgba(30,41,59,.8);border:1px solid #334155;
    border-left:4px solid #3b82f6;border-radius:8px;padding:12px 14px;
}
.metric.gpu{border-left-color:#a855f7;}
.metric.sys{border-left-color:#22c55e;}
.metric.net{border-left-color:#f59e0b;}
.mlabel{font-size:11px;color:#94a3b8;margin-bottom:4px;}
.mval{font-size:18px;font-weight:700;color:#e5e7eb;word-break:break-all;}
.face-observer{
    max-width:1280px;margin:0 auto 22px;
    background:rgba(15,23,42,.72);border:1px solid #334155;
    border-radius:10px;padding:14px;
}
.face-observer-title{
    display:flex;justify-content:space-between;align-items:center;
    gap:12px;margin-bottom:12px;
}
.face-observer-title h2{font-size:15px;font-weight:700;color:#e5e7eb;}
.face-observer-title span{font-size:12px;color:#94a3b8;}
.face-track-grid{
    display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));
    gap:12px;
}
.face-track-card{
    background:rgba(30,41,59,.88);border:1px solid #334155;
    border-radius:8px;padding:12px;min-width:0;
}
.face-track-head{
    display:flex;justify-content:space-between;align-items:center;
    gap:10px;margin-bottom:10px;font-size:12px;color:#94a3b8;
}
.face-track-head strong{font-size:14px;color:#e5e7eb;}
.face-badge{
    display:inline-flex;align-items:center;gap:6px;
    border:1px solid #0ea5e9;background:rgba(14,165,233,.15);
    color:#7dd3fc;border-radius:999px;padding:3px 8px;font-size:12px;
}
.face-system{
    border-top:1px solid rgba(51,65,85,.8);
    padding-top:10px;margin-top:10px;
}
.face-system-title{
    display:flex;justify-content:space-between;gap:8px;
    font-size:12px;font-weight:700;color:#cbd5e1;margin-bottom:8px;
}
.face-system-title span{font-weight:500;color:#94a3b8;}
.face-row{display:flex;gap:8px;align-items:stretch;overflow-x:auto;padding-bottom:2px;}
.face-crop{
    flex:0 0 88px;min-height:116px;
    background:rgba(2,6,23,.75);border:1px solid #334155;
    border-radius:7px;padding:6px;text-align:center;
}
.face-crop img{
    width:74px;height:74px;object-fit:cover;border-radius:5px;
    background:#020617;border:1px solid rgba(148,163,184,.25);
}
.face-crop .cap{font-size:10px;color:#cbd5e1;margin-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.face-crop .sim{font-size:11px;color:#fbbf24;margin-top:3px;}
.face-quality{
    font-size:11px;color:#94a3b8;line-height:1.6;margin-top:8px;
    word-break:break-word;
}
.face-empty{
    color:#94a3b8;font-size:13px;text-align:center;
    padding:18px;border:1px dashed #334155;border-radius:8px;
}
.statusbar{
    position:fixed;bottom:0;left:0;right:0;
    background:rgba(15,23,42,.97);padding:9px 20px;
    display:flex;justify-content:space-between;align-items:center;
    border-top:1px solid #334155;font-size:12px;color:#94a3b8;z-index:100;
}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle;}
.dot-green{background:#22c55e;}
.dot-red{background:#ef4444;}
#toast{
    position:fixed;top:20px;right:20px;background:#1e293b;border:1px solid #334155;
    padding:12px 18px;border-radius:8px;font-size:13px;opacity:0;transition:opacity .3s;
    pointer-events:none;z-index:200;
}
#toast.show{opacity:1;}
</style>
</head>
<body>
<h1>鱼眼全景 YOLO 姿态检测 — GPU版</h1>
<p class="subtitle">本地摄像头推流 → WebSocket → GPU推理 → 实时显示</p>

<div class="howto">
    <strong>使用步骤：</strong>
    <ol>
        <li>本地安装：<code>pip install opencv-python websockets</code></li>
        <li>运行推流：<code>python camera_client.py ws://<span id="srv-addr">服务器IP:端口</span>/ws/camera</code></li>
        <li>本页面自动显示推理结果</li>
    </ol>
    <div style="margin-top:8px;color:#64748b;font-size:12px;">
        可选：<code>--cam 0 --fps 15 --width 1280 --height 720</code>
    </div>
</div>

<div class="controls">
    <div class="record-dest" id="record-dest">
        <label>
            <input type="radio" name="record-dest" value="server" checked>
            <span>保存在服务器</span>
        </label>
        <label>
            <input type="radio" name="record-dest" value="local">
            <span>保存在本地</span>
        </label>
    </div>
    <button id="btn-record" class="btn-record" onclick="toggleRecord()">开始录制</button>
    <span id="record-badge" class="record-badge" style="display:none;">
        <span class="record-dot"></span><span id="record-badge-text">录制中</span>
    </span>
</div>

<div class="video-grid">
    <div class="vcard">
        <div class="vcard-title">原始画面（本地摄像头）</div>
        <img id="img-original" src="/video/original" alt="等待推流连接...">
    </div>
    <div class="vcard">
        <div class="vcard-title">GPU 推理结果（WebRTC · H.264）</div>
        <video id="video-infer" autoplay playsinline muted
               style="display:block;width:100%;min-height:200px;object-fit:contain;background:#000;"></video>
    </div>
</div>

<div class="metrics">
    <div class="metric"><div class="mlabel">实际处理帧率 (FPS)</div><div class="mval" id="m-fps">—</div></div>
    <div class="metric"><div class="mlabel">理论最高 FPS <span style="font-size:10px;color:#94a3b8">1000/耗时</span></div><div class="mval" id="m-theoretical-fps">—</div></div>
    <div class="metric"><div class="mlabel">帧处理耗时（解码+推理）</div><div class="mval" id="m-infer">—</div></div>
    <div class="metric"><div class="mlabel">检测人数</div><div class="mval" id="m-persons">—</div></div>
    <div class="metric"><div class="mlabel">跟踪 ID</div><div class="mval" id="m-ids" style="font-size:13px;">—</div></div>
    <div class="metric gpu"><div class="mlabel">GPU 使用率</div><div class="mval" id="m-gpu-util">—</div></div>
    <div class="metric gpu"><div class="mlabel">GPU 内存</div><div class="mval" id="m-gpu-mem" style="font-size:14px;">—</div></div>
    <div class="metric gpu"><div class="mlabel">GPU 温度</div><div class="mval" id="m-gpu-temp">—</div></div>
    <div class="metric sys"><div class="mlabel">系统 CPU</div><div class="mval" id="m-cpu">—</div></div>
    <div class="metric sys"><div class="mlabel">系统内存</div><div class="mval" id="m-mem">—</div></div>
    <div class="metric sys"><div class="mlabel">可用内存</div><div class="mval" id="m-mem-avail">—</div></div>
    <div class="metric net"><div class="mlabel">推流客户端</div><div class="mval" id="m-clients">—</div></div>
    <div class="metric" style="border-left-color:#ec4899;"><div class="mlabel">WebRTC 状态</div><div class="mval" id="m-webrtc-state" style="font-size:13px;">连接中…</div></div>
    <div class="metric" style="border-left-color:#ec4899;"><div class="mlabel">浏览器帧率 (FPS)</div><div class="mval" id="m-display-fps">—</div></div>
</div>

<div class="face-observer">
    <div class="face-observer-title">
        <h2>人脸识别特征相似度观察</h2>
        <span id="face-observer-status">等待 FaceRec 数据...</span>
    </div>
    <div id="face-observer-body" class="face-empty">
        暂无人脸识别 crop。只有实际送入 AdaFace 的帧会更新这里。
    </div>
</div>

<div class="statusbar">
    <div><span class="dot" id="stream-dot" style="background:#ef4444;"></span><span id="stream-status">等待推流连接...</span></div>
    <div>最后更新: <span id="last-update">—</span></div>
</div>
<div id="toast"></div>

<script>
document.getElementById('srv-addr').textContent = location.host;

// ── 推理结果 WebRTC 推流（H.264，UDP 低延迟）──────────────────────────
(function initWebRTC() {
    const video = document.getElementById('video-infer');
    let _pc = null;

    // 用 requestVideoFrameCallback 统计真实渲染帧率（Chrome/Edge 支持）
    function startFpsCounter() {
        if (!('requestVideoFrameCallback' in HTMLVideoElement.prototype)) return;
        let fpsCount = 0, fpsTs = performance.now();
        function tick(now) {
            fpsCount++;
            if (now - fpsTs >= 1000) {
                setText('m-display-fps', (fpsCount * 1000 / (now - fpsTs)).toFixed(0));
                fpsCount = 0; fpsTs = now;
            }
            video.requestVideoFrameCallback(tick);
        }
        video.requestVideoFrameCallback(tick);
    }

    async function connect() {
        setText('m-webrtc-state', '信令中…');
        if (_pc) { _pc.close(); _pc = null; }

        const pc = new RTCPeerConnection({ iceServers: [] });
        _pc = pc;

        // 只接收视频，不发送
        pc.addTransceiver('video', { direction: 'recvonly' });

        pc.ontrack = (e) => {
            if (e.track.kind !== 'video') return;
            video.srcObject = e.streams[0];
            setText('m-webrtc-state', '已连接');
            startFpsCounter();
        };

        pc.onconnectionstatechange = () => {
            setText('m-webrtc-state', pc.connectionState);
            if (['failed', 'closed', 'disconnected'].includes(pc.connectionState)) {
                setTimeout(connect, 2000);
            }
        };

        // 创建 Offer 并等待本端 ICE 收集完成（Vanilla ICE，一次往返完成信令）
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        await new Promise(resolve => {
            if (pc.iceGatheringState === 'complete') return resolve();
            const onchange = () => { if (pc.iceGatheringState === 'complete') { resolve(); } };
            pc.addEventListener('icegatheringstatechange', onchange);
            setTimeout(resolve, 3000);   // 3s 超时保底
        });

        // 通过 WebSocket 发送 Offer，接收 Answer
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const sigWs = new WebSocket(`${proto}//${location.host}/ws/webrtc`);
        sigWs.onopen  = () => sigWs.send(JSON.stringify({
            sdp: pc.localDescription.sdp, type: pc.localDescription.type
        }));
        sigWs.onmessage = async (e) => {
            const answer = JSON.parse(e.data);
            await pc.setRemoteDescription(new RTCSessionDescription(answer));
            sigWs.close();   // 信令完成，媒体流走独立 UDP
        };
        sigWs.onerror = () => setTimeout(connect, 2000);
    }

    connect();
})();

let _isRecording        = false;
let _mediaRecorderInfer = null;   // 录推理流（WebRTC）
let _mediaRecorderOrig  = null;   // 录原始流（canvas 捕获 MJPEG img）
let _chunksInfer        = [];
let _chunksOrig         = [];
let _origCanvas         = null;
let _origCapInterval    = null;
let _localRecordTs      = '';
let _recStartTime       = 0;      // 本地录制开始时间（ms），用于修正 WebM Duration

function _getRecordDest() {
    return document.querySelector('input[name="record-dest"]:checked')?.value || 'server';
}

function _setRecordingUI(active, destLabel) {
    const btn   = document.getElementById('btn-record');
    const badge = document.getElementById('record-badge');
    const dest  = document.getElementById('record-dest');
    btn.textContent     = active ? '停止录制' : '开始录制';
    btn.className       = active ? 'btn-record-stop' : 'btn-record';
    badge.style.display = active ? 'inline-flex' : 'none';
    if (active && destLabel)
        document.getElementById('record-badge-text').textContent = `录制中（${destLabel}）`;
    dest.classList.toggle('disabled', active);
    dest.querySelectorAll('input').forEach(i => i.disabled = active);
}

// ── 让浏览器自己扫描 WebM blob 以确定真实时长 ────────────────────────────
// MediaRecorder 记录的 WebM 的 duration 字段为 0 或缺失；
// 将 currentTime 设为极大值可强制浏览器全文件扫描并更新 video.duration。
function _getBlobDuration(blob) {
    return new Promise(resolve => {
        const url = URL.createObjectURL(blob);
        const v   = document.createElement('video');
        v.preload = 'metadata'; v.muted = true; v.src = url;
        let done = false;
        const finish = ms => {
            if (done) return; done = true;
            URL.revokeObjectURL(url); v.src = '';
            resolve(ms);
        };
        const tid = setTimeout(() => finish(0), 12000);
        v.onloadedmetadata = () => {
            if (isFinite(v.duration) && v.duration > 0) { clearTimeout(tid); finish(v.duration * 1000); }
            else v.currentTime = 1e101;   // 触发全文件扫描
        };
        v.onseeked  = () => { clearTimeout(tid); finish(isFinite(v.duration) && v.duration > 0 ? v.duration * 1000 : 0); };
        v.onerror   = () => { clearTimeout(tid); finish(0); };
    });
}

// ── EBML VINT 解析：返回 {v: 值, n: 字节数} ───────────────────────────────
function _vintRead(u8, pos) {
    const b = u8[pos]; let mask = 0x80, len = 1;
    while (mask > 1 && !(b & mask)) { mask >>= 1; len++; }
    let val = b ^ mask;
    for (let k = 1; k < len; k++) val = val * 256 + u8[pos + k];
    return {v: val, n: len};
}

// ── WebM Duration 修正（支持修改现有值 + 不存在时插入）───────────────────
// MediaRecorder 有时写 Duration=0（可直接改），有时完全省略（需插入 11 字节）。
// 两种情况都在这里处理，确保所有播放器的进度条正常工作。
async function _fixWebmDuration(blob, fallbackMs) {
    const durMs = (await _getBlobDuration(blob)) || fallbackMs;
    if (!(durMs > 0)) return blob;
    try {
        const ab  = await blob.arrayBuffer();
        const u8  = new Uint8Array(ab);
        const dv  = new DataView(ab);
        const end = Math.min(u8.length - 12, 1 << 20);   // 最多扫描 1 MB

        // ① Duration 元素已存在（ID=0x4489）：直接覆写值
        for (let i = 0; i < end; i++) {
            if (u8[i] !== 0x44 || u8[i+1] !== 0x89) continue;
            if (u8[i+2] === 0x88) { dv.setFloat64(i+3, durMs, false); return new Blob([ab], {type: blob.type}); }
            if (u8[i+2] === 0x84) { dv.setFloat32(i+3, durMs, false); return new Blob([ab], {type: blob.type}); }
        }

        // ② Duration 元素缺失：在 Info 块（0x1549A966）开头插入 11 字节
        for (let i = 0; i < end; i++) {
            if (u8[i] !== 0x15 || u8[i+1] !== 0x49 || u8[i+2] !== 0xA9 || u8[i+3] !== 0x66) continue;
            const sv   = _vintRead(u8, i + 4);
            const body = i + 4 + sv.n;          // Info 内容起始偏移

            // 构造 Duration 元素：[44 89 88] + float64
            const durEl = new Uint8Array(11);
            durEl[0] = 0x44; durEl[1] = 0x89; durEl[2] = 0x88;
            new DataView(durEl.buffer).setFloat64(3, durMs, false);

            // 更新 Info 的 VINT size（保持原字节宽度）
            const masks = [0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01];
            const newSz = sv.v + 11;
            const szBuf = new Uint8Array(sv.n);
            szBuf[0] = masks[sv.n - 1] | (newSz >> (8 * (sv.n - 1)));
            for (let k = 1; k < sv.n; k++) szBuf[k] = (newSz >> (8 * (sv.n - 1 - k))) & 0xFF;

            // 拼接：头部 + 新 size + Duration元素 + 原 Info 内容
            const out = new Uint8Array(u8.length + 11);
            out.set(u8.subarray(0, i + 4));
            out.set(szBuf, i + 4);
            out.set(durEl, i + 4 + sv.n);
            out.set(u8.subarray(body), i + 4 + sv.n + 11);
            return new Blob([out], {type: blob.type});
        }
        console.warn('[录制] 未找到 EBML Info 块，Duration 修正跳过');
    } catch(e) { console.warn('[录制] Duration 修正异常:', e); }
    return blob;
}

function _downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a   = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 2000);
}

// ── 服务器录制 ─────────────────────────────────────────────────────────
async function _serverStart() {
    const r = await fetch('/record/start', {method: 'POST'});
    const j = await r.json();
    if (j.msg === 'started' || j.msg === 'already recording') {
        _isRecording = true;
        _setRecordingUI(true, '服务器');
        showToast('录制已开始 — 原始 + 推理双路保存至服务器');
    } else {
        showToast('录制启动失败: ' + j.msg);
    }
}

async function _serverStop() {
    const r = await fetch('/record/stop', {method: 'POST'});
    const j = await r.json();
    _isRecording = false;
    _setRecordingUI(false);
    const f = j.filenames || {};
    showToast(`已停止 → ${f.original || '?'} | ${f.annotated || '?'}`);
}

// ── 本地录制（原始 + 推理双路，完成后各自触发浏览器下载）──────────────
// 原始流：将 MJPEG <img> 每帧绘入 canvas，再用 canvas.captureStream() 录制
// 推理流：直接用 MediaRecorder 录 WebRTC srcObject

function _origCanvasStart() {
    const img = document.getElementById('img-original');
    if (!_origCanvas) {
        _origCanvas = document.createElement('canvas');
        _origCanvas.style.display = 'none';
        document.body.appendChild(_origCanvas);
    }
    // 先用默认尺寸，首帧绘制时按实际尺寸自动更新
    _origCanvas.width  = img.naturalWidth  || 1920;
    _origCanvas.height = img.naturalHeight || 1080;
    const ctx = _origCanvas.getContext('2d');

    _origCapInterval = setInterval(() => {
        if (!img.complete || !img.naturalWidth) return;
        if (_origCanvas.width  !== img.naturalWidth ||
            _origCanvas.height !== img.naturalHeight) {
            _origCanvas.width  = img.naturalWidth;
            _origCanvas.height = img.naturalHeight;
        }
        ctx.drawImage(img, 0, 0);
    }, 40);   // ~25 fps

    return _origCanvas.captureStream(25);
}

function _origCanvasStop() {
    if (_origCapInterval) { clearInterval(_origCapInterval); _origCapInterval = null; }
}

function _localStart() {
    const video = document.getElementById('video-infer');
    if (!video.srcObject) { showToast('WebRTC 流未就绪，请等待连接'); return; }

    _chunksInfer = []; _chunksOrig = [];
    // 优先 mp4：Chrome(Windows/Mac) 输出 fMP4，原生支持进度条拖拽，无需修正。
    // 回退 webm：Linux Chrome / Firefox，录制完成后由 _fixWebmDuration 修正 Duration。
    const mimeType =
        MediaRecorder.isTypeSupported('video/mp4;codecs=h264,aac')   ? 'video/mp4'  :
        MediaRecorder.isTypeSupported('video/mp4;codecs=avc1,mp4a')  ? 'video/mp4'  :
        MediaRecorder.isTypeSupported('video/mp4')                   ? 'video/mp4'  :
        MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus')  ? 'video/webm' :
        MediaRecorder.isTypeSupported('video/webm;codecs=vp8,opus')  ? 'video/webm' :
        'video/webm';
    const ext        = mimeType.includes('mp4') ? 'mp4' : 'webm';
    const needFix    = ext === 'webm';   // mp4(fMP4) 原生可拖，webm 需修正 Duration
    _localRecordTs   = new Date().toISOString().replace(/[:.]/g, '-');
    _recStartTime    = Date.now();

    // 推理流录制
    try {
        _mediaRecorderInfer = new MediaRecorder(video.srcObject, {mimeType});
    } catch(e) { showToast('MediaRecorder 初始化失败: ' + e); return; }
    _mediaRecorderInfer.ondataavailable = e => { if (e.data?.size > 0) _chunksInfer.push(e.data); };
    _mediaRecorderInfer.onstop = async () => {
        const raw   = new Blob(_chunksInfer, {type: mimeType});
        const fixed = needFix ? await _fixWebmDuration(raw, Date.now() - _recStartTime) : raw;
        _downloadBlob(fixed, `annotated_${_localRecordTs}.${ext}`);
        showToast('推理视频已下载');
    };

    // 原始流录制（canvas 捕获 MJPEG img）
    const origStream = _origCanvasStart();
    try {
        _mediaRecorderOrig = new MediaRecorder(origStream, {mimeType});
        _mediaRecorderOrig.ondataavailable = e => { if (e.data?.size > 0) _chunksOrig.push(e.data); };
        _mediaRecorderOrig.onstop = async () => {
            _origCanvasStop();
            const raw   = new Blob(_chunksOrig, {type: mimeType});
            const fixed = needFix ? await _fixWebmDuration(raw, Date.now() - _recStartTime) : raw;
            _downloadBlob(fixed, `original_${_localRecordTs}.${ext}`);
            showToast('原始视频已下载');
        };
        _mediaRecorderOrig.start(1000);
    } catch(e) {
        _origCanvasStop();
        _mediaRecorderOrig = null;
        showToast('原始流捕获失败，仅录推理画面');
    }

    _mediaRecorderInfer.start(1000);
    _isRecording = true;
    _setRecordingUI(true, '本地');
    showToast('录制已开始 — 原始 + 推理双路保存至本地');
}

function _localStop() {
    if (_mediaRecorderInfer?.state !== 'inactive') _mediaRecorderInfer.stop();
    if (_mediaRecorderOrig?.state  !== 'inactive') _mediaRecorderOrig.stop();
    else _origCanvasStop();
    _isRecording = false;
    _setRecordingUI(false);
}

// ── 统一入口 ───────────────────────────────────────────────────────────
async function toggleRecord() {
    const btn = document.getElementById('btn-record');
    btn.disabled = true;
    try {
        const dest = _getRecordDest();
        if (!_isRecording) {
            dest === 'local' ? _localStart() : await _serverStart();
        } else {
            dest === 'local' ? _localStop() : await _serverStop();
        }
    } catch(e) {
        showToast('操作失败: ' + e);
    } finally {
        btn.disabled = false;
    }
}

function _escHtml(v) {
    return String(v ?? '').replace(/[&<>"']/g, c => ({
        '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'
    }[c]));
}

function _simText(v) {
    return (typeof v === 'number' && Number.isFinite(v)) ? v.toFixed(3) : '—';
}

function _cropCard(title, image, similarity) {
    const imgHtml = image
        ? `<img src="${image}" alt="${_escHtml(title)}">`
        : `<div style="width:74px;height:74px;border-radius:5px;background:#020617;border:1px dashed #334155;"></div>`;
    return `
        <div class="face-crop">
            ${imgHtml}
            <div class="cap">${_escHtml(title)}</div>
            <div class="sim">sim ${_simText(similarity)}</div>
        </div>`;
}

function _qualityLine(q, topScores) {
    q = q || {};
    const flags = [
        q.trigger_ok ? 'trigger_ok' : 'trigger_no',
        q.sample_ok ? 'sample_ok' : 'sample_no',
        q.primary_ok ? 'primary_ok' : 'primary_no',
        q.pose_sample_quality_ok ? 'pose_ok' : 'pose_no',
        q.locked_pose_quality_ok ? 'locked_pose_ok' : ''
    ].filter(Boolean).join(' / ');
    const base = [
        `质量: ${flags}`,
        `reason=${_escHtml(q.reason || '—')}`,
        `sample=${_escHtml(q.sample_reason || '—')}`,
        `primary=${_escHtml(q.primary_reason || '—')}`,
        `update=${_escHtml(q.update_reason || '—')}`,
        `side=${_simText(q.min_side)}`,
        `kp=${_simText(q.min_keypoint_conf)}`,
        `yaw=${q.yaw_valid ? _simText(q.yaw_deg) + '°' : '—'}`,
        `same=${_simText(q.best_existing)}`,
        `other=${_simText(q.best_other)}`,
        `need=${_simText(q.sample_similarity)}`,
        `pose_same=${_simText(q.pose_library_similarity)}`,
        `pose_need=${_simText(q.pose_library_threshold)}`,
        `pose_max=${_simText(q.pose_library_max_similarity)}`,
        `pose_gap=${q.pose_sample_missing_frames ?? '—'}/${q.pose_sample_interval ?? '—'}`
    ].join(' ｜ ');
    const tops = Array.isArray(topScores) && topScores.length
        ? ' ｜ Top: ' + topScores.map(s => `${_escHtml(s.face_id)}=${_simText(s.score)}`).join(', ')
        : '';
    return base + tops;
}

function renderFaceObserver(observer) {
    const status = document.getElementById('face-observer-status');
    const body = document.getElementById('face-observer-body');
    if (!status || !body) return;

    if (!observer || observer.enabled === false) {
        status.textContent = 'FaceRec 观察面板未启用';
        body.className = 'face-empty';
        body.textContent = '启动时加入 --face-rec-observer 可开启实时相似度观察。';
        return;
    }

    const tracks = Array.isArray(observer.tracks) ? observer.tracks : [];
    if (!tracks.length) {
        status.textContent = '等待 AdaFace 输入帧...';
        body.className = 'face-empty';
        body.textContent = '暂无人脸识别 crop。只有实际送入 AdaFace 的帧会更新这里。';
        return;
    }

    status.textContent = `显示最近 ${tracks.length} 个 FaceID / 待建库样本的 AdaFace 输入`;
    body.className = 'face-track-grid';
    body.innerHTML = tracks.map(t => {
        const anchor = t.frontal_anchor || null;
        const gallery = t.gallery || {};
        const primary = Array.isArray(gallery.primary) ? gallery.primary : [];
        const supplement = Array.isArray(gallery.supplement) ? gallery.supplement : [];
        const displayFace = t.target_face_id || t.face_id || t.raw_face_id || '待建库';
        const currentCrop = _cropCard('实时人脸', t.current_crop, null);
        const anchorHtml = anchor
            ? _cropCard(`正脸基准 ${anchor.face_id || ''}`, anchor.crop, anchor.similarity)
            : '<div class="face-empty" style="padding:12px;min-width:180px;">还没有符合主特征质量条件的正脸基准</div>';
        const galleryCards = [
            currentCrop,
            ...primary.map((s, i) => _cropCard(`主特征 ${i + 1}`, s.crop, s.similarity)),
            ...supplement.map((s, i) => _cropCard(`姿态 ${i + 1}`, s.crop, s.similarity))
        ].join('');
        const galleryEmpty = (!primary.length && !supplement.length)
            ? '<div class="face-empty" style="padding:12px;min-width:180px;">当前 FaceID 还没有可显示的特征库样本</div>'
            : '';
        return `
            <div class="face-track-card">
                <div class="face-track-head">
                    <strong>Face ${_escHtml(displayFace)}</strong>
                    <span class="face-badge">最近 Track ${_escHtml(t.track_id)}</span>
                </div>
                <div class="face-quality">
                    frame=${_escHtml(t.frame_id)} ｜ event=${_escHtml(t.event || '—')} ｜
                    score=${_simText(t.score)} ｜ second=${_simText(t.second_score)}
                </div>
                <div class="face-system">
                    <div class="face-system-title">
                        <div>方案一：最正脸基准对比</div>
                        <span>当前 crop vs 该 FaceID 最正脸样本</span>
                    </div>
                    <div class="face-row">${currentCrop}${anchorHtml}</div>
                </div>
                <div class="face-system">
                    <div class="face-system-title">
                        <div>方案二：主特征 + 多姿态样本</div>
                        <span>当前 crop vs 主特征/姿态样本</span>
                    </div>
                    <div class="face-row">${galleryCards}${galleryEmpty}</div>
                </div>
                <div class="face-quality">${_qualityLine(t.quality, t.top_scores)}</div>
            </div>`;
    }).join('');
}

let _faceObserverWs = null;
async function _parseInferencePayload(data) {
    if (typeof data === 'string') return JSON.parse(data);
    if (data instanceof ArrayBuffer) {
        return JSON.parse(new TextDecoder('utf-8').decode(data));
    }
    if (data instanceof Blob) {
        return JSON.parse(await data.text());
    }
    return null;
}

function initFaceObserverWs() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws/inference`;
    try {
        _faceObserverWs = new WebSocket(url);
        _faceObserverWs.binaryType = 'arraybuffer';
        _faceObserverWs.onmessage = async e => {
            try {
                const payload = await _parseInferencePayload(e.data);
                renderFaceObserver(payload && payload.face_rec_observer);
            } catch (_) {}
        };
        _faceObserverWs.onclose = () => setTimeout(initFaceObserverWs, 2000);
        _faceObserverWs.onerror = () => { try { _faceObserverWs.close(); } catch (_) {} };
    } catch (_) {
        setTimeout(initFaceObserverWs, 2000);
    }
}

async function updateMetrics() {
    try {
        const r = await fetch('/performance');
        const d = await r.json();
        setText('m-fps',             d.fps.toFixed(1));
        setText('m-theoretical-fps', d.theoretical_fps.toFixed(1));
        setText('m-infer',           d.inference_time_ms.toFixed(0) + ' ms');
        setText('m-persons',  d.detected_persons);
        setText('m-ids',      d.tracking_ids?.length ? d.tracking_ids.join(', ') : '无');
        setText('m-gpu-util', d.gpu_usage);
        setText('m-gpu-mem',  d.gpu_memory);
        setText('m-gpu-temp', d.gpu_temp);
        setText('m-cpu',      d.system_cpu.toFixed(1) + '%');
        setText('m-mem',      d.system_memory.toFixed(1) + '%');
        // 可用内存：低于阈值时显示红色警告
        const availMb = d.system_memory_avail_mb ?? 0;
        const availEl = document.getElementById('m-mem-avail');
        if (availEl) {
            availEl.textContent = availMb.toFixed(0) + ' MB';
            availEl.style.color = availMb < 500 ? '#ef4444' : (availMb < 1024 ? '#f59e0b' : '');
        }
        setText('m-clients',     d.connected_clients);
        // WebRTC 状态由 RTCPeerConnection.onconnectionstatechange 直接更新，此处不轮询
        setText('last-update', new Date().toLocaleTimeString());
        const dot = document.getElementById('stream-dot');
        const txt = document.getElementById('stream-status');
        if (d.connected_clients > 0) {
            dot.className = 'dot dot-green';
            txt.textContent = `推流中（${d.connected_clients} 个客户端）`;
        } else {
            dot.className = 'dot dot-red';
            txt.textContent = '等待推流连接...';
        }
    } catch(_) {}
}

function setText(id, val) { const el = document.getElementById(id); if(el) el.textContent = val; }

let toastTimer = null;
function showToast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg; t.classList.add('show');
    if(toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove('show'), 3500);
}

setInterval(updateMetrics, 1000);
updateMetrics();
initFaceObserverWs();
</script>
</body>
</html>"""
