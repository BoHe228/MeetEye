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
.btn-snap{background:#3b82f6;color:#fff;}
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
<p class="subtitle">本地摄像头推流（WebSocket / UDP）→ GPU推理 → 实时显示</p>

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
    <button class="btn-snap" onclick="takeSnapshot()">截图（保存到服务端）</button>
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
    <div class="metric net"><div class="mlabel">推流客户端</div><div class="mval" id="m-clients">—</div></div>
    <div class="metric" style="border-left-color:#06b6d4;"><div class="mlabel">上传方式</div><div class="mval" id="m-upload-mode" style="font-size:14px;">—</div></div>
    <div class="metric" style="border-left-color:#ec4899;"><div class="mlabel">WebRTC 状态</div><div class="mval" id="m-webrtc-state" style="font-size:13px;">连接中…</div></div>
    <div class="metric" style="border-left-color:#ec4899;"><div class="mlabel">浏览器帧率 (FPS)</div><div class="mval" id="m-display-fps">—</div></div>
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

async function takeSnapshot() {
    try {
        const r = await fetch('/snapshot');
        const j = await r.json();
        showToast(j.filename ? `截图已保存: ${j.filename}` : (j.msg || '截图失败'));
    } catch(e) { showToast('截图请求失败'); }
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
        setText('m-clients',     d.connected_clients);
        setText('m-upload-mode', d.upload_mode === 'udp' ? 'UDP' : 'WebSocket');
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
</script>
</body>
</html>"""
