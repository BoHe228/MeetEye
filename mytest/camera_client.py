"""
本地摄像头推流客户端（零编解码）

数据链路：
  相机硬件 MJPEG
    → FFmpeg (-c:v copy，直通不重编码) → stdout
    → Python 按 FFD8/FFD9 切割完整 JPEG 帧（最新帧优先，丢弃积压旧帧）
    → WebSocket 发往服务器 /ws/camera

用法:
    python camera_client.py ws://<server-ip>:<port>/ws/camera
    python camera_client.py ws://192.168.1.100:8000/ws/camera --cam 1 --fps 30 --width 1920 --height 1080

    # 列出 Windows 可用摄像头
    ffmpeg -list_devices true -f dshow -i dummy

依赖:
    ffmpeg 已安装并在系统 PATH 中
    pip install websockets
"""
import argparse
import asyncio
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse


# ── 服务器在线检测 ─────────────────────────────────────────────────────

def check_server_http(ws_url: str) -> bool:
    parsed = urlparse(ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    base_url = f"{scheme}://{parsed.netloc}/"
    print(f"[检测] HTTP 预检: {base_url} ...")
    try:
        req = urllib.request.Request(base_url, headers={"User-Agent": "camera-client/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"[检测] 服务器在线 (HTTP {resp.status})")
            return True
    except urllib.error.HTTPError as e:
        if e.code < 500:
            print(f"[检测] 服务器在线 (HTTP {e.code})")
            return True
        print(f"[检测] 服务器错误 HTTP {e.code}")
        return False
    except urllib.error.URLError as e:
        reason = getattr(e, 'reason', str(e))
        host = parsed.hostname
        port = parsed.port or 8000
        print(f"\n❌ 无法连接到服务器 {base_url}")
        print(f"   错误: {reason}")
        print(f"\n请依次排查：")
        print(f"   1. Linux 服务器上是否已运行:  python main_GPU_webui.py --webui")
        print(f"   2. IP 地址是否正确: {host}")
        print(f"   3. 端口 {port} 是否未被防火墙屏蔽")
        print(f"      Linux 开放端口命令: sudo ufw allow {port}")
        print(f"   4. 两台电脑是否在同一局域网")
        return False
    except Exception as e:
        print(f"[检测] 预检异常: {type(e).__name__}: {e}")
        return False


# ── FFmpeg 设备枚举与命令构建 ──────────────────────────────────────────

def _list_dshow_devices() -> list:
    """列出 Windows DirectShow 视频设备名列表。"""
    r = subprocess.run(
        ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy'],
        capture_output=True, text=True, timeout=8,
        encoding='utf-8', errors='replace',
    )
    devices = []
    for line in r.stderr.splitlines():
        if 'Alternative name' in line:
            continue
        if '(video)' in line:
            m = re.search(r'"([^"]+)"', line)
            if m:
                devices.append(m.group(1))
    if devices:
        return devices
    # 旧版 FFmpeg 回退
    in_video = False
    for line in r.stderr.splitlines():
        if 'DirectShow video devices' in line:
            in_video = True
        elif 'DirectShow audio devices' in line:
            break
        elif in_video and 'Alternative name' not in line:
            m = re.search(r'"([^"]+)"', line)
            if m:
                devices.append(m.group(1))
    return devices


def _build_ffmpeg_cmd(cam_index: int, width: int, height: int, fps: float) -> list:
    """
    构建 FFmpeg 零编解码命令：摄像头原生 MJPEG → stdout。
      -c:v copy  不解码、不重编码，原始 JPEG 字节直出
      -f mjpeg   输出格式为连续 JPEG 帧流
      pipe:1     写入 stdout 供 Python 读取
    """
    size = f'{width}x{height}'
    rate = str(int(fps))

    if sys.platform.startswith('linux'):
        device = f'/dev/video{cam_index}'
        input_args = [
            '-f', 'v4l2',
            '-input_format', 'mjpeg',
            '-video_size', size,
            '-framerate', rate,
            '-i', device,
        ]
        print(f"📷 Linux V4L2 设备: {device}")

    elif sys.platform == 'win32':
        try:
            devices = _list_dshow_devices()
        except FileNotFoundError:
            raise RuntimeError("未找到 ffmpeg，请安装并添加到 PATH")
        if not devices:
            raise RuntimeError("未检测到 DirectShow 视频设备")
        if cam_index >= len(devices):
            raise RuntimeError(
                f"摄像头索引 {cam_index} 超出范围，可用设备:\n  "
                + "\n  ".join(f"[{i}] {d}" for i, d in enumerate(devices))
            )
        name = devices[cam_index]
        print(f"📷 Windows DirectShow 设备 [{cam_index}]: {name}")
        input_args = [
            '-f', 'dshow',
            '-rtbufsize', '200M',
            '-vcodec', 'mjpeg',
            '-video_size', size,
            '-framerate', rate,
            '-i', f'video={name}',
        ]

    else:
        raise RuntimeError(f"不支持的平台: {sys.platform}")

    return [
        'ffmpeg', *input_args,
        '-c:v', 'copy',
        '-f', 'mjpeg',
        'pipe:1',
    ]


async def _drain_ffmpeg_stderr(proc) -> None:
    """将 FFmpeg stderr 实时转发到控制台，便于排查设备警告。"""
    async for line in proc.stderr:
        msg = line.decode('utf-8', errors='replace').rstrip()
        if msg:
            print(f'[ffmpeg] {msg}')


# ── 主推流协程 ────────────────────────────────────────────────────────

async def run_stream(ws_url: str, cam_index: int, fps: float,
                     width: int, height: int, retry_sec: float) -> None:
    try:
        import websockets
        import websockets.exceptions
    except ImportError:
        print("❌ 未找到 websockets 库，请先运行: pip install websockets")
        return

    if not check_server_http(ws_url):
        return

    try:
        cmd = _build_ffmpeg_cmd(cam_index, width, height, fps)
    except RuntimeError as e:
        print(f"❌ {e}")
        return

    print(f"🎥 FFmpeg: {' '.join(cmd)}")

    while True:   # 外层断线重连
        proc = None
        try:
            print(f"🔗 连接 WebSocket: {ws_url}")
            async with websockets.connect(
                ws_url,
                max_size=30 * 1024 * 1024,
                open_timeout=10,
                ping_interval=None,
                ping_timeout=None,
            ) as ws:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                asyncio.create_task(_drain_ffmpeg_stderr(proc))
                print("✅ 已连接，FFmpeg 零编解码推流中（Ctrl+C 停止）...")

                buf     = b''
                frame_n = 0
                t0      = time.time()

                while True:
                    # 512KB 单次读取，通常一次可读完整帧（~150KB），减少 await 轮转
                    chunk = await proc.stdout.read(524288)
                    if not chunk:
                        print("⚠️  FFmpeg 输出结束（摄像头断开或不支持原生 MJPEG）")
                        break
                    buf += chunk

                    # 从缓冲区提取所有完整帧，只保留最新帧，丢弃积压旧帧
                    latest_jpeg = None
                    while True:
                        s = buf.find(b'\xff\xd8')
                        if s == -1:
                            buf = b''
                            break
                        e = buf.find(b'\xff\xd9', s + 2)
                        if e == -1:
                            buf = buf[s:]   # 保留不完整帧头等待下次读取
                            break
                        latest_jpeg = buf[s:e + 2]   # 覆盖，循环结束后为最新帧
                        buf = buf[e + 2:]

                    if latest_jpeg is not None:
                        await ws.send(latest_jpeg)
                        frame_n += 1
                        if frame_n == 1:
                            n_pkts = len(latest_jpeg) // 1024
                            print(f"✅ 首帧已发  {n_pkts} KB")
                        elif frame_n % 150 == 0:
                            elapsed = time.time() - t0
                            print(f"  已推送 {frame_n} 帧，均速 {frame_n/elapsed:.1f} fps  "
                                  f"{len(latest_jpeg)//1024} KB/帧")

        except KeyboardInterrupt:
            print("\n🛑 用户中断，停止推流")
            return

        except websockets.exceptions.ConnectionClosedError as e:
            print(f"🔌 WebSocket 断开: {e}  → {retry_sec}s 后重连...")

        except websockets.exceptions.SecurityError as e:
            print(f"\n❌ WebSocket 安全错误: {e}")
            print("   若页面跳转到 HTTPS，请将 ws:// 改为 wss://")
            return

        except (ConnectionRefusedError, OSError) as e:
            print(f"❌ 网络连接失败: {e}  → {retry_sec}s 后重连...")

        except asyncio.TimeoutError:
            print(f"❌ 连接超时（10s）→ {retry_sec}s 后重连...")

        except Exception as e:
            print(f"❌ 推流异常: {type(e).__name__}: {e}  → {retry_sec}s 后重连...")

        finally:
            if proc and proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    proc.kill()

        await asyncio.sleep(retry_sec)


# ── 入口 ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="鱼眼全景 GPU WebUI — 本地摄像头推流客户端（零编解码）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python camera_client.py ws://192.168.1.100:8000/ws/camera
  python camera_client.py ws://192.168.1.100:8000/ws/camera --cam 1 --fps 30
  python camera_client.py ws://192.168.1.100:8000/ws/camera --width 1920 --height 1080

  # 列出 Windows 可用摄像头索引
  ffmpeg -list_devices true -f dshow -i dummy
""",
    )
    p.add_argument("url",    nargs="?", default="ws://localhost:8000/ws/camera",
                             help="服务器 WebSocket 地址")
    p.add_argument("--cam",  type=int,   default=0,    help="摄像头索引（默认 0）")
    p.add_argument("--fps",  type=float, default=30.0, help="帧率（默认 30）")
    p.add_argument("--width",  type=int, default=1920, help="分辨率宽（默认 1920）")
    p.add_argument("--height", type=int, default=1080, help="分辨率高（默认 1080）")
    p.add_argument("--retry",  type=float, default=3.0, help="断线重连等待秒数（默认 3）")
    args = p.parse_args()

    print("=" * 56)
    print("  摄像头推流客户端（FFmpeg 零编解码）")
    print(f"  服务器  : {args.url}")
    print(f"  摄像头  : {args.cam}   FPS: {args.fps}   分辨率: {args.width}x{args.height}")
    print("=" * 56)

    asyncio.run(run_stream(
        args.url, args.cam, args.fps,
        args.width, args.height, args.retry,
    ))


if __name__ == "__main__":
    main()
