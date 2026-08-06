"""
RK3588 board camera push client.

Data path:
  /dev/video* -> FFmpeg V4L2 -> JPEG byte stream -> WebSocket /ws/camera

Examples:
  python board_camera_client.py ws://SERVER:8000/ws/camera --device /dev/video0
  python board_camera_client.py ws://SERVER:8000/ws/camera --device /dev/video0 --width 1920 --height 1080 --fps 30
  python board_camera_client.py ws://SERVER:8000/ws/camera --device /dev/video0 --input-format yuyv422 --reencode
"""
import argparse
import asyncio
import subprocess
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse


def check_server_http(ws_url: str) -> bool:
    parsed = urlparse(ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    base_url = f"{scheme}://{parsed.netloc}/"
    print(f"[check] HTTP preflight: {base_url}")
    try:
        req = urllib.request.Request(base_url, headers={"User-Agent": "board-camera-client/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"[check] server online (HTTP {resp.status})")
            return True
    except urllib.error.HTTPError as exc:
        if exc.code < 500:
            print(f"[check] server online (HTTP {exc.code})")
            return True
        print(f"[check] server error HTTP {exc.code}")
        return False
    except Exception as exc:
        print(f"[check] cannot reach {base_url}: {exc}")
        return False


def build_ffmpeg_cmd(args) -> list:
    size = f"{int(args.width)}x{int(args.height)}"
    rate = str(int(args.fps))
    input_format = str(args.input_format or "mjpeg").lower()

    input_args = [
        "-f", "v4l2",
        "-input_format", input_format,
        "-video_size", size,
        "-framerate", rate,
        "-i", args.device,
    ]

    if args.reencode or input_format != "mjpeg":
        return [
            "ffmpeg", *input_args,
            "-an",
            "-vcodec", "mjpeg",
            "-q:v", str(int(args.quality)),
            "-f", "mjpeg",
            "pipe:1",
        ]

    return [
        "ffmpeg", *input_args,
        "-c:v", "copy",
        "-f", "mjpeg",
        "pipe:1",
    ]


async def drain_stderr(proc) -> None:
    async for line in proc.stderr:
        msg = line.decode("utf-8", errors="replace").rstrip()
        if msg:
            print(f"[ffmpeg] {msg}")


async def run_stream(args) -> None:
    try:
        import websockets
        import websockets.exceptions
    except ImportError:
        print("missing dependency: pip install websockets")
        return

    if not check_server_http(args.url):
        return

    cmd = build_ffmpeg_cmd(args)
    print("=" * 64)
    print("  RK3588 camera push client")
    print(f"  server : {args.url}")
    print(f"  device : {args.device}")
    print(f"  format : {args.input_format}  {args.width}x{args.height}@{args.fps:g}")
    print(f"  mode   : {'reencode' if args.reencode or args.input_format != 'mjpeg' else 'copy'}")
    print(f"  ffmpeg : {' '.join(cmd)}")
    print("=" * 64)

    while True:
        proc = None
        try:
            print(f"[ws] connecting: {args.url}")
            async with websockets.connect(
                args.url,
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
                asyncio.create_task(drain_stderr(proc))

                buf = b""
                frame_n = 0
                t0 = time.time()
                print("[ws] connected, streaming JPEG frames. Ctrl+C to stop.")

                while True:
                    chunk = await proc.stdout.read(524288)
                    if not chunk:
                        print("[ffmpeg] stdout ended")
                        break
                    buf += chunk

                    latest_jpeg = None
                    while True:
                        start = buf.find(b"\xff\xd8")
                        if start < 0:
                            buf = b""
                            break
                        end = buf.find(b"\xff\xd9", start + 2)
                        if end < 0:
                            buf = buf[start:]
                            break
                        latest_jpeg = buf[start:end + 2]
                        buf = buf[end + 2:]

                    if latest_jpeg is None:
                        continue

                    await ws.send(latest_jpeg)
                    frame_n += 1
                    if frame_n == 1:
                        print(f"[stream] first frame sent: {len(latest_jpeg) // 1024} KB")
                    elif frame_n % int(args.log_interval) == 0:
                        elapsed = max(time.time() - t0, 1e-6)
                        print(
                            f"[stream] sent={frame_n} avg={frame_n / elapsed:.1f} fps "
                            f"last={len(latest_jpeg) // 1024} KB"
                        )

        except KeyboardInterrupt:
            print("\n[stream] stopped by user")
            return
        except websockets.exceptions.ConnectionClosedError as exc:
            print(f"[ws] disconnected: {exc}; retry in {args.retry:g}s")
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError) as exc:
            print(f"[ws] connection failed: {exc}; retry in {args.retry:g}s")
        except Exception as exc:
            print(f"[stream] error {type(exc).__name__}: {exc}; retry in {args.retry:g}s")
        finally:
            if proc and proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3)
                except asyncio.TimeoutError:
                    proc.kill()

        await asyncio.sleep(float(args.retry))


def main() -> None:
    parser = argparse.ArgumentParser(description="RK3588 board camera WebSocket push client")
    parser.add_argument("url", help="server WebSocket URL, e.g. ws://SERVER:8000/ws/camera")
    parser.add_argument("--device", default="/dev/video0", help="V4L2 camera device")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--input-format", default="mjpeg",
                        help="V4L2 input format, e.g. mjpeg or yuyv422")
    parser.add_argument("--reencode", action="store_true",
                        help="convert non-MJPEG input to MJPEG before sending")
    parser.add_argument("--quality", type=int, default=4,
                        help="MJPEG quality for --reencode; smaller is higher quality")
    parser.add_argument("--retry", type=float, default=3.0)
    parser.add_argument("--log-interval", type=int, default=150)
    args = parser.parse_args()

    asyncio.run(run_stream(args))


if __name__ == "__main__":
    main()
