import asyncio
import websockets
import logging
import struct
import os
import time
import threading
import cv2
from queue import Queue, Empty
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ==================== 默认配置 ====================
SAVE_DIR = "VGDSE/data"
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 30
AUDIO_BITS_PER_SAMPLE = 16


# ==================== 摄像头工具函数 ====================
def list_cameras(max_check=10):
    """列出可用的摄像头"""
    available = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            cap.release()
        else:
            cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
            if cap.isOpened():
                available.append(i)
                cap.release()
    return available


def get_camera_info(camera_id):
    """获取摄像头支持的分辨率和帧率"""
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id, cv2.CAP_MSMF)
        if not cap.isOpened():
            return None

    info = {
        'id': camera_id,
        'default_width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'default_height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'default_fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'backend': cap.getBackendName()
    }
    cap.release()
    return info


# ==================== 视频写入线程 ====================
class VideoWriterThread(threading.Thread):
    """视频写入线程，异步处理视频编码和写入"""

    def __init__(self, queue, filename, fps, frame_size):
        super().__init__(daemon=True)
        self.queue = queue
        self.filename = filename
        self.fps = fps
        self.frame_size = frame_size
        self.running = True
        self.writer = None
        self.frame_count = 0
        self.start_time = None

    def run(self):
        """线程主循环"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, self.frame_size)

        if not self.writer.isOpened():
            logger.error(f"❌ 视频写入器初始化失败: {self.filename}")
            self.running = False
            return

        self.start_time = time.time()
        logger.info(f"🎬 视频录制已开始: {self.filename}")

        while self.running:
            try:
                frame = self.queue.get(timeout=0.1)
                if frame is None:
                    break

                self.writer.write(frame)
                self.frame_count += 1
                self.queue.task_done()

            except Empty:
                continue
            except Exception as e:
                logger.error(f"视频写入错误: {e}")
                break

        if self.writer and self.writer.isOpened():
            self.writer.release()

    def stop(self):
        self.running = False
        if self.is_alive():
            self.join(timeout=2.0)


# ==================== WAV 头生成 ====================
def create_wav_header(data_size, sample_rate=48000, channels=30, bits_per_sample=16):
    """生成标准 44 字节 WAV 文件头"""
    chunk_size = 36 + data_size
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    fmt_data = struct.pack('<HHIIHH', 1, channels, sample_rate, byte_rate, block_align, bits_per_sample)

    header = (
        b'RIFF' + struct.pack('<I', chunk_size) + b'WAVE' +
        b'fmt ' + struct.pack('<I', 16) + fmt_data +
        b'data' + struct.pack('<I', data_size)
    )
    return header


# ==================== 共享状态 ====================
class SharedState:
    def __init__(self):
        self.start_time = None
        self.stop_time = None
        self.save_audio = False  # 控制是否开始保存音频


# ==================== 音频录制任务 ====================
async def record_audio(uri, output_file, stats_dict, ready_event, shared_state):
    """录制音频"""
    received_bytes = 0
    discarded_bytes = 0

    f = open(output_file, 'wb')
    try:
        f.write(create_wav_header(0))

        async with websockets.connect(uri) as ws:
            logger.info(f"🎙 音频已连接")
            ready_event.set()

            # 先接收但丢弃音频，等待视频准备好
            logger.info(f"🎙 音频预热中（暂时丢弃数据）...")
            while not shared_state.save_audio:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    if isinstance(message, bytes):
                        discarded_bytes += len(message)
                except asyncio.TimeoutError:
                    continue

            logger.info(f"🎙 音频开始录制（已丢弃 {discarded_bytes/1024/1024:.2f} MB）")

            try:
                while time.time() < shared_state.stop_time:
                    remaining = shared_state.stop_time - time.time()
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=min(0.1, remaining))
                        if isinstance(message, bytes):
                            f.write(message)
                            received_bytes += len(message)
                            stats_dict['audio_bytes'] = received_bytes
                    except asyncio.TimeoutError:
                        if time.time() >= shared_state.stop_time:
                            break
                        continue

            except websockets.exceptions.ConnectionClosed:
                logger.warning("🔌 音频连接断开")

    except Exception as e:
        logger.error(f"❌ 音频错误: {e}")
        ready_event.set()
    finally:
        f.close()
        logger.info(f"📝 修复 WAV 文件头 ({received_bytes} 字节)")
        with open(output_file, 'r+b') as wf:
            wf.seek(0)
            wf.write(create_wav_header(received_bytes))

        final_size = os.path.getsize(output_file)
        logger.info(f"🏁 音频完成: {output_file} ({final_size / 1024 / 1024:.2f} MB)")
        stats_dict['audio_done'] = True


# ==================== 视频录制任务 ====================
def record_video(output_file, stats_dict, camera_config, ready_event, shared_state, only_video=False):
    """录制视频"""
    camera_id = camera_config.get('camera_id', 1)
    width = camera_config.get('width', 2560)
    height = camera_config.get('height', 1440)
    fps = camera_config.get('fps', 30)

    # 打开摄像头 (保持默认参数，不修改任何图像设置)
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.warning("⚠️  DSHOW 打开失败，尝试 MSMF")
        cap = cv2.VideoCapture(camera_id, cv2.CAP_MSMF)

    if not cap.isOpened():
        logger.error(f"❌ 无法打开摄像头 {camera_id}")
        stats_dict['video_done'] = True
        ready_event.set()
        return

    # 只设置分辨率和帧率，保持其他参数为摄像头默认值
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # 获取实际参数
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info(f"📷 摄像头 {camera_id} 配置:")
    logger.info(f"   分辨率: {actual_width}x{actual_height}")
    logger.info(f"   帧率: {actual_fps}")
    logger.info(f"   (其他参数保持摄像头默认值)")

    frame_queue = Queue(maxsize=120)
    writer_thread = VideoWriterThread(frame_queue, output_file, fps, (actual_width, actual_height))
    writer_thread.start()

    time.sleep(0.1)

    if not writer_thread.writer or not writer_thread.writer.isOpened():
        logger.error("❌ 视频写入器启动失败")
        cap.release()
        stats_dict['video_done'] = True
        ready_event.set()
        return

    # 预热，读取几帧丢弃
    for _ in range(5):
        cap.read()

    stats_dict['video_frames'] = 0
    logger.info(f"📷 视频准备就绪")
    ready_event.set()

    # 等待开始信号
    while shared_state.start_time is None:
        time.sleep(0.001)

    # 信号音频可以开始保存了
    if not only_video:
        shared_state.save_audio = True

    logger.info(f"🎬 视频开始录制")

    try:
        while time.time() < shared_state.stop_time:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                frame_queue.put(frame.copy(), block=False)
                stats_dict['video_frames'] += 1
            except:
                pass

            time.sleep(0.001)

    except Exception as e:
        logger.error(f"❌ 视频错误: {e}")
    finally:
        frame_queue.put(None)
        writer_thread.stop()
        cap.release()

        final_size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
        logger.info(f"🏁 视频完成: {output_file} ({final_size / 1024 / 1024:.2f} MB, {writer_thread.frame_count} 帧)")
        stats_dict['video_done'] = True


# ==================== 主程序 ====================
async def main():
    import argparse
    parser = argparse.ArgumentParser(description="同时录制音频和视频 - 完整版")

    # 列出摄像头命令
    parser.add_argument("--list", action="store_true", help="列出可用的摄像头")
    parser.add_argument("--info", type=int, help="显示指定摄像头的详细信息")

    # 音频参数
    parser.add_argument("--ip", type=str, default="172.16.30.51", help="音频服务器 IP")
    parser.add_argument("--port", type=int, default=8765, help="音频服务器端口")
    parser.add_argument("--no-audio", action="store_true", help="不录制音频")

    # 视频参数
    parser.add_argument("--camera", type=int, default=1, help="摄像头序号 (默认: 1)")
    parser.add_argument("--width", type=int, default=2560, help="视频宽度 (默认: 2560)")
    parser.add_argument("--height", type=int, default=1440, help="视频高度 (默认: 1440)")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率 (默认: 30)")
    parser.add_argument("--no-video", action="store_true", help="不录制视频")

    # 录制参数
    parser.add_argument("--duration", type=float, help="录制时长 (秒)")
    parser.add_argument("--prefix", type=str, default="", help="文件名前缀")
    parser.add_argument("--output-dir", type=str, default="VGDSE/data", help="输出目录")

    args = parser.parse_args()

    # 列出摄像头并退出
    if args.list:
        print("=" * 60)
        print("📷 可用摄像头列表:")
        cameras = list_cameras()
        if cameras:
            for cam_id in cameras:
                info = get_camera_info(cam_id)
                if info:
                    print(f"  摄像头 {cam_id}: {info['default_width']}x{info['default_height']} @ {info['default_fps']}fps ({info['backend']})")
                else:
                    print(f"  摄像头 {cam_id}: 可用")
        else:
            print("  未找到可用摄像头")
        print("=" * 60)
        return

    # 显示摄像头详细信息并退出
    if args.info is not None:
        print("=" * 60)
        print(f"📷 摄像头 {args.info} 详细信息:")
        info = get_camera_info(args.info)
        if info:
            print(f"  默认分辨率: {info['default_width']}x{info['default_height']}")
            print(f"  默认帧率: {info['default_fps']}fps")
            print(f"  后端: {info['backend']}")
        else:
            print(f"  无法打开摄像头 {args.info}")
        print("=" * 60)
        return

    # 检查是否至少录制一种
    if args.no_audio and args.no_video:
        logger.error("❌ 不能同时禁用音频和视频!")
        return

    # 检查录制时长
    if args.duration is None and not (args.no_audio or args.no_video):
        parser.error("请提供 --duration 参数指定录制时长")

    # 创建输出目录
    SAVE_DIR = args.output_dir
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.prefix}_" if args.prefix else ""
    audio_file = os.path.join(SAVE_DIR, f"{prefix}{timestamp}.wav")
    video_file = os.path.join(SAVE_DIR, f"{prefix}{timestamp}.mp4")

    logger.info("=" * 60)
    logger.info("🎬 开始录制")
    if not args.no_video:
        logger.info(f"  摄像头: {args.camera}")
        logger.info(f"  分辨率: {args.width}x{args.height} @ {args.fps}fps")
        logger.info(f"  视频文件: {video_file}")
    if not args.no_audio:
        logger.info(f"  音频服务器: {args.ip}:{args.port}")
        logger.info(f"  音频文件: {audio_file}")
    if args.duration:
        logger.info(f"  时长: {args.duration} 秒")
    logger.info("=" * 60)

    stats_dict = {
        'audio_bytes': 0,
        'video_frames': 0,
        'audio_done': args.no_audio,
        'video_done': args.no_video
    }

    # 创建共享状态
    shared_state = SharedState()

    # 创建准备事件
    video_ready = threading.Event() if not args.no_video else None
    audio_ready = asyncio.Event() if not args.no_audio else None

    video_thread = None
    audio_task = None

    # 先启动音视频（不开始录制，只准备）
    if not args.no_video:
        camera_config = {
            'camera_id': args.camera,
            'width': args.width,
            'height': args.height,
            'fps': args.fps
        }

        video_thread = threading.Thread(
            target=record_video,
            args=(video_file, stats_dict, camera_config, video_ready, shared_state, args.no_audio),
            daemon=True
        )
        video_thread.start()

    if not args.no_audio:
        uri = f"ws://{args.ip}:{args.port}"
        audio_task = asyncio.create_task(
            record_audio(uri, audio_file, stats_dict, audio_ready, shared_state)
        )

    # 等待音视频都准备好
    logger.info("⏳ 等待音视频准备就绪...")
    if not args.no_video:
        while not video_ready.is_set():
            await asyncio.sleep(0.01)
    if not args.no_audio:
        await audio_ready.wait()

    # 音视频都准备好了，同时开始
    shared_state.start_time = time.time()
    shared_state.stop_time = shared_state.start_time + args.duration if args.duration else float('inf')

    # 如果只录音频，这里启动保存
    if args.no_video and not args.no_audio:
        shared_state.save_audio = True

    logger.info(f"🚀 开始录制！时间: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    logger.info(f"⏰ 停止时间: {datetime.fromtimestamp(shared_state.stop_time).strftime('%H:%M:%S.%f')[:-3]}")

    last_log_time = 0

    try:
        while not stats_dict['audio_done'] or not stats_dict['video_done']:
            await asyncio.sleep(0.01)

            elapsed = time.time() - shared_state.start_time
            if elapsed - last_log_time >= 2:
                log_msg = f"⏱️  已录制 {elapsed:.1f}s"
                if args.duration:
                    remaining = max(0, shared_state.stop_time - time.time())
                    log_msg += f" (剩余 {remaining:.1f}s)"
                if not args.no_audio:
                    audio_mb = stats_dict['audio_bytes'] / 1024 / 1024
                    log_msg += f" | 音频: {audio_mb:.2f} MB"
                if not args.no_video:
                    frames = stats_dict['video_frames']
                    log_msg += f" | 视频: {frames} 帧"
                logger.info(log_msg)
                last_log_time = elapsed

            if args.duration and time.time() >= shared_state.stop_time and not stats_dict['audio_done'] and not stats_dict['video_done']:
                logger.info(f"⏰ 到达停止时间！")

    except KeyboardInterrupt:
        logger.info("\n⌨️  用户中断")

    if audio_task:
        await audio_task
    if video_thread:
        video_thread.join(timeout=2)

    total_time = time.time() - shared_state.start_time
    logger.info("=" * 60)
    logger.info(f"🏆 录制完成！总耗时: {total_time:.1f} 秒")
    if not args.no_audio:
        logger.info(f"  🎙 音频: {audio_file}")
    if not args.no_video:
        logger.info(f"  🎬 视频: {video_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
