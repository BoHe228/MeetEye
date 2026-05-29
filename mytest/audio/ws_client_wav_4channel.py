import asyncio
import websockets
import argparse
import logging
import struct
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_wav_header(data_size, sample_rate=48000, channels=4, bits_per_sample=16):
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

async def ws_audio_client(uri, output_file, duration_sec):
    sample_rate = 48000
    channels = 4
    bits_per_sample = 16
    target_bytes = int(duration_sec * sample_rate * channels * (bits_per_sample // 8))
    received_bytes = 0
    last_log_percent = 0  # 记录上一次打印的百分比阈值

    logger.info(f"🎙 准备录制 {duration_sec} 秒音频...")
    logger.info(f"📦 目标数据大小: {target_bytes / 1024 / 1024:.2f} MB")

    f = open(output_file, 'wb')
    try:
        f.write(create_wav_header(0))  # 先写入 44 字节占位头
        
        async with websockets.connect(uri) as ws:
            logger.info(f"✅ 已连接至 WebSocket 服务器: {uri}")
            async for message in ws:
                if isinstance(message, bytes):
                    f.write(message)
                    received_bytes += len(message)
                    
                    # 计算当前进度百分比
                    progress = (received_bytes / target_bytes) * 100
                    
                    # 每达到 10% 打印一次
                    if progress >= last_log_percent + 10:
                        last_log_percent += 10
                        log_pct = min(last_log_percent, 100)
                        logger.info(f"📊 进度: {log_pct}% | 已接收: {received_bytes / 1024 / 1024:.2f} MB")
                    
                    if received_bytes >= target_bytes:
                        logger.info("⏱️ 已达到设定录音时长，准备退出...")
                        break
                else:
                    logger.warning("⚠️ 收到非二进制数据帧，已跳过")
                    
    except websockets.exceptions.ConnectionClosed:
        logger.warning("🔌 服务器主动断开连接")
    except Exception as e:
        logger.error(f"❌ 发生未预期错误: {e}")
    finally:
        f.close()
        logger.info(f"📝 正在修复 WAV 文件头 (实际写入: {received_bytes} 字节)...")
        with open(output_file, 'r+b') as wf:
            wf.seek(0)
            wf.write(create_wav_header(received_bytes))
            
        final_size = os.path.getsize(output_file)
        logger.info(f"🏁 录制完成！文件已保存: {output_file} (总大小: {final_size} 字节)")

async def main():
    parser = argparse.ArgumentParser(description="WebSocket 音频流录制客户端 (自动添加WAV头)")
    parser.add_argument("--ip", type=str, required=True, help="服务器 IP 地址")
    parser.add_argument("--port", type=int, default=8765, help="服务器 WebSocket 端口")
    parser.add_argument("--duration", type=float, required=True, help="录音时长 (秒)")
    parser.add_argument("--output", type=str, default="recorded.wav", help="输出文件名")
    args = parser.parse_args()

    uri = f"ws://{args.ip}:{args.port}"
    try:
        await ws_audio_client(uri, args.output, args.duration)
    except KeyboardInterrupt:
        logger.info("\n⌨️  收到中断信号，安全退出...")

if __name__ == "__main__":
    asyncio.run(main())
