import asyncio
import json
import socket
import argparse
import logging
from typing import Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 配置常量
CHUNK_SIZE = 384 * 2 * 30       # 3072 字节
MAX_BUFFER_SIZE = CHUNK_SIZE * 1000 # 3,072,000 字节
WS_PORT = 8765
AUDIO_PORT = 50002
TCP_READ_SIZE = 65536

class AudioBuffer:
    """
    先进先出缓冲区 C 的实现
    使用 asyncio.Condition 实现线程/协程安全的读写同步
    """
    def __init__(self):
        self.buf = bytearray()
        self.condition = asyncio.Condition()

    async def write(self, data: bytes):
        """写入数据，并在缓冲区满时按规则丢弃旧数据"""
        async with self.condition:
            self.buf.extend(data)
            # 如果超过最大限制，按照 CHUNK_SIZE 为单位丢弃头部数据
            while len(self.buf) > MAX_BUFFER_SIZE:
                if len(self.buf) >= CHUNK_SIZE:
                    del self.buf[:CHUNK_SIZE]
                else:
                    break
            self.condition.notify_all()

    async def read_chunk(self) -> bytes:
        """读取一个 CHUNK_SIZE 大小的块，不足时阻塞等待"""
        async with self.condition:
            while len(self.buf) < CHUNK_SIZE:
                await self.condition.wait()
            
            # 提取头部数据
            chunk = bytes(self.buf[:CHUNK_SIZE])
            del self.buf[:CHUNK_SIZE]
            return chunk

    def reset(self):
        """清空缓冲区"""
        self.buf = bytearray()
        # 唤醒可能阻塞在 read_chunk 的读取者，避免死锁
        asyncio.get_event_loop().call_soon_threadsafe(self.condition.notify_all)

def get_local_ip() -> str:
    """获取本机局域网 IP"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

async def control_device(device_ip: str, buffer: AudioBuffer, trigger_event: asyncio.Event):
    """
    步骤 6: 向设备 D 建立 TCP 连接，发送 JSON 控制指令
    """
    local_ip = get_local_ip()
    payload = json.dumps({"opcode": 1, "ip": local_ip, "port": AUDIO_PORT})

    while True:
        try:
            reader, writer = await asyncio.open_connection(device_ip, 50001)
            writer.write(payload.encode('utf-8'))
            await writer.drain()
            
            # 等待 100 毫秒
            await asyncio.sleep(0.1)
            
            writer.close()
            await writer.wait_closed()
            logger.info("设备控制指令发送成功")
            
            # 发送成功，退出循环，等待音频连接建立
            return
        except Exception as e:
            logger.warning(f"连接设备 {device_ip}:50001 失败: {e}。10 秒后重试...")
            await asyncio.sleep(10)

async def tcp_audio_server(host: str, buffer: AudioBuffer, ctrl_task_factory):
    """
    步骤 2: TCP 服务器 A，接收 PCM 数据
    """
    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info('peername')
        logger.info(f"音频设备已连接: {addr}")
        try:
            while True:
                # 每次读取尽可能多的数据（通常底层会分片）
                data = await reader.read(TCP_READ_SIZE)
                if not data:
                    raise ConnectionError("设备主动断开连接或 EOF")
                await buffer.write(data)
        except (ConnectionError, asyncio.CancelledError) as e:
            logger.warning(f"音频连接断开: {addr}。触发重连与重置。")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass
            
            # 重置缓冲区
            buffer.reset()
            # 触发重连控制指令逻辑
            if ctrl_task_factory:
                ctrl_task_factory()

    server = await asyncio.start_server(handle_client, host, AUDIO_PORT)
    logger.info(f"TCP 音频服务器启动: {host}:{AUDIO_PORT}")
    async with server:
        await server.serve_forever()

async def ws_server(host: str, buffer: AudioBuffer):
    """
    步骤 4 & 5: WebSocket 服务器 B，转发音频流
    """
    ws_clients: Set[object] = set()

    import websockets
    async def ws_handler(websocket):
        ws_clients.add(websocket)
        logger.info(f"WebSocket 客户端连接: {websocket.remote_address}，当前连接数: {len(ws_clients)}")
        try:
            await websocket.wait_closed()
        finally:
            ws_clients.discard(websocket)
            logger.info(f"WebSocket 客户端断开，当前连接数: {len(ws_clients)}")

    server = await websockets.serve(ws_handler, host, WS_PORT)
    logger.info(f"WebSocket 音频服务器启动: {host}:{WS_PORT}")

    # 独立广播循环：不断从缓冲区取数据并转发
    async def broadcast_loop():
        while True:
            chunk = await buffer.read_chunk()
            if ws_clients:
                # 向所有客户端发送
                send_tasks = [ws.send(chunk) for ws in ws_clients]
                await asyncio.gather(*send_tasks, return_exceptions=True)

    # 启动广播任务
    asyncio.create_task(broadcast_loop())
    
    async with server:
        await server.serve_forever()

async def main():
    parser = argparse.ArgumentParser(description="网络音频管理服务器")
    parser.add_argument("--device", type=str, required=True, help="录音设备 D 的 IP 地址")
    args = parser.parse_args()

    device_ip = args.device
    buffer = AudioBuffer()

    # 用于管理重连逻辑的引用
    ctrl_task = None

    def start_control_connection():
        """启动控制连接任务，并保存引用以便管理"""
        nonlocal ctrl_task
        if ctrl_task and not ctrl_task.done():
            ctrl_task.cancel()
        ctrl_task = asyncio.create_task(control_device(device_ip, buffer, None))

    # 首次启动执行步骤 6
    start_control_connection()

    # 并行运行 TCP 音频服务器和 WebSocket 服务器
    await asyncio.gather(
        tcp_audio_server("0.0.0.0", buffer, start_control_connection),
        ws_server("0.0.0.0", buffer),
        return_exceptions=True
    )

if __name__ == "__main__":
    asyncio.run(main())
