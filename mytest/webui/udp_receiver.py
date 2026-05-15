"""
UDP JPEG 接收器（后台线程 + asyncio 帧队列）

客户端（camera_client.py --format udp）流程：
  摄像头 MJPEG → FFmpeg (-c:v copy，直通) → stdout
  → Python 按 FFD8/FFD9 切割完整帧 → UDP 分片发送

服务端（本模块）流程：
  后台线程阻塞 recvfrom → 按帧序列号重组 → call_soon_threadsafe → asyncio.Queue
  → _dispatch_loop 任务 await 推理（与 WebSocket 模式完全一致）

架构说明：
  UDP 接收（_udp_recv_thread）运行在独立后台线程，使用阻塞 recvfrom，
  与 asyncio 事件循环完全隔离，无论每帧分多少包都不会占用事件循环时间。
  帧组装完成后通过 call_soon_threadsafe 将完整 JPEG 推入 asyncio.Queue，
  推理触发方式与 WebSocket 模式完全一致。

包格式（每个 UDP 数据报）：
  [frame_id: uint16 big-endian][chunk_idx: uint16 big-endian][JPEG 净荷]
"""
import asyncio
import socket as _socket
import struct
import threading
import time

import webui.state as state

_HEADER_SIZE = 4   # frame_id (2B) + chunk_idx (2B)
_port: int = 5000


def configure(port: int) -> None:
    global _port
    _port = port


def _udp_recv_thread(sock: _socket.socket,
                     loop: asyncio.AbstractEventLoop,
                     frame_queue: asyncio.Queue,
                     shared: dict) -> None:
    """
    后台线程：阻塞 recvfrom 接收 UDP 分片，重组 JPEG 帧。
    不产生任何 asyncio 回调，与事件循环零竞争。
    帧完整时通过 call_soon_threadsafe 安全推入 asyncio 帧队列。
    """
    buf            = bytearray()
    cur_frame_id   = -1
    expected_chunk = 0
    frame_n        = 0
    t0             = time.time()

    while True:
        try:
            data, addr = sock.recvfrom(65535)
        except OSError:
            break   # socket 已关闭，退出线程

        if len(data) <= _HEADER_SIZE:
            continue

        now = time.time()
        shared['last_ts'] = now

        if not shared['connected']:
            shared['connected'] = True
            shared['addr']      = addr
            t0 = now
            with state.perf_lock:
                state.performance_data['connected_clients'] += 1
            print(f'[udp] ✅ 摄像头已连接  {addr[0]}:{addr[1]}')

        frame_id, chunk_idx = struct.unpack('>HH', data[:_HEADER_SIZE])
        payload = data[_HEADER_SIZE:]

        # ── 帧边界检测 ──────────────────────────────────────────────
        if frame_id != cur_frame_id:
            delta = (frame_id - cur_frame_id) & 0xFFFF
            if delta >= 0x8000:
                continue   # 旧帧迟到包，直接丢弃
            buf.clear()
            cur_frame_id   = frame_id
            expected_chunk = 0

        # ── 分片连续性检测 ──────────────────────────────────────────
        if chunk_idx != expected_chunk:
            buf.clear()
            cur_frame_id   = -1
            expected_chunk = 0
            continue

        expected_chunk += 1
        buf.extend(payload)

        # ── O(1) 帧完整性检测 ───────────────────────────────────────
        if (len(buf) < 4
                or buf[0] != 0xff or buf[1] != 0xd8
                or buf[-2] != 0xff or buf[-1] != 0xd9):
            continue

        jpeg = bytes(buf)
        buf.clear()
        cur_frame_id   = -1
        expected_chunk = 0

        frame_n += 1
        if frame_n == 1:
            print(f'[udp] ✅ 首帧到达  大小={len(jpeg)//1024} KB')
        elif frame_n % 150 == 0:
            el = now - t0
            print(f'[udp] {frame_n} 帧  '
                  f'{frame_n / el:.1f} fps  '
                  f'{len(jpeg) // 1024} KB/帧')

        with state.frame_lock:
            state.latest_original_jpeg = jpeg

        # 在事件循环线程安全地推入帧队列
        def _enqueue(j=jpeg):
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                frame_queue.put_nowait(j)
            except asyncio.QueueFull:
                pass

        loop.call_soon_threadsafe(_enqueue)

    print('[udp] 接收线程已停止')


async def _dispatch_loop(frame_queue: asyncio.Queue) -> None:
    """
    独立 asyncio 任务：从帧队列取完整 JPEG 帧，await 推理。
    与 WebSocket 接收器推理触发方式完全一致。
    """
    loop = asyncio.get_running_loop()
    while True:
        try:
            jpeg = await frame_queue.get()
            if state.inference_fn is None:
                continue
            await loop.run_in_executor(
                state.inference_executor, state.inference_fn, jpeg
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f'[udp] ⚠ 推理异常: {type(e).__name__}: {e}')


async def recv_loop() -> None:
    """
    启动 UDP 监听后台线程 + 推理分发任务。
    由 FastAPI startup 事件以 asyncio.create_task 启动。
    """
    loop = asyncio.get_running_loop()

    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind(('0.0.0.0', _port))
    print(f'[udp] 监听 UDP:{_port}，等待摄像头连接…')

    # 帧队列：最多缓存 1 帧，保证分发的始终是最新帧
    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    shared = {'connected': False, 'addr': None, 'last_ts': 0.0}

    recv_thread = threading.Thread(
        target=_udp_recv_thread,
        args=(sock, loop, frame_queue, shared),
        daemon=True,
        name='udp-recv',
    )
    recv_thread.start()

    dispatch_task = asyncio.create_task(_dispatch_loop(frame_queue))

    try:
        while True:
            await asyncio.sleep(1.0)
            # 断线检测：5 秒无数据
            if shared['connected'] and time.time() - shared['last_ts'] > 5.0:
                shared['connected'] = False
                with state.perf_lock:
                    state.performance_data['connected_clients'] = max(
                        0, state.performance_data['connected_clients'] - 1
                    )
                print('[udp] 摄像头断开（超时 5s 无数据），等待重连…')
    except asyncio.CancelledError:
        dispatch_task.cancel()
        sock.close()   # 触发 recvfrom OSError，使接收线程退出
        print('[udp] 接收器已停止')
