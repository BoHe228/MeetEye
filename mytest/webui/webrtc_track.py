"""
aiortc VideoStreamTrack — 固定帧率定时器推流，消除 PTS 抖动
"""
import asyncio
import fractions
import time

import av
import numpy as np
from aiortc import VideoStreamTrack

import webui.state as ws

_BLANK_W = 3840


class InferenceVideoTrack(VideoStreamTrack):
    """
    固定 FPS 定时器驱动帧推送，保证 PTS 等间隔，避免浏览器 jitter buffer 膨胀。
    BGR→YUV420P 转换已在推理线程完成（存入 ws.latest_webrtc_frame），
    recv() 只做 PTS 赋值，事件循环阻塞时间接近零。
    """

    _CLOCK_RATE = 90000

    def __init__(self, fps: int = 30):
        super().__init__()
        self._fps = fps
        self._step = self._CLOCK_RATE // fps
        self._pts: int = 0
        self._next_time: float = 0.0

    async def recv(self) -> av.VideoFrame:
        now = time.time()
        if self._next_time == 0.0:
            self._next_time = now
        wait = self._next_time - now
        if wait > 0:
            await asyncio.sleep(wait)
        self._next_time += 1.0 / self._fps
        self._pts += self._step

        with ws.frame_lock:
            vf = ws.latest_webrtc_frame

        if vf is None:
            blank = np.zeros((720 * 3 // 2, _BLANK_W), dtype=np.uint8)
            vf = av.VideoFrame.from_ndarray(blank, format='yuv420p')

        vf.pts = self._pts
        vf.time_base = fractions.Fraction(1, self._CLOCK_RATE)
        return vf
