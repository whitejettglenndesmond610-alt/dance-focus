from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0


def inspect_video(path: str | Path) -> tuple[VideoInfo, object]:
    video_path = Path(path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("无法打开这个视频，请尝试转换为 MP4 后重试")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first_frame = capture.read()
    capture.release()
    if not ok or width <= 0 or height <= 0 or frame_count <= 0:
        raise ValueError("视频中没有可读取的画面")

    return VideoInfo(video_path, width, height, fps, frame_count), first_frame


def read_frame(info: VideoInfo, frame_index: int):
    capture = cv2.VideoCapture(str(info.path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame_index, info.frame_count - 1)))
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None
