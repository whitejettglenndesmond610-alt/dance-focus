from __future__ import annotations

from collections.abc import Callable, Sequence
import logging
from pathlib import Path
import shutil
import subprocess

import cv2

from dance_focus.geometry import CameraPath, CropRect
from dance_focus.sam2_tracker import cached_frames_dir
from dance_focus.video import VideoInfo


def _available_encoders() -> str:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def choose_video_encoder() -> tuple[str, list[str]]:
    encoders = _available_encoders()
    if "libx264" in encoders:
        return "libx264", ["-preset", "medium", "-crf", "18"]
    if "libopenh264" in encoders:
        return "libopenh264", ["-b:v", "8M"]
    if "mpeg4" in encoders:
        return "mpeg4", ["-q:v", "2"]
    raise RuntimeError("FFmpeg 中没有可用的 MP4 视频编码器")


def export_video(
    info: VideoInfo,
    camera_path: CameraPath | Sequence[CropRect],
    output_path: str | Path,
    progress: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    if isinstance(camera_path, CameraPath):
        crop_path = camera_path.frames
        output_size = camera_path.output_size
    else:
        crop_path = camera_path
        output_size = (
            max((crop.width for crop in crop_path), default=0),
            max((crop.height for crop in crop_path), default=0),
        )
    if not crop_path:
        raise ValueError("还没有可导出的裁剪路径")
    if len(crop_path) != info.frame_count:
        raise ValueError("镜头路径帧数与源视频不一致，请重新分析")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("没有找到 FFmpeg，请先安装 FFmpeg")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    crop_width, crop_height = output_size
    if crop_width <= 0 or crop_height <= 0 or crop_width % 2 or crop_height % 2:
        raise ValueError("输出分辨率必须是正偶数")
    encoder, encoder_args = choose_video_encoder()
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-video_size",
        f"{crop_width}x{crop_height}",
        "-framerate",
        f"{info.fps:.6f}",
        "-i",
        "pipe:0",
        "-i",
        str(info.path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        encoder,
        *encoder_args,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output),
    ]

    frames_dir = cached_frames_dir(info)
    capture = None if frames_dir is not None else cv2.VideoCapture(str(info.path))
    if capture is not None and not capture.isOpened():
        raise ValueError("无法打开源视频")

    logging.info(
        "Export started: source=%s output=%s frames=%d cache=%s encoder=%s",
        info.path,
        output,
        len(crop_path),
        frames_dir is not None,
        encoder,
    )

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    written = 0
    try:
        for frame_index, crop_rect in enumerate(crop_path):
            if cancelled and cancelled():
                raise InterruptedError("操作已取消")
            if frames_dir is not None:
                frame = cv2.imread(str(frames_dir / f"{frame_index:06d}.jpg"))
                ok = frame is not None
            else:
                ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"读取第 {frame_index + 1} 帧失败，导出已停止")
            crop = frame[
                crop_rect.y : crop_rect.y + crop_rect.height,
                crop_rect.x : crop_rect.x + crop_rect.width,
            ]
            if crop.size == 0:
                raise RuntimeError(f"第 {frame_index + 1} 帧的镜头窗口无效")
            if crop.shape[1] != crop_width or crop.shape[0] != crop_height:
                interpolation = (
                    cv2.INTER_AREA
                    if crop.shape[1] > crop_width or crop.shape[0] > crop_height
                    else cv2.INTER_CUBIC
                )
                crop = cv2.resize(
                    crop, (crop_width, crop_height), interpolation=interpolation
                )
            if process.stdin is None:
                raise RuntimeError("无法向 FFmpeg 写入视频")
            process.stdin.write(crop.tobytes())
            written += 1
            if progress and (frame_index % 3 == 0 or frame_index + 1 == len(crop_path)):
                progress(round((frame_index + 1) * 100 / len(crop_path)))

        if process.stdin:
            process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            raise RuntimeError(error.strip() or "FFmpeg 导出失败")
        if written == 0:
            raise RuntimeError("没有读取到可导出的视频帧")
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError("FFmpeg 没有生成有效的输出文件")
        validation = cv2.VideoCapture(str(output))
        valid, _ = validation.read()
        validation.release()
        if not valid:
            raise RuntimeError("导出文件生成后无法解码")
        logging.info("Export completed: output=%s frames=%d", output, written)
        return output
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait()
        output.unlink(missing_ok=True)
        raise
    finally:
        if capture is not None:
            capture.release()
