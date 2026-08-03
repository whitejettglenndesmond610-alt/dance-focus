from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
import logging
import math
from pathlib import Path
import shutil
import subprocess

import cv2

from dance_focus.geometry import CameraPath, CropRect
from dance_focus.video import VideoInfo


class ExportQuality(StrEnum):
    HIGH = "high"
    STANDARD = "standard"
    SMALL = "small"


class ExportResolution(StrEnum):
    NATIVE = "native"
    P720 = "720p"
    P1080 = "1080p"


class ExportFrameRate(StrEnum):
    SOURCE = "source"
    FPS_30 = "30"
    FPS_60 = "60"


@dataclass(frozen=True)
class ExportSettings:
    quality: ExportQuality = ExportQuality.HIGH
    resolution: ExportResolution = ExportResolution.NATIVE
    frame_rate: ExportFrameRate = ExportFrameRate.SOURCE
    interpolate: bool = False


_RESOLUTION_SIZES = {
    ExportResolution.P720: (
        (16 / 9, (1280, 720)),
        (9 / 16, (720, 1280)),
        (1.0, (720, 720)),
        (4 / 5, (720, 900)),
    ),
    ExportResolution.P1080: (
        (16 / 9, (1920, 1080)),
        (9 / 16, (1080, 1920)),
        (1.0, (1080, 1080)),
        (4 / 5, (1080, 1350)),
    ),
}


def resolve_output_size(
    native_size: tuple[int, int], resolution: ExportResolution
) -> tuple[int, int]:
    width, height = native_size
    if width <= 0 or height <= 0:
        raise ValueError("输出分辨率必须为正数")
    if resolution is ExportResolution.NATIVE:
        return max(2, width - width % 2), max(2, height - height % 2)

    aspect = width / height
    sizes = _RESOLUTION_SIZES[resolution]
    # Ratio distance is logarithmic so portrait and landscape are compared evenly.
    _, size = min(sizes, key=lambda candidate: abs(math.log(aspect / candidate[0])))
    return size[0] - size[0] % 2, size[1] - size[1] % 2


def output_frame_rate(source_fps: float, setting: ExportFrameRate) -> float:
    if setting is ExportFrameRate.FPS_30:
        return 30.0
    if setting is ExportFrameRate.FPS_60:
        return 60.0
    return source_fps


def resize_interpolation(
    source_size: tuple[int, int], target_size: tuple[int, int]
) -> int:
    source_width, source_height = source_size
    target_width, target_height = target_size
    if source_width > target_width or source_height > target_height:
        return cv2.INTER_AREA
    return cv2.INTER_LANCZOS4


def video_filter(
    source_fps: float,
    settings: ExportSettings,
    duration: float | None = None,
) -> str | None:
    if settings.frame_rate is ExportFrameRate.SOURCE:
        return None
    if (
        settings.frame_rate is ExportFrameRate.FPS_60
        and settings.interpolate
        and source_fps < 60
    ):
        interpolation = (
            "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:"
            "me_mode=bidir:vsbmc=1"
        )
        if duration is not None and duration > 0:
            return (
                f"tpad=stop_mode=clone:stop_duration=1,{interpolation},"
                f"trim=duration={duration:.9f},setpts=PTS-STARTPTS"
            )
        return interpolation
    return f"fps={int(output_frame_rate(source_fps, settings.frame_rate))}"


def _available_encoders() -> str:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _encoder_works(encoder: str) -> bool:
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=size=256x256:rate=1",
                "-frames:v",
                "1",
                "-c:v",
                encoder,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def encoder_args(
    encoder: str,
    quality: ExportQuality,
    width: int,
    height: int,
    fps: float,
) -> list[str]:
    quality_index = {
        ExportQuality.HIGH: 0,
        ExportQuality.STANDARD: 1,
        ExportQuality.SMALL: 2,
    }[quality]
    if encoder == "h264_nvenc":
        preset, cq = (("p7", "16"), ("p5", "19"), ("p4", "23"))[quality_index]
        return [
            "-preset",
            preset,
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            cq,
            "-b:v",
            "0",
        ]
    if encoder == "libx264":
        return ["-preset", "medium", "-crf", ("16", "19", "23")[quality_index]]
    if encoder == "libopenh264":
        bits_per_pixel = (0.18, 0.12, 0.08)[quality_index]
        bitrate = round(width * height * fps * bits_per_pixel)
        bitrate = min(50_000_000, max(750_000, bitrate))
        return [
            "-b:v",
            str(bitrate),
            "-maxrate",
            str(bitrate),
            "-bufsize",
            str(bitrate * 2),
        ]
    if encoder == "mpeg4":
        return ["-q:v", ("2", "4", "7")[quality_index]]
    raise ValueError(f"不支持的视频编码器: {encoder}")


def choose_video_encoder(
    quality: ExportQuality = ExportQuality.HIGH,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    available_encoders: str | None = None,
) -> tuple[str, list[str]]:
    probe_encoders = available_encoders is None
    available = _available_encoders() if probe_encoders else available_encoders
    for encoder in ("h264_nvenc", "libx264", "libopenh264", "mpeg4"):
        if encoder in available and (not probe_encoders or _encoder_works(encoder)):
            return encoder, encoder_args(encoder, quality, width, height, fps)
    raise RuntimeError("FFmpeg 中没有可用的 MP4 视频编码器")


def build_ffmpeg_command(
    info: VideoInfo,
    output: Path,
    output_size: tuple[int, int],
    settings: ExportSettings,
    encoder: str,
    codec_args: Sequence[str],
) -> list[str]:
    width, height = output_size
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-video_size", f"{width}x{height}",
        "-framerate", f"{info.fps:.6f}", "-i", "pipe:0",
        "-i", str(info.path), "-map", "0:v:0", "-map", "1:a?",
    ]
    frame_filter = video_filter(info.fps, settings, info.duration)
    if frame_filter:
        command.extend(("-vf", frame_filter))
    command.extend(
        (
            "-c:v", encoder, *codec_args, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-t", f"{info.duration:.9f}",
            "-movflags", "+faststart", str(output),
        )
    )
    return command


def export_video(
    info: VideoInfo,
    camera_path: CameraPath | Sequence[CropRect],
    output_path: str | Path,
    progress: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    settings: ExportSettings | None = None,
) -> Path:
    settings = settings or ExportSettings()
    if isinstance(camera_path, CameraPath):
        crop_path = camera_path.frames
        native_size = camera_path.output_size
    else:
        crop_path = camera_path
        native_size = (
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
    output_size = resolve_output_size(native_size, settings.resolution)
    crop_width, crop_height = output_size
    target_fps = output_frame_rate(info.fps, settings.frame_rate)
    encoder, codec_args = choose_video_encoder(
        settings.quality, crop_width, crop_height, target_fps
    )
    command = build_ffmpeg_command(
        info, output, output_size, settings, encoder, codec_args
    )

    capture = cv2.VideoCapture(str(info.path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("无法打开源视频")

    logging.info(
        "Export started: source=%s output=%s frames=%d encoder=%s size=%sx%s fps=%s",
        info.path, output, len(crop_path), encoder, crop_width, crop_height, target_fps,
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
                interpolation = resize_interpolation(
                    (crop.shape[1], crop.shape[0]), output_size
                )
                crop = cv2.resize(
                    crop, (crop_width, crop_height), interpolation=interpolation
                )
            if process.stdin is None:
                raise RuntimeError("无法向 FFmpeg 写入视频")
            try:
                process.stdin.write(crop.tobytes())
            except BrokenPipeError as exc:
                process.wait()
                error = (
                    process.stderr.read().decode("utf-8", errors="replace")
                    if process.stderr
                    else ""
                )
                raise RuntimeError(error.strip() or "FFmpeg 导出失败") from exc
            written += 1
            if progress and (frame_index % 3 == 0 or frame_index + 1 == len(crop_path)):
                progress(round((frame_index + 1) * 100 / len(crop_path)))

        if process.stdin:
            process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            error = (
                process.stderr.read().decode("utf-8", errors="replace")
                if process.stderr
                else ""
            )
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
        capture.release()
