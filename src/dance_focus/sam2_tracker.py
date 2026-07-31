from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
from urllib.request import Request, urlopen

import cv2
import numpy as np

from dance_focus.geometry import Box
from dance_focus.video import VideoInfo


SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
MODEL_NAME = "SAM 2.1 Small"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
MODEL_FILENAME = "sam2.1_hiera_small.pt"
MODEL_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_small.pt"
)
MODEL_SHA256 = "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
MODEL_SIZE = 184_416_285
DEFAULT_CHUNK_FRAMES = 240
DEFAULT_OVERLAP_FRAMES = 8

_MODEL = None
_MODEL_DEVICE: str | None = None
_MODEL_LOCK = threading.Lock()


def cache_root() -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "dance-focus"


def runtime_description() -> str:
    try:
        import torch
        import sam2  # noqa: F401
    except Exception as error:
        return f"SAM 2 运行环境异常：{error}"
    if torch.cuda.is_available():
        return f"{MODEL_NAME} · GPU: {torch.cuda.get_device_name(0)}"
    return f"{MODEL_NAME} · CPU（会非常慢）"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_checkpoint(progress: Callable[[int], None] | None = None) -> Path:
    model_dir = cache_root() / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / MODEL_FILENAME
    verified = checkpoint.with_suffix(".verified")

    if checkpoint.exists() and verified.exists():
        if verified.read_text(encoding="ascii").strip() == MODEL_SHA256:
            if progress:
                progress(100)
            return checkpoint

    if checkpoint.exists() and _sha256(checkpoint) == MODEL_SHA256:
        verified.write_text(MODEL_SHA256, encoding="ascii")
        if progress:
            progress(100)
        return checkpoint

    checkpoint.unlink(missing_ok=True)
    verified.unlink(missing_ok=True)
    partial = checkpoint.with_suffix(".part")
    partial.unlink(missing_ok=True)
    request = Request(MODEL_URL, headers={"User-Agent": "Dance-Focus/0.4"})
    downloaded = 0
    try:
        with urlopen(request, timeout=60) as response, partial.open("wb") as output:
            total = int(response.headers.get("Content-Length", MODEL_SIZE))
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if progress:
                    progress(min(99, round(downloaded * 100 / max(total, 1))))
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    actual_sha = _sha256(partial)
    if actual_sha != MODEL_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError("SAM 2 模型校验失败，请检查网络后重试")
    partial.replace(checkpoint)
    verified.write_text(MODEL_SHA256, encoding="ascii")
    if progress:
        progress(100)
    return checkpoint


def _get_model(checkpoint: Path):
    global _MODEL, _MODEL_DEVICE
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with _MODEL_LOCK:
        if _MODEL is None or _MODEL_DEVICE != device:
            if device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            _MODEL = build_sam2_video_predictor(
                MODEL_CONFIG,
                str(checkpoint),
                device=device,
                apply_postprocessing=True,
                vos_optimized=False,
            )
            # The optional SAM 2 CUDA extension only fills tiny mask holes. Disabling
            # it avoids requiring a system CUDA compiler and keeps installation portable.
            _MODEL.fill_hole_area = 0
            _MODEL_DEVICE = device
    return _MODEL, device


def _video_cache_key(info: VideoInfo) -> str:
    stat = info.path.stat()
    source = f"{info.path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(source.encode()).hexdigest()[:20]


def cached_frames_dir(info: VideoInfo) -> Path | None:
    frames_dir = cache_root() / "videos" / _video_cache_key(info) / "frames"
    if (frames_dir / "000000.jpg").exists() and (
        frames_dir / f"{info.frame_count - 1:06d}.jpg"
    ).exists():
        return frames_dir
    return None


def _extract_frames(
    info: VideoInfo,
    progress: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    video_cache = cache_root() / "videos" / _video_cache_key(info)
    frames_dir = video_cache / "frames"
    manifest_path = video_cache / "manifest.json"
    expected_last_frame = frames_dir / f"{info.frame_count - 1:06d}.jpg"
    if manifest_path.exists() and expected_last_frame.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("frame_count") == info.frame_count
                and manifest.get("width") == info.width
                and manifest.get("height") == info.height
            ):
                if progress:
                    progress(100)
                return frames_dir
        except (OSError, ValueError):
            pass

    if video_cache.exists():
        shutil.rmtree(video_cache)
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(info.path))
    if not capture.isOpened():
        raise ValueError("无法为 SAM 2 解码视频")

    frame_index = 0
    try:
        while frame_index < info.frame_count:
            if cancelled and cancelled():
                raise InterruptedError("操作已取消")
            ok, frame = capture.read()
            if not ok:
                break
            frame_path = frames_dir / f"{frame_index:06d}.jpg"
            if not cv2.imwrite(
                str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]
            ):
                raise RuntimeError("无法写入 SAM 2 帧缓存")
            frame_index += 1
            if progress and (frame_index % 3 == 0 or frame_index == info.frame_count):
                progress(round(frame_index * 100 / info.frame_count))
    finally:
        capture.release()

    if frame_index != info.frame_count:
        shutil.rmtree(video_cache, ignore_errors=True)
        raise RuntimeError(
            f"视频标记为 {info.frame_count} 帧，但只解码出 {frame_index} 帧"
        )
    manifest_path.write_text(
        json.dumps(
            {
                "source": str(info.path.resolve()),
                "frame_count": info.frame_count,
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return frames_dir


def _chunk_directory(frames_dir: Path, start: int, end: int) -> Path:
    chunk_dir = frames_dir.parent / "chunks" / f"{start:06d}-{end:06d}"
    expected_count = end - start
    expected_last = chunk_dir / f"{expected_count - 1:06d}.jpg"
    if expected_last.exists():
        return chunk_dir
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True)
    for local_index, global_index in enumerate(range(start, end)):
        source = frames_dir / f"{global_index:06d}.jpg"
        (chunk_dir / f"{local_index:06d}.jpg").symlink_to(source)
    return chunk_dir


def _mask_box(mask_logits) -> tuple[Box | None, np.ndarray]:
    mask = mask_logits[0, 0] > 0.0
    coordinates = mask.nonzero(as_tuple=False)
    cpu_mask = mask.to("cpu", non_blocking=False).numpy()
    if coordinates.numel() == 0:
        return None, cpu_mask
    top_left = coordinates.min(dim=0).values
    bottom_right = coordinates.max(dim=0).values
    top, left = top_left.tolist()
    bottom, right = bottom_right.tolist()
    return Box(left, top, right - left + 1, bottom - top + 1), cpu_mask


def _track_range(
    predictor,
    device: str,
    frames_dir: Path,
    start_frame: int,
    end_frame: int,
    initial_box: Box,
    progress: Callable[[int], None] | None,
    cancelled: Callable[[], bool] | None,
    chunk_frames: int,
    overlap_frames: int,
) -> list[Box | None]:
    import torch

    boxes: list[Box | None] = [None] * (end_frame - start_frame)
    start = start_frame
    covered_until = start_frame
    processed_until = start_frame
    seed_mask: np.ndarray | None = None
    overlap_frames = max(1, min(overlap_frames, chunk_frames - 1))

    while start < end_frame:
        if cancelled and cancelled():
            raise InterruptedError("操作已取消")
        end = min(start + chunk_frames, end_frame)
        next_start = end - overlap_frames if end < end_frame else None
        chunk_dir = _chunk_directory(frames_dir, start, end)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else nullcontext()
        )
        next_seed: np.ndarray | None = None
        with context:
            state = predictor.init_state(
                video_path=str(chunk_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=False,
            )
            if seed_mask is not None:
                predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=seed_mask)
            else:
                prompt = np.asarray(
                    [
                        initial_box.x,
                        initial_box.y,
                        initial_box.right,
                        initial_box.bottom,
                    ],
                    dtype=np.float32,
                )
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=1,
                    box=prompt,
                )

            for local_index, _, mask_logits in predictor.propagate_in_video(state):
                if cancelled and cancelled():
                    raise InterruptedError("操作已取消")
                global_index = start + local_index
                box, mask = _mask_box(mask_logits)
                if global_index >= covered_until:
                    boxes[global_index - start_frame] = box
                if next_start is not None and global_index == next_start:
                    next_seed = mask
                processed_until = max(processed_until, global_index + 1)
                if progress:
                    progress(
                        round(
                            (processed_until - start_frame)
                            * 100
                            / (end_frame - start_frame)
                        )
                    )

        del state
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        if end >= end_frame:
            break
        if next_seed is None:
            raise RuntimeError("SAM 2 无法生成下一段的衔接掩码")
        seed_mask = next_seed
        covered_until = end
        start = next_start

    boxes[0] = initial_box
    if progress:
        progress(100)
    return boxes


def track_subject_interval(
    info: VideoInfo,
    start_frame: int,
    end_frame: int,
    initial_box: Box,
    progress: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    overlap_frames: int = DEFAULT_OVERLAP_FRAMES,
) -> list[Box | None]:
    if not 0 <= start_frame < end_frame <= info.frame_count:
        raise ValueError("局部跟踪区间超出视频范围")
    if chunk_frames <= overlap_frames:
        raise ValueError("分段帧数必须大于重叠帧数")

    frames_dir = _extract_frames(
        info,
        (lambda value: progress(round(value * 0.2))) if progress else None,
        cancelled,
    )
    checkpoint = ensure_checkpoint(
        (lambda value: progress(20 + round(value * 0.1))) if progress else None
    )
    predictor, device = _get_model(checkpoint)
    return _track_range(
        predictor,
        device,
        frames_dir,
        start_frame,
        end_frame,
        initial_box,
        (lambda value: progress(30 + round(value * 0.7))) if progress else None,
        cancelled,
        chunk_frames,
        overlap_frames,
    )


def track_subject(
    info: VideoInfo,
    keyframes: Mapping[int, Box],
    progress: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    overlap_frames: int = DEFAULT_OVERLAP_FRAMES,
) -> list[Box | None]:
    if 0 not in keyframes:
        raise ValueError("请先在第一帧完整框选自己")
    if chunk_frames <= overlap_frames:
        raise ValueError("分段帧数必须大于重叠帧数")
    if any(not 0 <= index < info.frame_count for index in keyframes):
        raise ValueError("人物修正帧超出视频范围")

    def extraction_progress(value: int) -> None:
        if progress:
            progress(round(value * 0.2))

    frames_dir = _extract_frames(info, extraction_progress, cancelled)

    def download_progress(value: int) -> None:
        if progress:
            progress(20 + round(value * 0.1))

    checkpoint = ensure_checkpoint(download_progress)
    if progress:
        progress(31)
    predictor, device = _get_model(checkpoint)
    if progress:
        progress(35)

    boxes: list[Box | None] = [None] * info.frame_count
    prompt_indices = sorted(keyframes)
    completed = 0
    for position, start in enumerate(prompt_indices):
        end = (
            prompt_indices[position + 1]
            if position + 1 < len(prompt_indices)
            else info.frame_count
        )
        interval_length = end - start

        def interval_progress(value: int) -> None:
            if progress:
                done = completed + interval_length * value / 100
                progress(35 + round(done * 65 / info.frame_count))

        boxes[start:end] = _track_range(
            predictor,
            device,
            frames_dir,
            start,
            end,
            keyframes[start],
            interval_progress,
            cancelled,
            chunk_frames,
            overlap_frames,
        )
        completed += interval_length

    if progress:
        progress(100)
    return boxes
