from __future__ import annotations

from collections.abc import Callable, Mapping

import cv2
import numpy as np

from dance_focus.geometry import (
    Box,
    PoseAnchor,
    TrackSample,
    TrackState,
    TrackingResult,
)
from dance_focus.pose import (
    POSE_MODEL_NAME,
    POSE_STRIDE,
    detect_pose_candidates,
    interpolate_pose_anchors,
    match_pose_candidate,
)
from dance_focus.reid import (
    REID_MODEL_NAME,
    IdentityGallery,
    crop_person,
    ensure_reid_checkpoint,
)
from dance_focus.sam2_tracker import (
    SAM2_COMMIT,
    cached_frames_dir,
    track_subject,
    track_subject_interval,
)
from dance_focus.video import VideoInfo


ENGINE_VERSION = (
    f"sam2:{SAM2_COMMIT[:8]}+pose:torchvision0.28+reid:osnet-ain+gate:v2"
)


def affected_correction_interval(
    old_prompts: Mapping[int, Box],
    new_prompts: Mapping[int, Box],
    changed_frame: int,
    frame_count: int,
) -> tuple[int, int]:
    """Return the half-open interval affected by one prompt edit."""
    if frame_count <= 0 or not 0 <= changed_frame < frame_count:
        raise ValueError("修正帧超出视频范围")
    if 0 not in new_prompts:
        raise ValueError("第一帧人物框不能删除")

    if changed_frame in new_prompts:
        start = changed_frame
    else:
        previous = [index for index in new_prompts if index < changed_frame]
        if not previous:
            raise ValueError("找不到删除修正后的前一个人物框")
        start = max(previous)

    later = [index for index in new_prompts if index > changed_frame]
    end = min(later) if later else frame_count
    if changed_frame not in old_prompts and changed_frame not in new_prompts:
        raise ValueError("指定帧没有发生人物修正")
    return start, end


def splice_tracking_result(
    previous: TrackingResult,
    replacement: TrackingResult,
    start: int,
    end: int,
) -> TrackingResult:
    if not 0 <= start < end <= len(previous.samples):
        raise ValueError("局部跟踪区间无效")
    if len(replacement.samples) != end - start:
        raise ValueError("局部跟踪结果长度不匹配")
    samples = (
        previous.samples[:start]
        + replacement.samples
        + previous.samples[end:]
    )
    return TrackingResult(
        samples=samples,
        engine_version=replacement.engine_version,
        pose_model=replacement.pose_model,
        reid_model=replacement.reid_model,
    )


def _identity_recovery_is_plausible(
    previous: Box | None,
    candidate: Box,
    frame_gap: int,
    frame_width: int,
    pose_stride: int,
) -> bool:
    if previous is None:
        return True
    distance = float(
        np.hypot(
            candidate.center_x - previous.center_x,
            candidate.center_y - previous.center_y,
        )
    )
    diagonal = float(np.hypot(previous.width, previous.height))
    interval_scale = max(1.0, frame_gap / max(1, pose_stride))
    allowed = max(frame_width * 0.12, diagonal * 1.5) * interval_scale
    return distance <= allowed


def _tracking_confidences(boxes: list[Box | None]) -> np.ndarray:
    confidence = np.zeros(len(boxes), dtype=np.float64)
    previous = None
    for index, box in enumerate(boxes):
        if box is None:
            previous = None
            continue
        if previous is None:
            confidence[index] = 0.72
        else:
            diagonal = max(1.0, float(np.hypot(previous.width, previous.height)))
            movement = np.hypot(
                box.center_x - previous.center_x,
                box.center_y - previous.center_y,
            ) / diagonal
            previous_area = max(1.0, previous.width * previous.height)
            area_change = abs(np.log(max(1.0, box.width * box.height) / previous_area))
            confidence[index] = float(np.exp(-2.8 * movement - 1.4 * area_change))
        previous = box
    if len(confidence) >= 3:
        confidence = np.convolve(
            np.pad(confidence, (1, 1), mode="edge"),
            np.asarray([0.2, 0.6, 0.2]),
            mode="valid",
        )
    return np.clip(confidence, 0.0, 1.0)


def _read_analysis_frame(
    info: VideoInfo, frames_dir, capture, frame_index: int
):
    if frames_dir is not None:
        return cv2.imread(str(frames_dir / f"{frame_index:06d}.jpg"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    return frame if ok else None


def _interpolate_sampled_boxes(
    original: list[Box | None], sampled: dict[int, Box]
) -> list[Box | None]:
    if len(sampled) < 2:
        return original.copy()
    sample_indices = np.asarray(sorted(sampled), dtype=np.float64)
    values = np.asarray(
        [
            [sampled[index].x, sampled[index].y, sampled[index].width, sampled[index].height]
            for index in sample_indices.astype(int)
        ],
        dtype=np.float64,
    )
    indices = np.arange(len(original), dtype=np.float64)
    interpolated = np.column_stack(
        [np.interp(indices, sample_indices, values[:, column]) for column in range(4)]
    )
    result = []
    first = int(sample_indices[0])
    last = int(sample_indices[-1])
    for index, original_box in enumerate(original):
        if first <= index <= last:
            result.append(Box(*interpolated[index]))
        else:
            result.append(original_box)
    return result


def analyze_subject(
    info: VideoInfo,
    prompts: Mapping[int, Box],
    progress: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    pose_stride: int = POSE_STRIDE,
    sam_boxes_override: list[Box | None] | None = None,
    analysis_range: tuple[int, int] | None = None,
) -> TrackingResult:
    if sam_boxes_override is None:
        def sam_progress(value: int) -> None:
            if progress:
                progress(round(value * 0.70))

        sam_boxes = track_subject(
            info,
            prompts,
            progress=sam_progress,
            cancelled=cancelled,
        )
    else:
        if len(sam_boxes_override) != info.frame_count:
            raise ValueError("预计算跟踪框数量与视频帧数不一致")
        sam_boxes = sam_boxes_override
        if progress:
            progress(70)
    sam_quality = _tracking_confidences(sam_boxes)
    if cancelled and cancelled():
        raise InterruptedError("操作已取消")

    analysis_start, analysis_end = analysis_range or (0, info.frame_count)
    if not 0 <= analysis_start < analysis_end <= info.frame_count:
        raise ValueError("姿态分析区间超出视频范围")
    if analysis_start not in prompts:
        raise ValueError("姿态分析起点缺少可信人物框")

    frames_dir = cached_frames_dir(info)
    capture = None if frames_dir is not None else cv2.VideoCapture(str(info.path))
    first_frame = _read_analysis_frame(
        info, frames_dir, capture, analysis_start
    )
    if first_frame is None:
        raise RuntimeError("无法读取首帧建立人物身份特征")
    ensure_reid_checkpoint()
    gallery = IdentityGallery(crop_person(first_frame, prompts[analysis_start]))

    sample_indices = list(
        range(analysis_start, analysis_end, max(1, pose_stride))
    )
    if sample_indices[-1] != analysis_end - 1:
        sample_indices.append(analysis_end - 1)
    sampled_anchors: dict[int, PoseAnchor] = {}
    sampled_boxes: dict[int, Box] = {}
    sampled_pose_quality: dict[int, float] = {}
    sampled_identity: dict[int, float] = {}
    reidentified_frames: set[int] = set()
    last_selected_box: Box | None = prompts[analysis_start]
    last_selected_frame = analysis_start

    try:
        for sample_number, frame_index in enumerate(sample_indices):
            if cancelled and cancelled():
                raise InterruptedError("操作已取消")
            frame = _read_analysis_frame(info, frames_dir, capture, frame_index)
            if frame is None:
                continue
            candidates = detect_pose_candidates(frame)
            target = sam_boxes[frame_index]
            spatial_candidate = None
            spatial_quality = 0.0
            if target is not None:
                spatial_candidate, spatial_quality = match_pose_candidate(
                    target, candidates
                )

            crops = [crop_person(frame, candidate.box) for candidate in candidates]
            similarities = gallery.similarities(crops) if crops else np.empty(0)
            selected = spatial_candidate
            identity_confidence = 0.0
            if spatial_candidate is not None:
                selected_index = candidates.index(spatial_candidate)
                identity_confidence = float(similarities[selected_index])

            if len(similarities):
                order = np.argsort(similarities)[::-1]
                best_index = int(order[0])
                best_similarity = float(similarities[best_index])
                second_similarity = (
                    float(similarities[order[1]]) if len(order) > 1 else -1.0
                )
                has_identity_margin = (
                    best_similarity >= 0.65
                    and best_similarity - second_similarity >= 0.08
                )
                identity_candidate = candidates[best_index]
                plausible_recovery = _identity_recovery_is_plausible(
                    last_selected_box,
                    identity_candidate.box,
                    frame_index - last_selected_frame,
                    info.width,
                    pose_stride,
                )
                if selected is None and has_identity_margin and plausible_recovery:
                    selected = candidates[best_index]
                    spatial_quality = 0.45
                    identity_confidence = best_similarity
                    reidentified_frames.add(frame_index)
                elif (
                    selected is not None
                    and identity_confidence < 0.40
                    and has_identity_margin
                    and plausible_recovery
                ):
                    selected = candidates[best_index]
                    identity_confidence = best_similarity
                    reidentified_frames.add(frame_index)

            if selected is not None:
                sampled_boxes[frame_index] = selected.box
                sampled_anchors[frame_index] = selected.anchor
                sampled_pose_quality[frame_index] = spatial_quality
                sampled_identity[frame_index] = identity_confidence
                gallery.update(crop_person(frame, selected.box), identity_confidence)
                last_selected_box = selected.box
                last_selected_frame = frame_index

            if progress:
                progress(70 + round((sample_number + 1) * 30 / len(sample_indices)))
    finally:
        if capture is not None:
            capture.release()

    refined_boxes = _interpolate_sampled_boxes(sam_boxes, sampled_boxes)
    anchors = interpolate_pose_anchors(refined_boxes, sampled_anchors)
    indices = np.arange(info.frame_count, dtype=np.float64)

    def interpolate_quality(values: dict[int, float], fallback: float) -> np.ndarray:
        if not values:
            return np.full(info.frame_count, fallback, dtype=np.float64)
        source = np.asarray(sorted(values), dtype=np.float64)
        scores = np.asarray([values[int(index)] for index in source], dtype=np.float64)
        return np.interp(indices, source, scores)

    pose_quality = interpolate_quality(sampled_pose_quality, 0.0)
    identity_quality = interpolate_quality(sampled_identity, 0.0)
    samples: list[TrackSample] = []
    manual_frames = set(prompts)
    for frame_index, (box, anchor) in enumerate(
        zip(refined_boxes, anchors, strict=True)
    ):
        quality = float(
            np.clip(
                0.50 * sam_quality[frame_index]
                + 0.30 * pose_quality[frame_index]
                + 0.20 * max(identity_quality[frame_index], 0.0),
                0.0,
                1.0,
            )
        )
        if frame_index in manual_frames:
            box = prompts[frame_index]
            state = TrackState.MANUAL
            quality = 1.0
        elif frame_index in reidentified_frames:
            state = TrackState.REIDENTIFIED
        elif box is None or quality < 0.15:
            state = TrackState.LOST
        elif quality < 0.42:
            state = TrackState.OCCLUDED
        else:
            state = TrackState.TRACKED
        samples.append(
            TrackSample(
                box=box,
                anchor=anchor,
                state=state,
                tracking_confidence=quality,
                identity_confidence=float(identity_quality[frame_index]),
            )
        )

    if progress:
        progress(100)
    return TrackingResult(
        samples=tuple(samples),
        engine_version=ENGINE_VERSION,
        pose_model=POSE_MODEL_NAME,
        reid_model=REID_MODEL_NAME,
    )


def reanalyze_subject_interval(
    info: VideoInfo,
    prompts: Mapping[int, Box],
    previous: TrackingResult,
    start_frame: int,
    end_frame: int,
    progress: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    pose_stride: int = POSE_STRIDE,
) -> TrackingResult:
    if len(previous.samples) != info.frame_count:
        raise ValueError("已有跟踪结果与视频帧数不一致")
    if start_frame not in prompts:
        raise ValueError("局部跟踪起点缺少人物修正框")

    interval_boxes = track_subject_interval(
        info,
        start_frame,
        end_frame,
        prompts[start_frame],
        progress=(
            (lambda value: progress(round(value * 0.70)))
            if progress
            else None
        ),
        cancelled=cancelled,
    )
    merged_boxes = previous.boxes
    merged_boxes[start_frame:end_frame] = interval_boxes
    candidate = analyze_subject(
        info,
        prompts,
        progress=progress,
        cancelled=cancelled,
        pose_stride=pose_stride,
        sam_boxes_override=merged_boxes,
        analysis_range=(start_frame, end_frame),
    )
    replacement = TrackingResult(
        samples=candidate.samples[start_frame:end_frame],
        engine_version=candidate.engine_version,
        pose_model=candidate.pose_model,
        reid_model=candidate.reid_model,
    )
    return splice_tracking_result(previous, replacement, start_frame, end_frame)
