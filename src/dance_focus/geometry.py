from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass(frozen=True)
class Point:
    x: float
    y: float


class TrackState(StrEnum):
    MANUAL = "manual"
    TRACKED = "tracked"
    OCCLUDED = "occluded"
    REIDENTIFIED = "reidentified"
    LOST = "lost"


class StabilizationPreset(StrEnum):
    STABLE = "stable"
    BALANCED = "balanced"
    RESPONSIVE = "responsive"


@dataclass(frozen=True)
class PoseAnchor:
    point: Point
    confidence: float
    extent: Box | None = None


@dataclass(frozen=True)
class TrackSample:
    box: Box | None
    anchor: PoseAnchor | None
    state: TrackState
    tracking_confidence: float
    identity_confidence: float | None = None


@dataclass(frozen=True)
class TrackingResult:
    samples: tuple[TrackSample, ...]
    engine_version: str
    pose_model: str | None = None
    reid_model: str | None = None

    @property
    def boxes(self) -> list[Box | None]:
        return [sample.box for sample in self.samples]


@dataclass(frozen=True)
class CameraKeyframe:
    frame_index: int
    center: Point | None = None
    zoom: float | None = None
    follow_strength: float | None = None


@dataclass(frozen=True)
class FramingSettings:
    aspect_ratio: float
    smoothing_seconds: float = 0.45
    subject_margin: float = 0.08
    auto_zoom: bool = True
    max_zoom: float = 1.8
    target_fill: float = 0.72
    stabilization_preset: StabilizationPreset = StabilizationPreset.BALANCED


@dataclass(frozen=True)
class CameraPath:
    output_size: tuple[int, int]
    frames: tuple["CropRect", ...]


@dataclass(frozen=True)
class CropRect:
    x: int
    y: int
    width: int
    height: int


def crop_size_for_aspect(
    frame_width: int, frame_height: int, aspect_ratio: float
) -> tuple[int, int]:
    """Return the largest even-sized crop fitting the requested aspect ratio."""
    if frame_width <= 0 or frame_height <= 0 or aspect_ratio <= 0:
        raise ValueError("Frame dimensions and aspect ratio must be positive")

    if frame_width / frame_height >= aspect_ratio:
        crop_height = frame_height
        crop_width = round(crop_height * aspect_ratio)
    else:
        crop_width = frame_width
        crop_height = round(crop_width / aspect_ratio)

    crop_width = max(2, min(frame_width, crop_width - crop_width % 2))
    crop_height = max(2, min(frame_height, crop_height - crop_height % 2))
    return crop_width, crop_height


def _interpolate_boxes(boxes: Sequence[Box | None]) -> list[Box]:
    if not boxes:
        return []

    valid_indices = [index for index, box in enumerate(boxes) if box is not None]
    if not valid_indices:
        raise ValueError("At least one valid tracking box is required")

    indices = np.arange(len(boxes), dtype=np.float64)
    valid = np.asarray(valid_indices, dtype=np.float64)
    values = np.asarray(
        [
            [boxes[index].x, boxes[index].y, boxes[index].width, boxes[index].height]
            for index in valid_indices
        ],
        dtype=np.float64,
    )
    interpolated = np.column_stack(
        [np.interp(indices, valid, values[:, column]) for column in range(4)]
    )
    return [Box(*row) for row in interpolated]


def _smooth(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(values) < 3:
        return values.copy()

    radius = min(radius, max(1, (len(values) - 1) // 2))
    ramp = np.arange(1, radius + 2, dtype=np.float64)
    kernel = np.concatenate((ramp, ramp[-2::-1]))
    kernel /= kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _median_filter(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(values) < 3:
        return values.copy()
    radius = min(radius, max(1, (len(values) - 1) // 4))
    padded = np.pad(values, (radius, radius), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, radius * 2 + 1)
    return np.median(windows, axis=1)


def _stabilize(values: np.ndarray, smooth_radius: int, outlier_radius: int) -> np.ndarray:
    return _smooth(_median_filter(values, outlier_radius), smooth_radius)


def _preset_limits(preset: StabilizationPreset) -> tuple[float, float, float, float]:
    return {
        StabilizationPreset.STABLE: (0.55, 1.5, 0.45, 1.2),
        StabilizationPreset.BALANCED: (0.90, 3.0, 0.75, 2.5),
        StabilizationPreset.RESPONSIVE: (1.60, 7.0, 1.20, 5.0),
    }[preset]


def _limit_vector_trajectory(
    values: np.ndarray,
    fps: float,
    scale: float,
    max_speed: float,
    max_acceleration: float,
) -> np.ndarray:
    if len(values) < 2:
        return values.copy()
    dt = 1.0 / max(fps, 1.0)

    def run(source: np.ndarray) -> np.ndarray:
        result = np.empty_like(source, dtype=np.float64)
        result[0] = source[0]
        velocity = np.zeros(source.shape[1], dtype=np.float64)
        for index in range(1, len(source)):
            desired_velocity = (source[index] - result[index - 1]) / dt
            delta = desired_velocity - velocity
            delta_norm = float(np.linalg.norm(delta))
            max_delta = max_acceleration * scale * dt
            if delta_norm > max_delta:
                delta *= max_delta / delta_norm
            velocity += delta
            speed = float(np.linalg.norm(velocity))
            speed_limit = max_speed * scale
            if speed > speed_limit:
                velocity *= speed_limit / speed
            result[index] = result[index - 1] + velocity * dt
        return result

    forward = run(values)
    backward = run(values[::-1])[::-1]
    return (forward + backward) / 2


def _limit_scalar_trajectory(
    values: np.ndarray,
    fps: float,
    max_speed: float,
    max_acceleration: float,
) -> np.ndarray:
    limited = _limit_vector_trajectory(
        values[:, None], fps, 1.0, max_speed, max_acceleration
    )
    return limited[:, 0]


def _keyframe_channel(
    keyframes: Sequence[CameraKeyframe],
    frame_count: int,
    getter,
) -> np.ndarray | None:
    points = [
        (keyframe.frame_index, getter(keyframe))
        for keyframe in sorted(keyframes, key=lambda item: item.frame_index)
        if getter(keyframe) is not None
    ]
    if not points:
        return None
    indices = np.arange(frame_count, dtype=np.float64)
    source_indices = np.asarray([point[0] for point in points], dtype=np.float64)
    values = np.asarray([point[1] for point in points], dtype=np.float64)
    return np.interp(indices, source_indices, values)


def build_camera_path(
    tracking: TrackingResult,
    frame_size: tuple[int, int],
    fps: float,
    settings: FramingSettings,
    keyframes: Sequence[CameraKeyframe] = (),
) -> CameraPath:
    if not tracking.samples:
        return CameraPath((0, 0), ())
    if settings.max_zoom < 1 or not 0 < settings.target_fill <= 1:
        raise ValueError("Invalid automatic zoom settings")

    frame_width, frame_height = frame_size
    base_width, base_height = crop_size_for_aspect(
        frame_width, frame_height, settings.aspect_ratio
    )
    boxes = _interpolate_boxes(tracking.boxes)
    frame_count = len(boxes)
    anchors_x = np.asarray(
        [
            sample.anchor.point.x
            if sample.anchor is not None
            else box.center_x
            for sample, box in zip(tracking.samples, boxes, strict=True)
        ],
        dtype=np.float64,
    )
    anchors_y = np.asarray(
        [
            sample.anchor.point.y
            if sample.anchor is not None
            else box.center_y
            for sample, box in zip(tracking.samples, boxes, strict=True)
        ],
        dtype=np.float64,
    )
    extents = [
        sample.anchor.extent
        if sample.anchor is not None and sample.anchor.extent is not None
        else box
        for sample, box in zip(tracking.samples, boxes, strict=True)
    ]

    radius = max(0, round(max(fps, 1.0) * settings.smoothing_seconds))
    outlier_radius = max(1, round(max(fps, 1.0) * 0.12))
    automatic_x = _stabilize(anchors_x, radius, outlier_radius)
    automatic_y = _stabilize(anchors_y, radius, outlier_radius)
    stable_left = _stabilize(
        np.asarray([extent.x for extent in extents]), radius, outlier_radius
    )
    stable_top = _stabilize(
        np.asarray([extent.y for extent in extents]), radius, outlier_radius
    )
    stable_right = _stabilize(
        np.asarray([extent.right for extent in extents]), radius, outlier_radius
    )
    stable_bottom = _stabilize(
        np.asarray([extent.bottom for extent in extents]), radius, outlier_radius
    )
    stable_extents = [
        Box(left, top, max(1.0, right - left), max(1.0, bottom - top))
        for left, top, right, bottom in zip(
            stable_left, stable_top, stable_right, stable_bottom, strict=True
        )
    ]

    max_pan_speed, max_pan_acceleration, max_zoom_speed, max_zoom_acceleration = (
        _preset_limits(settings.stabilization_preset)
    )
    limited_centers = _limit_vector_trajectory(
        np.column_stack((automatic_x, automatic_y)),
        fps,
        base_height,
        max_pan_speed,
        max_pan_acceleration,
    )
    automatic_x = limited_centers[:, 0]
    automatic_y = limited_centers[:, 1]

    if settings.auto_zoom:
        desired_heights = np.asarray(
            [
                max(
                    extent.height / settings.target_fill,
                    extent.width / settings.aspect_ratio / settings.target_fill,
                )
                for extent in stable_extents
            ],
            dtype=np.float64,
        )
        desired_heights = np.clip(
            desired_heights,
            base_height / settings.max_zoom,
            base_height,
        )
        crop_heights = _smooth(desired_heights, radius * 2)
        automatic_zoom = base_height / crop_heights
        automatic_zoom = np.exp(
            _limit_scalar_trajectory(
                np.log(np.maximum(automatic_zoom, 1e-6)),
                fps,
                max_zoom_speed,
                max_zoom_acceleration,
            )
        )
        crop_heights = base_height / automatic_zoom
    else:
        crop_heights = np.full(frame_count, base_height, dtype=np.float64)

    zoom_override = _keyframe_channel(keyframes, frame_count, lambda key: key.zoom)
    if zoom_override is not None:
        zoom_override = np.clip(zoom_override, 1.0, settings.max_zoom)
        crop_heights = base_height / zoom_override

    manual_x = _keyframe_channel(
        keyframes,
        frame_count,
        lambda key: key.center.x if key.center is not None else None,
    )
    manual_y = _keyframe_channel(
        keyframes,
        frame_count,
        lambda key: key.center.y if key.center is not None else None,
    )
    follow = _keyframe_channel(
        keyframes, frame_count, lambda key: key.follow_strength
    )
    if follow is None:
        follow = np.ones(frame_count, dtype=np.float64)
    follow = np.clip(follow, 0.0, 1.0)
    if manual_x is None or manual_y is None:
        camera_x = automatic_x
        camera_y = automatic_y
    else:
        automatic_reference_x = _keyframe_channel(
            keyframes,
            frame_count,
            lambda key: automatic_x[
                min(max(key.frame_index, 0), frame_count - 1)
            ]
            if key.center is not None
            else None,
        )
        automatic_reference_y = _keyframe_channel(
            keyframes,
            frame_count,
            lambda key: automatic_y[
                min(max(key.frame_index, 0), frame_count - 1)
            ]
            if key.center is not None
            else None,
        )
        camera_x = manual_x + (automatic_x - automatic_reference_x) * follow
        camera_y = manual_y + (automatic_y - automatic_reference_y) * follow

    result: list[CropRect] = []
    for index, extent in enumerate(stable_extents):
        crop_height = int(round(float(crop_heights[index])))
        crop_height = max(2, min(base_height, crop_height - crop_height % 2))
        crop_width = int(round(crop_height * settings.aspect_ratio))
        crop_width = max(2, min(base_width, crop_width - crop_width % 2))
        crop_height = int(round(crop_width / settings.aspect_ratio))
        crop_height = max(2, min(base_height, crop_height - crop_height % 2))

        half_width = crop_width / 2
        half_height = crop_height / 2
        margin_x = crop_width * settings.subject_margin
        margin_y = crop_height * settings.subject_margin
        center_x = float(camera_x[index])
        center_y = float(camera_y[index])

        min_center_x = extent.right + margin_x - half_width
        max_center_x = extent.x - margin_x + half_width
        if min_center_x <= max_center_x:
            center_x = min(max(center_x, min_center_x), max_center_x)
        else:
            center_x = extent.center_x

        min_center_y = extent.bottom + margin_y - half_height
        max_center_y = extent.y - margin_y + half_height
        if min_center_y <= max_center_y:
            center_y = min(max(center_y, min_center_y), max_center_y)
        else:
            center_y = extent.center_y

        center_x = min(max(center_x, half_width), frame_width - half_width)
        center_y = min(max(center_y, half_height), frame_height - half_height)
        x = min(
            max(int(round(center_x - half_width)), 0), frame_width - crop_width
        )
        y = min(
            max(int(round(center_y - half_height)), 0), frame_height - crop_height
        )
        result.append(CropRect(x, y, crop_width, crop_height))

    return CameraPath((base_width, base_height), tuple(result))


def build_crop_path(
    boxes: Sequence[Box | None],
    frame_size: tuple[int, int],
    crop_size: tuple[int, int],
    fps: float,
    smoothing_seconds: float = 0.45,
    subject_margin: float = 0.06,
) -> list[CropRect]:
    """Build a smooth crop path while keeping the tracked subject inside frame."""
    if not boxes:
        return []

    frame_width, frame_height = frame_size
    crop_width, crop_height = crop_size
    if crop_width > frame_width or crop_height > frame_height:
        raise ValueError("Crop must fit inside the source frame")

    filled_boxes = _interpolate_boxes(boxes)
    target_x = np.asarray([box.center_x for box in filled_boxes])
    target_y = np.asarray([box.center_y for box in filled_boxes])
    radius = max(0, round(max(fps, 1.0) * smoothing_seconds))
    camera_x = _smooth(target_x, radius)
    camera_y = _smooth(target_y, radius)

    half_width = crop_width / 2
    half_height = crop_height / 2
    margin_x = crop_width * subject_margin
    margin_y = crop_height * subject_margin

    result: list[CropRect] = []
    for index, box in enumerate(filled_boxes):
        center_x = float(camera_x[index])
        center_y = float(camera_y[index])

        min_center_x = box.right + margin_x - half_width
        max_center_x = box.x - margin_x + half_width
        if min_center_x <= max_center_x:
            center_x = min(max(center_x, min_center_x), max_center_x)
        else:
            center_x = box.center_x

        min_center_y = box.bottom + margin_y - half_height
        max_center_y = box.y - margin_y + half_height
        if min_center_y <= max_center_y:
            center_y = min(max(center_y, min_center_y), max_center_y)
        else:
            center_y = box.center_y

        center_x = min(max(center_x, half_width), frame_width - half_width)
        center_y = min(max(center_y, half_height), frame_height - half_height)
        x = int(round(center_x - half_width))
        y = int(round(center_y - half_height))
        x = min(max(x, 0), frame_width - crop_width)
        y = min(max(y, 0), frame_height - crop_height)
        result.append(CropRect(x, y, crop_width, crop_height))

    return result
