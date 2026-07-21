from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
