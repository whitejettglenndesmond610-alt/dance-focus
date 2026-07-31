from __future__ import annotations

from dataclasses import dataclass
import threading

import cv2
import numpy as np

from dance_focus.geometry import Box, Point, PoseAnchor


POSE_MODEL_NAME = "torchvision Keypoint R-CNN COCO_V1"
POSE_STRIDE = 3

_MODEL = None
_TRANSFORM = None
_DEVICE = None
_MODEL_LOCK = threading.Lock()


@dataclass(frozen=True)
class PoseCandidate:
    box: Box
    score: float
    anchor: PoseAnchor


def _get_model():
    global _MODEL, _TRANSFORM, _DEVICE
    import torch
    from torchvision.models.detection import (
        KeypointRCNN_ResNet50_FPN_Weights,
        keypointrcnn_resnet50_fpn,
    )

    with _MODEL_LOCK:
        if _MODEL is None:
            weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
            _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _MODEL = keypointrcnn_resnet50_fpn(
                weights=weights,
                box_score_thresh=0.60,
                box_detections_per_img=20,
            ).eval().to(_DEVICE)
            _TRANSFORM = weights.transforms()
    return _MODEL, _TRANSFORM, _DEVICE


def _iou(first: Box, second: Box) -> tuple[float, float]:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(1.0, first.width * first.height)
    second_area = max(1.0, second.width * second.height)
    union = first_area + second_area - intersection
    return intersection / max(union, 1.0), intersection / min(first_area, second_area)


def detect_pose_candidates(frame_bgr) -> list[PoseCandidate]:
    import torch

    model, transform, device = _get_model()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(frame_rgb).permute(2, 0, 1)
    image = transform(image).to(device, non_blocking=True)
    with torch.inference_mode():
        prediction = model([image])[0]

    boxes = prediction["boxes"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    keypoints = prediction["keypoints"].detach().cpu().numpy()
    keypoint_scores = prediction.get("keypoints_scores")
    if keypoint_scores is not None:
        keypoint_scores = keypoint_scores.detach().cpu().numpy()

    candidates: list[PoseCandidate] = []
    for index, (raw_box, label, score) in enumerate(zip(boxes, labels, scores, strict=True)):
        if label != 1 or score < 0.60:
            continue
        left, top, right, bottom = raw_box.tolist()
        box = Box(left, top, right - left, bottom - top)
        points = keypoints[index, :, :2]
        if keypoint_scores is None:
            valid = np.ones(17, dtype=bool)
            point_quality = np.ones(17, dtype=np.float32)
        else:
            logits = keypoint_scores[index]
            valid = logits > 2.0
            point_quality = 1 / (1 + np.exp(-logits))

        torso_indices = np.asarray([5, 6, 11, 12])
        valid_torso = torso_indices[valid[torso_indices]]
        if len(valid_torso) >= 2:
            anchor_xy = points[valid_torso].mean(axis=0)
            anchor_confidence = float(point_quality[valid_torso].mean())
        else:
            anchor_xy = np.asarray([box.center_x, box.center_y])
            anchor_confidence = float(score * 0.5)

        extent_indices = np.asarray([0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 15, 16])
        valid_extent = extent_indices[valid[extent_indices]]
        extent = box
        if len(valid_extent) >= 5:
            visible = points[valid_extent]
            min_x, min_y = visible.min(axis=0)
            max_x, max_y = visible.max(axis=0)
            min_x = min(float(min_x), box.x)
            min_y = min(float(min_y), box.y)
            max_x = max(float(max_x), box.right)
            max_y = max(float(max_y), box.bottom)
            extent = Box(min_x, min_y, max_x - min_x, max_y - min_y)

        candidates.append(
            PoseCandidate(
                box=box,
                score=float(score),
                anchor=PoseAnchor(
                    point=Point(float(anchor_xy[0]), float(anchor_xy[1])),
                    confidence=anchor_confidence,
                    extent=extent,
                ),
            )
        )
    return candidates


def match_pose_candidate(
    target: Box, candidates: list[PoseCandidate]
) -> tuple[PoseCandidate | None, float]:
    best = None
    best_quality = 0.0
    diagonal = max(1.0, float(np.hypot(target.width, target.height)))
    expanded = Box(
        target.x - target.width * 0.125,
        target.y - target.height * 0.125,
        target.width * 1.25,
        target.height * 1.25,
    )
    for candidate in candidates:
        iou, ios = _iou(target, candidate.box)
        center = candidate.anchor.point
        if not (
            expanded.x <= center.x <= expanded.right
            and expanded.y <= center.y <= expanded.bottom
        ):
            continue
        distance = np.hypot(
            candidate.box.center_x - target.center_x,
            candidate.box.center_y - target.center_y,
        ) / diagonal
        quality = 0.55 * iou + 0.30 * ios + 0.15 * np.exp(-2 * distance**2)
        if (iou >= 0.10 or ios >= 0.50) and quality > best_quality:
            best = candidate
            best_quality = float(quality)
    if best_quality < 0.35:
        return None, best_quality
    return best, best_quality


def interpolate_pose_anchors(
    boxes: list[Box | None],
    sampled: dict[int, PoseAnchor],
) -> list[PoseAnchor | None]:
    if not sampled:
        return [None] * len(boxes)
    valid_boxes = [box for box in boxes if box is not None]
    if not valid_boxes:
        return [None] * len(boxes)

    sample_indices = np.asarray(sorted(sampled), dtype=np.float64)
    relative = []
    for index in sample_indices.astype(int):
        box = boxes[index]
        anchor = sampled[index]
        if box is None:
            relative.append((0.5, 0.5, 0.0, 0.0, 1.0, 1.0, anchor.confidence))
            continue
        extent = anchor.extent or box
        relative.append(
            (
                (anchor.point.x - box.x) / max(box.width, 1.0),
                (anchor.point.y - box.y) / max(box.height, 1.0),
                (extent.x - box.x) / max(box.width, 1.0),
                (extent.y - box.y) / max(box.height, 1.0),
                extent.width / max(box.width, 1.0),
                extent.height / max(box.height, 1.0),
                anchor.confidence,
            )
        )
    values = np.asarray(relative, dtype=np.float64)
    frame_indices = np.arange(len(boxes), dtype=np.float64)
    interpolated = np.column_stack(
        [
            np.interp(frame_indices, sample_indices, values[:, column])
            for column in range(values.shape[1])
        ]
    )

    result: list[PoseAnchor | None] = []
    last_box = valid_boxes[0]
    for box, row in zip(boxes, interpolated, strict=True):
        if box is None:
            box = last_box
        else:
            last_box = box
        u, v, extent_u, extent_v, extent_w, extent_h, confidence = row
        extent = Box(
            box.x + extent_u * box.width,
            box.y + extent_v * box.height,
            max(1.0, extent_w * box.width),
            max(1.0, extent_h * box.height),
        )
        result.append(
            PoseAnchor(
                point=Point(box.x + u * box.width, box.y + v * box.height),
                confidence=float(confidence),
                extent=extent,
            )
        )
    return result
