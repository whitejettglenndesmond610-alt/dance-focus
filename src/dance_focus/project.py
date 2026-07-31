from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from dance_focus.geometry import (
    Box,
    CameraKeyframe,
    CameraPath,
    CropRect,
    FramingSettings,
    Point,
    PoseAnchor,
    TrackSample,
    TrackState,
    TrackingResult,
    StabilizationPreset,
)
from dance_focus.video import VideoInfo


SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SourceRef:
    path: str
    fingerprint: str
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass
class ProjectDocument:
    source: SourceRef
    project_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    schema_version: int = SCHEMA_VERSION
    subject_prompts: dict[int, Box] = field(default_factory=dict)
    tracking: TrackingResult | None = None
    tracking_prompt_hash: str | None = None
    framing: FramingSettings = field(
        default_factory=lambda: FramingSettings(aspect_ratio=9 / 16)
    )
    camera_keyframes: list[CameraKeyframe] = field(default_factory=list)
    camera_path: CameraPath | None = None
    playhead_frame: int = 0
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def source_fingerprint(path: str | Path) -> str:
    source = Path(path)
    stat = source.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    with source.open("rb") as file:
        digest.update(file.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            file.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(file.read(1024 * 1024))
    return digest.hexdigest()


def source_ref(info: VideoInfo) -> SourceRef:
    return SourceRef(
        path=str(info.path.resolve()),
        fingerprint=source_fingerprint(info.path),
        width=info.width,
        height=info.height,
        fps=info.fps,
        frame_count=info.frame_count,
    )


def subject_prompt_hash(prompts: dict[int, Box]) -> str:
    payload = [
        [index, box.x, box.y, box.width, box.height]
        for index, box in sorted(prompts.items())
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _box_to_dict(box: Box | None):
    if box is None:
        return None
    return {"x": box.x, "y": box.y, "width": box.width, "height": box.height}


def _box_from_dict(data) -> Box | None:
    if data is None:
        return None
    return Box(float(data["x"]), float(data["y"]), float(data["width"]), float(data["height"]))


def _point_to_dict(point: Point | None):
    if point is None:
        return None
    return {"x": point.x, "y": point.y}


def _point_from_dict(data) -> Point | None:
    if data is None:
        return None
    return Point(float(data["x"]), float(data["y"]))


def document_to_dict(document: ProjectDocument) -> dict:
    tracking = None
    if document.tracking is not None:
        tracking = {
            "engine_version": document.tracking.engine_version,
            "pose_model": document.tracking.pose_model,
            "reid_model": document.tracking.reid_model,
            "samples": [
                {
                    "box": _box_to_dict(sample.box),
                    "anchor": (
                        {
                            "point": _point_to_dict(sample.anchor.point),
                            "confidence": sample.anchor.confidence,
                            "extent": _box_to_dict(sample.anchor.extent),
                        }
                        if sample.anchor is not None
                        else None
                    ),
                    "state": sample.state.value,
                    "tracking_confidence": sample.tracking_confidence,
                    "identity_confidence": sample.identity_confidence,
                }
                for sample in document.tracking.samples
            ],
        }

    camera_path = None
    if document.camera_path is not None:
        camera_path = {
            "output_size": list(document.camera_path.output_size),
            "frames": [
                {
                    "x": crop.x,
                    "y": crop.y,
                    "width": crop.width,
                    "height": crop.height,
                }
                for crop in document.camera_path.frames
            ],
        }

    return {
        "schema_version": document.schema_version,
        "project_id": document.project_id,
        "updated_at": datetime.now(UTC).isoformat(),
        "source": {
            "path": document.source.path,
            "fingerprint": document.source.fingerprint,
            "width": document.source.width,
            "height": document.source.height,
            "fps": document.source.fps,
            "frame_count": document.source.frame_count,
        },
        "subject_prompts": {
            str(index): _box_to_dict(box)
            for index, box in document.subject_prompts.items()
        },
        "tracking": tracking,
        "tracking_prompt_hash": document.tracking_prompt_hash,
        "framing": {
            "aspect_ratio": document.framing.aspect_ratio,
            "smoothing_seconds": document.framing.smoothing_seconds,
            "subject_margin": document.framing.subject_margin,
            "auto_zoom": document.framing.auto_zoom,
            "max_zoom": document.framing.max_zoom,
            "target_fill": document.framing.target_fill,
            "stabilization_preset": document.framing.stabilization_preset.value,
        },
        "camera_keyframes": [
            {
                "frame_index": keyframe.frame_index,
                "center": _point_to_dict(keyframe.center),
                "zoom": keyframe.zoom,
                "follow_strength": keyframe.follow_strength,
            }
            for keyframe in document.camera_keyframes
        ],
        "camera_path": camera_path,
        "playhead_frame": document.playhead_frame,
    }


def document_from_dict(data: dict) -> ProjectDocument:
    schema_version = int(data.get("schema_version", 0))
    if schema_version not in {1, SCHEMA_VERSION}:
        raise ValueError(f"不支持的项目版本：{schema_version}")
    source_data = data["source"]
    source = SourceRef(
        path=str(source_data["path"]),
        fingerprint=str(source_data["fingerprint"]),
        width=int(source_data["width"]),
        height=int(source_data["height"]),
        fps=float(source_data["fps"]),
        frame_count=int(source_data["frame_count"]),
    )
    framing_data = data.get("framing", {})
    framing = FramingSettings(
        aspect_ratio=float(framing_data.get("aspect_ratio", 9 / 16)),
        smoothing_seconds=float(framing_data.get("smoothing_seconds", 0.45)),
        subject_margin=float(framing_data.get("subject_margin", 0.08)),
        auto_zoom=bool(framing_data.get("auto_zoom", True)),
        max_zoom=float(framing_data.get("max_zoom", 1.8)),
        target_fill=float(framing_data.get("target_fill", 0.72)),
        stabilization_preset=StabilizationPreset(
            framing_data.get("stabilization_preset", "balanced")
        ),
    )

    tracking_data = data.get("tracking")
    tracking = None
    if tracking_data is not None:
        samples = []
        for item in tracking_data.get("samples", []):
            anchor_data = item.get("anchor")
            anchor = None
            if anchor_data is not None:
                anchor = PoseAnchor(
                    point=_point_from_dict(anchor_data["point"]),
                    confidence=float(anchor_data["confidence"]),
                    extent=_box_from_dict(anchor_data.get("extent")),
                )
            samples.append(
                TrackSample(
                    box=_box_from_dict(item.get("box")),
                    anchor=anchor,
                    state=TrackState(item["state"]),
                    tracking_confidence=float(item["tracking_confidence"]),
                    identity_confidence=(
                        float(item["identity_confidence"])
                        if item.get("identity_confidence") is not None
                        else None
                    ),
                )
            )
        tracking = TrackingResult(
            samples=tuple(samples),
            engine_version=str(tracking_data["engine_version"]),
            pose_model=tracking_data.get("pose_model"),
            reid_model=tracking_data.get("reid_model"),
        )

    camera_path_data = data.get("camera_path")
    camera_path = None
    if camera_path_data is not None:
        camera_path = CameraPath(
            output_size=tuple(int(value) for value in camera_path_data["output_size"]),
            frames=tuple(
                CropRect(
                    int(item["x"]),
                    int(item["y"]),
                    int(item["width"]),
                    int(item["height"]),
                )
                for item in camera_path_data.get("frames", [])
            ),
        )

    subject_prompts = {
        int(index): _box_from_dict(box)
        for index, box in data.get("subject_prompts", {}).items()
    }
    if any(
        box is None or not 0 <= index < source.frame_count
        for index, box in subject_prompts.items()
    ):
        raise ValueError("项目包含无效的人物修正帧")
    tracking_prompt_hash = data.get("tracking_prompt_hash")
    if tracking is not None and tracking_prompt_hash is None:
        tracking_prompt_hash = subject_prompt_hash(subject_prompts)

    return ProjectDocument(
        source=source,
        project_id=str(data.get("project_id", uuid.uuid4().hex)),
        schema_version=SCHEMA_VERSION,
        subject_prompts=subject_prompts,
        tracking=tracking,
        tracking_prompt_hash=tracking_prompt_hash,
        framing=framing,
        camera_keyframes=[
            CameraKeyframe(
                frame_index=int(item["frame_index"]),
                center=_point_from_dict(item.get("center")),
                zoom=(float(item["zoom"]) if item.get("zoom") is not None else None),
                follow_strength=(
                    float(item["follow_strength"])
                    if item.get("follow_strength") is not None
                    else None
                ),
            )
            for item in data.get("camera_keyframes", [])
        ],
        camera_path=camera_path,
        playhead_frame=int(data.get("playhead_frame", 0)),
        updated_at=str(data.get("updated_at", datetime.now(UTC).isoformat())),
    )


class ProjectStore:
    def __init__(self, root: Path | None = None):
        if root is None:
            state_home = Path(
                os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
            )
            root = state_home / "dance-focus"
        self.root = root
        self.projects_dir = root / "projects"
        self.last_project_file = root / "last-project"

    def autosave_path(self, document: ProjectDocument) -> Path:
        return self.projects_dir / f"{document.source.fingerprint[:20]}.dancefocus.json"

    def save(self, document: ProjectDocument, path: Path | None = None) -> Path:
        destination = path or self.autosave_path(document)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        backup = destination.with_suffix(destination.suffix + ".bak")
        if destination.exists():
            shutil.copy2(destination, backup)
        payload = json.dumps(
            document_to_dict(document), ensure_ascii=False, indent=2
        )
        with temporary.open("w", encoding="utf-8") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(destination)
        self.root.mkdir(parents=True, exist_ok=True)
        self.last_project_file.write_text(str(destination), encoding="utf-8")
        return destination

    def load(self, path: str | Path) -> ProjectDocument:
        project_path = Path(path)
        try:
            data = json.loads(project_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            backup = project_path.with_suffix(project_path.suffix + ".bak")
            if not backup.exists():
                raise
            data = json.loads(backup.read_text(encoding="utf-8"))
        document = document_from_dict(data)
        source_path = Path(document.source.path)
        if not source_path.exists():
            raise FileNotFoundError(f"项目素材不存在：{source_path}")
        if source_fingerprint(source_path) != document.source.fingerprint:
            raise ValueError("项目素材已发生变化，不能复用旧跟踪结果")
        return document

    def load_for_video(self, info: VideoInfo) -> ProjectDocument | None:
        fingerprint = source_fingerprint(info.path)
        path = self.projects_dir / f"{fingerprint[:20]}.dancefocus.json"
        return self.load(path) if path.exists() else None

    def load_last(self) -> ProjectDocument | None:
        if not self.last_project_file.exists():
            return None
        path = Path(self.last_project_file.read_text(encoding="utf-8").strip())
        return self.load(path) if path.exists() else None

    def clear_last(self) -> None:
        self.last_project_file.unlink(missing_ok=True)
