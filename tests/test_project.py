from pathlib import Path

from dance_focus.exporter import (
    ExportFrameRate,
    ExportQuality,
    ExportResolution,
    ExportSettings,
)
from dance_focus.geometry import (
    Box,
    CameraKeyframe,
    CameraPath,
    CropRect,
    Point,
    TrackSample,
    TrackState,
    TrackingResult,
)
from dance_focus.project import (
    SCHEMA_VERSION,
    ProjectDocument,
    ProjectStore,
    document_from_dict,
    document_to_dict,
    source_ref,
    subject_prompt_hash,
)
from dance_focus.video import VideoInfo


def test_project_round_trip_preserves_expensive_results(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video content")
    info = VideoInfo(video, 1280, 720, 30.0, 2)
    document = ProjectDocument(source=source_ref(info))
    document.subject_prompts = {0: Box(10, 20, 30, 40)}
    document.tracking = TrackingResult(
        samples=(
            TrackSample(Box(10, 20, 30, 40), None, TrackState.MANUAL, 1.0),
            TrackSample(Box(12, 20, 30, 40), None, TrackState.TRACKED, 0.8),
        ),
        engine_version="sam2-test",
    )
    document.tracking_prompt_hash = subject_prompt_hash(document.subject_prompts)
    document.camera_keyframes = [
        CameraKeyframe(1, center=Point(100, 200), zoom=1.5, follow_strength=0.4)
    ]
    document.camera_path = CameraPath(
        (404, 720),
        (CropRect(0, 0, 404, 720), CropRect(20, 10, 300, 534)),
    )
    document.export_settings = ExportSettings(
        ExportQuality.STANDARD,
        ExportResolution.P1080,
        ExportFrameRate.FPS_60,
        True,
    )
    store = ProjectStore(tmp_path / "state")

    path = store.save(document)
    restored = store.load(path)

    assert restored.source == document.source
    assert restored.subject_prompts == document.subject_prompts
    assert restored.tracking == document.tracking
    assert restored.tracking_prompt_hash == document.tracking_prompt_hash
    assert restored.camera_keyframes == document.camera_keyframes
    assert restored.camera_path == document.camera_path
    assert restored.export_settings == document.export_settings


def test_project_store_recovers_previous_backup(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video content")
    info = VideoInfo(video, 640, 360, 30.0, 1)
    document = ProjectDocument(source=source_ref(info))
    store = ProjectStore(tmp_path / "state")
    path = store.save(document)
    document.playhead_frame = 1
    store.save(document)
    path.write_text("not json", encoding="utf-8")

    restored = store.load(path)

    assert restored.playhead_frame == 0


def test_project_store_can_clear_startup_restore_pointer(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video content")
    info = VideoInfo(video, 640, 360, 30.0, 1)
    store = ProjectStore(tmp_path / "state")
    store.save(ProjectDocument(source=source_ref(info)))

    assert store.load_last() is not None

    store.clear_last()

    assert store.load_last() is None


def test_schema_one_project_migrates_with_prompt_hash(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video content")
    info = VideoInfo(video, 640, 360, 30.0, 1)
    document = ProjectDocument(source=source_ref(info))
    document.subject_prompts = {0: Box(10, 20, 30, 40)}
    document.tracking = TrackingResult(
        (TrackSample(Box(10, 20, 30, 40), None, TrackState.MANUAL, 1.0),),
        "old",
    )
    payload = document_to_dict(document)
    payload["schema_version"] = 1
    payload.pop("tracking_prompt_hash")

    restored = document_from_dict(payload)

    assert restored.schema_version == SCHEMA_VERSION
    assert restored.tracking_prompt_hash == subject_prompt_hash(
        restored.subject_prompts
    )


def test_schema_two_project_uses_compatible_export_defaults(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video content")
    info = VideoInfo(video, 640, 360, 30.0, 1)
    payload = document_to_dict(ProjectDocument(source=source_ref(info)))
    payload["schema_version"] = 2
    payload.pop("export_settings")

    restored = document_from_dict(payload)

    assert restored.schema_version == SCHEMA_VERSION
    assert restored.export_settings == ExportSettings()
