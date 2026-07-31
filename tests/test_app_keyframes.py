from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from dance_focus.app import MainWindow
from dance_focus.geometry import (
    Box,
    CameraKeyframe,
    Point,
    PoseAnchor,
    TrackSample,
    TrackState,
    TrackingResult,
)
from dance_focus.project import ProjectStore
from dance_focus.video import VideoInfo


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def test_keyframe_sliders_preview_commit_and_delete(app, tmp_path):
    window = MainWindow()
    window.project_store = ProjectStore(tmp_path / "state")
    window.video_info = VideoInfo(Path("/tmp/preview.mp4"), 1280, 720, 30, 10)
    window.current_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    samples = tuple(
        TrackSample(
            Box(600, 300, 80, 120),
            PoseAnchor(Point(640, 360), 0.9),
            TrackState.TRACKED,
            0.9,
        )
        for _ in range(10)
    )
    window.tracking_result = TrackingResult(samples, "test")
    window.tracked_boxes = window.tracking_result.boxes
    window.ratio_combo.setCurrentIndex(2)
    window.auto_zoom_combo.setCurrentIndex(0)
    window._rebuild_crop_path()
    window.preview_combo.setCurrentIndex(1)

    committed_path = window.camera_path
    original_crop = window.crop_path[0]
    window.keyframe_zoom_slider.setValue(150)
    window.keyframe_x_slider.setValue(100)

    preview_crop = window.crop_path[0]
    assert window._preview_camera_path is not None
    assert window.camera_path is committed_path
    assert window.camera_keyframes == []
    assert preview_crop.height < original_crop.height
    assert preview_crop.x + preview_crop.width / 2 == 740

    preview_path = tuple(window.crop_path)
    window.current_frame_index = 5
    window._update_keyframe_editor()

    assert tuple(window.crop_path) == preview_path
    assert window.add_keyframe_button.text() == "写入第 1 帧"
    assert window.remove_keyframe_button.text() == "取消预览"

    window._add_camera_keyframe()

    assert len(window.camera_keyframes) == 1
    assert window.camera_keyframes[0].frame_index == 0
    assert window.camera_path is not committed_path
    assert window.frame_slider._keyframes == {0}

    window._undo()
    assert window.camera_keyframes == []
    window._redo()
    assert len(window.camera_keyframes) == 1

    window.current_frame_index = 0
    window._update_keyframe_editor()
    assert window.remove_keyframe_button.isEnabled()
    window._remove_camera_keyframe()

    assert window.camera_keyframes == []
    assert not window.remove_keyframe_button.isEnabled()
    assert window.frame_slider._keyframes == set()
    assert window.crop_path[0] == original_crop

    window.keyframe_zoom_slider.setValue(140)
    assert window._preview_keyframe is not None
    window._remove_camera_keyframe()
    assert window._preview_keyframe is None
    assert window.remove_keyframe_button.text() == "删除当前帧"
    window.project_document = None
    window.close()


def test_reset_buttons_restore_framing_and_keyframe_defaults(app, tmp_path):
    window = MainWindow()
    window.project_store = ProjectStore(tmp_path / "state")
    assert window.smoothing_slider.minimum() == 25
    window.ratio_combo.setCurrentIndex(2)
    window.smoothing_slider.setValue(10)
    window.auto_zoom_combo.setCurrentIndex(0)
    window.max_zoom_slider.setValue(240)

    window.reset_framing_button.click()

    assert window.ratio_combo.currentIndex() == 0
    assert window.stabilization_combo.currentData().value == "balanced"
    assert window.smoothing_slider.value() == 45
    assert window.auto_zoom_combo.currentData() is True
    assert window.max_zoom_slider.value() == 180

    window._undo()
    assert window.ratio_combo.currentIndex() == 2
    assert window.smoothing_slider.value() == 25
    assert window.auto_zoom_combo.currentData() is False
    assert window.max_zoom_slider.value() == 240
    window._redo()
    assert window.ratio_combo.currentIndex() == 0
    assert window.smoothing_slider.value() == 45

    window.keyframe_x_slider.setValue(120)
    window.keyframe_y_slider.setValue(-80)
    window.keyframe_zoom_slider.setValue(220)
    window.keyframe_follow_slider.setValue(20)
    window.reset_keyframe_button.setEnabled(True)
    window.reset_keyframe_button.click()

    assert window.keyframe_x_slider.value() == 0
    assert window.keyframe_y_slider.value() == 0
    assert window.keyframe_zoom_slider.value() == 100
    assert window.keyframe_follow_slider.value() == 100
    window.project_document = None
    window.close()


def test_subject_correction_is_transactional_and_preserves_camera_keys(
    app, tmp_path, monkeypatch
):
    window = MainWindow()
    window.project_store = ProjectStore(tmp_path / "state")
    window.video_info = VideoInfo(Path("/tmp/preview.mp4"), 640, 360, 30, 6)
    window.current_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    window.current_frame_index = 2
    window.keyframes = {
        0: Box(100, 100, 50, 120),
        4: Box(180, 100, 50, 120),
    }
    samples = tuple(
        TrackSample(
            Box(100 + index * 20, 100, 50, 120),
            PoseAnchor(Point(125 + index * 20, 160), 0.9),
            TrackState.TRACKED,
            0.9,
        )
        for index in range(6)
    )
    original = TrackingResult(samples, "old")
    window.tracking_result = original
    window.tracked_boxes = original.boxes
    window.camera_keyframes = [
        analysis_camera_key := CameraKeyframe(3, Point(320, 180), 1.2, 0.5)
    ]
    window._rebuild_crop_path()
    workers = []
    monkeypatch.setattr(window, "_start_worker", workers.append)

    correction = Box(260, 100, 50, 120)
    window._selection_changed(correction)

    assert window.tracking_result is original
    assert window.camera_keyframes == [analysis_camera_key]
    assert window._pending_correction_interval == (2, 4)
    assert workers[0].args[-2:] == (2, 4)

    window._correction_error("操作已取消")
    window._set_busy(False)
    assert 2 not in window.keyframes
    assert window.tracking_result is original

    window._selection_changed(correction)
    replacement_samples = list(samples)
    replacement_samples[2] = TrackSample(
        correction, None, TrackState.MANUAL, 1.0
    )
    corrected = TrackingResult(tuple(replacement_samples), "new")
    window._correction_complete(corrected)
    window._set_busy(False)

    assert window.keyframes[2] == correction
    assert window.tracking_result is corrected
    assert window.camera_keyframes == [analysis_camera_key]
    window.project_document = None
    window.close()
