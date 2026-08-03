import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from dance_focus.ui_components import (
    ConfidenceTimeline,
    SegmentedControl,
    WorkflowIndicator,
)


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def test_segmented_control_tracks_text_data_and_selection(app):
    control = SegmentedControl()
    control.addItem("原始", "source")
    control.addItem("成片", "crop")
    changes = []
    control.currentIndexChanged.connect(changes.append)

    control.resize(240, 40)
    control.show()
    app.processEvents()
    control.setCurrentIndex(1)
    app.processEvents()

    assert control.currentIndex() == 1
    assert control.currentText() == "成片"
    assert control.currentData() == "crop"
    assert changes == [1]
    assert control._indicator.geometry().width() > 0
    control.setItemEnabled(1, False)
    assert not control.isItemEnabled(1)
    control.close()


def test_workflow_indicator_updates_without_visible_animation(app):
    indicator = WorkflowIndicator(["导入", "跟踪", "预览", "导出"])

    indicator.setStep(2)

    assert indicator.get_progress() == 2.0


def test_timeline_click_near_keyframe_snaps_to_exact_frame(app):
    timeline = ConfidenceTimeline()
    timeline.setRange(0, 99)
    timeline.set_keyframes([50])
    timeline.resize(400, 28)
    timeline.show()
    app.processEvents()

    marker_x = round(8 + (timeline.width() - 16) * 50 / 99)
    QTest.mouseClick(
        timeline,
        Qt.MouseButton.LeftButton,
        pos=QPoint(marker_x + 6, timeline.height() // 2),
    )

    assert timeline.value() == 50
    timeline.close()
