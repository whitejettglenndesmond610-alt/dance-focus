from __future__ import annotations

import logging
from pathlib import Path
import sys

from dance_focus.analysis_process import configure_native_threads

configure_native_threads()

import cv2
from PySide6.QtCore import QThreadPool, QTimer, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QPalette
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from dance_focus.analysis import affected_correction_interval
from dance_focus.analysis_process import (
    analyze_subject_isolated,
    reanalyze_subject_interval_isolated,
)
from dance_focus.canvas import VideoCanvas
from dance_focus.diagnostics import configure_logging, log_path
from dance_focus.exporter import export_video
from dance_focus.geometry import (
    Box,
    CameraKeyframe,
    CameraPath,
    CropRect,
    FramingSettings,
    Point,
    StabilizationPreset,
    TrackingResult,
    build_camera_path,
)
from dance_focus.history import EditHistory, EditState
from dance_focus.project import (
    ProjectDocument,
    ProjectStore,
    source_ref,
    subject_prompt_hash,
)
from dance_focus.sam2_tracker import cached_frames_dir, runtime_description
from dance_focus.theme import APP_STYLESHEET
from dance_focus.ui_components import (
    AnimatedStatusLabel,
    ConfidenceTimeline,
    MotionButton,
    SegmentedControl,
    WorkflowIndicator,
    animate_fade_in,
    animate_refresh,
)
from dance_focus.video import VideoInfo, inspect_video
from dance_focus.workers import FunctionWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dance Focus · AI Reframe Studio")
        self.resize(1440, 900)
        self.setMinimumSize(1120, 720)

        self.thread_pool = QThreadPool.globalInstance()
        self.active_workers: set[FunctionWorker] = set()
        self.current_worker: FunctionWorker | None = None
        self.video_info: VideoInfo | None = None
        self.current_frame = None
        self.current_frame_index = 0
        self.keyframes: dict[int, Box] = {}
        self.camera_keyframes: list[CameraKeyframe] = []
        self.tracking_result: TrackingResult | None = None
        self.camera_path: CameraPath | None = None
        self.tracked_boxes: list[Box | None] = []
        self.crop_path: list[CropRect] = []
        self.busy = False
        self.preview_capture = None
        self.last_export_path: Path | None = None
        self._intro_played = False
        self._motion_animations = []
        self._updating_keyframe_editor = False
        self._keyframe_edit_base_center: Point | None = None
        self._keyframe_edit_frame_index = 0
        self._preview_keyframe: CameraKeyframe | None = None
        self._preview_camera_path: CameraPath | None = None
        self._pending_prompts_before: dict[int, Box] | None = None
        self._pending_correction_interval: tuple[int, int] | None = None
        self.edit_history = EditHistory()
        self._history_restoring = False
        self._history_transaction: tuple[str, EditState] | None = None
        self._pending_edit_before: EditState | None = None
        self.project_store = ProjectStore()
        self.project_document: ProjectDocument | None = None
        self._restoring_project = False
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.setInterval(500)
        self.autosave_timer.timeout.connect(self._save_project)

        self.play_timer = QTimer(self)
        self.play_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.play_timer.timeout.connect(self._advance_playback)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.8)
        self.media_player = QMediaPlayer(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.errorOccurred.connect(self._media_error)

        self._build_interface()
        self._setup_history_actions()
        self.setStyleSheet(APP_STYLESHEET)

    def _build_interface(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        self.top_bar = self._build_top_bar()
        self.stage_card = self._build_stage()
        self.inspector = self._build_inspector()

        body = QHBoxLayout()
        body.setSpacing(14)
        body.addWidget(self.stage_card, 1)
        body.addWidget(self.inspector)

        root_layout.addWidget(self.top_bar)
        root_layout.addLayout(body, 1)
        self.setCentralWidget(root)

    def _build_top_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("topBar")
        bar.setFixedHeight(68)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)

        mark = QLabel("DF")
        mark.setObjectName("brandMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(38, 38)

        brand = QVBoxLayout()
        brand.setSpacing(0)
        brand_title = QLabel("DANCE FOCUS")
        brand_title.setObjectName("brandTitle")
        brand_subtitle = QLabel("SAM 2 · AI REFRAME STUDIO")
        brand_subtitle.setObjectName("eyebrow")
        brand.addWidget(brand_title)
        brand.addWidget(brand_subtitle)

        self.project_label = QLabel("NO CLIP LOADED")
        self.project_label.setObjectName("projectName")
        self.project_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        runtime = runtime_description().replace(" · ", "  /  ")
        runtime_chip = QLabel(runtime)
        runtime_chip.setObjectName("runtimeChip")
        runtime_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(mark)
        layout.addLayout(brand)
        layout.addStretch()
        layout.addWidget(self.project_label, 1)
        layout.addSpacing(10)
        layout.addWidget(runtime_chip)
        return bar

    def _build_stage(self) -> QFrame:
        card = QFrame()
        card.setObjectName("stageCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("monitorHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 13, 18, 11)
        header_layout.setSpacing(12)

        title_stack = QVBoxLayout()
        title_stack.setSpacing(1)
        eyebrow = QLabel("PROGRAM MONITOR")
        eyebrow.setObjectName("eyebrow")
        monitor_title = QLabel("动态构图监看器")
        monitor_title.setObjectName("panelTitle")
        title_stack.addWidget(eyebrow)
        title_stack.addWidget(monitor_title)

        self.video_meta_label = QLabel("— × —  /  — FPS")
        self.video_meta_label.setObjectName("videoMeta")

        self.preview_combo = SegmentedControl()
        self.preview_combo.setFixedWidth(220)
        self.preview_combo.addItem("原始画面", "source")
        self.preview_combo.addItem("成片预览", "crop")
        self.preview_combo.currentIndexChanged.connect(self._preview_mode_changed)

        header_layout.addLayout(title_stack)
        header_layout.addStretch()
        header_layout.addWidget(self.video_meta_label)
        header_layout.addWidget(self.preview_combo)

        self.canvas = VideoCanvas()
        self.canvas.selection_changed.connect(self._selection_changed)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        transport = QFrame()
        transport.setObjectName("transportBar")
        transport_layout = QHBoxLayout(transport)
        transport_layout.setContentsMargins(18, 12, 18, 15)
        transport_layout.setSpacing(12)

        self.play_button = MotionButton("播放原片", "secondary")
        self.play_button.setFixedWidth(104)
        self.play_button.setEnabled(False)
        self.play_button.setShortcut("Space")
        self.play_button.clicked.connect(self._toggle_playback)

        self.previous_issue_button = MotionButton("上一异常", "ghost")
        self.previous_issue_button.setFixedWidth(82)
        self.previous_issue_button.setEnabled(False)
        self.previous_issue_button.clicked.connect(lambda: self._jump_issue(-1))
        self.next_issue_button = MotionButton("下一异常", "ghost")
        self.next_issue_button.setFixedWidth(82)
        self.next_issue_button.setEnabled(False)
        self.next_issue_button.clicked.connect(lambda: self._jump_issue(1))

        self.frame_slider = ConfidenceTimeline()
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._frame_changed)
        self.frame_slider.sliderPressed.connect(self._pause_playback)
        self.frame_slider.sliderReleased.connect(self._sync_audio_position)

        self.time_label = QLabel("00:00.0  /  00:00.0")
        self.time_label.setObjectName("timecode")

        transport_layout.addWidget(self.play_button)
        transport_layout.addWidget(self.previous_issue_button)
        transport_layout.addWidget(self.next_issue_button)
        transport_layout.addWidget(self.frame_slider, 1)
        transport_layout.addWidget(self.time_label)

        layout.addWidget(header)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(transport)
        return card

    def _build_inspector(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setObjectName("inspectorScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(372)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(10)

        intro = QFrame()
        intro.setObjectName("inspectorCard")
        intro_layout = QVBoxLayout(intro)
        intro_layout.setContentsMargins(16, 15, 16, 13)
        intro_layout.setSpacing(4)
        intro_eyebrow = QLabel("CONTROL ROOM")
        intro_eyebrow.setObjectName("eyebrow")
        intro_title = QLabel("自动跟拍工作台")
        intro_title.setObjectName("panelTitle")
        intro_copy = QLabel("指定一个舞者，让镜头路径在整段视频中持续跟随。")
        intro_copy.setObjectName("muted")
        intro_copy.setWordWrap(True)
        self.workflow = WorkflowIndicator(["导入", "跟踪", "预览", "导出"])
        intro_layout.addWidget(intro_eyebrow)
        intro_layout.addWidget(intro_title)
        intro_layout.addWidget(intro_copy)
        intro_layout.addWidget(self.workflow)
        layout.addWidget(intro)

        media_card, media_layout = self._section_card("CLIP", "素材")
        self.source_info = QLabel("尚未导入视频")
        self.source_info.setObjectName("sourceInfo")
        self.source_info.setWordWrap(True)
        self.open_button = MotionButton("导入舞蹈视频", "secondary")
        self.open_button.clicked.connect(self._open_video)
        project_buttons = QHBoxLayout()
        self.save_project_button = MotionButton("保存项目", "ghost")
        self.save_project_button.clicked.connect(self._save_project_now)
        self.save_project_button.setEnabled(False)
        self.open_project_button = MotionButton("打开项目", "ghost")
        self.open_project_button.clicked.connect(self._open_project)
        self.undo_button = MotionButton("撤销", "ghost")
        self.undo_button.clicked.connect(self._undo)
        self.undo_button.setEnabled(False)
        self.redo_button = MotionButton("重做", "ghost")
        self.redo_button.clicked.connect(self._redo)
        self.redo_button.setEnabled(False)
        project_buttons.addWidget(self.save_project_button)
        project_buttons.addWidget(self.open_project_button)
        project_buttons.addWidget(self.undo_button)
        project_buttons.addWidget(self.redo_button)
        media_layout.addWidget(self.source_info)
        media_layout.addWidget(self.open_button)
        media_layout.addLayout(project_buttons)
        layout.addWidget(media_card)

        tracking_card, tracking_layout = self._section_card("01 · SUBJECT", "锁定舞者")
        tracking_copy = QLabel("在第一帧拖框完整包住目标。SAM 2 会传播人物掩码并保持遮挡记忆。")
        tracking_copy.setObjectName("muted")
        tracking_copy.setWordWrap(True)
        self.analyze_button = MotionButton("运行 SAM 2 跟踪", "accent")
        self.analyze_button.clicked.connect(self._analyze)
        self.analyze_button.setEnabled(False)
        correction_buttons = QHBoxLayout()
        self.correct_subject_button = MotionButton("修正当前帧人物", "secondary")
        self.correct_subject_button.clicked.connect(self._begin_subject_correction)
        self.correct_subject_button.setEnabled(False)
        self.remove_subject_correction_button = MotionButton("删除人物修正", "ghost")
        self.remove_subject_correction_button.clicked.connect(
            self._remove_subject_correction
        )
        self.remove_subject_correction_button.setEnabled(False)
        correction_buttons.addWidget(self.correct_subject_button, 1)
        correction_buttons.addWidget(self.remove_subject_correction_button)
        self.cancel_operation_button = MotionButton("取消当前操作", "ghost")
        self.cancel_operation_button.clicked.connect(self._cancel_active_operation)
        self.cancel_operation_button.hide()
        tracking_layout.addWidget(tracking_copy)
        tracking_layout.addWidget(self.analyze_button)
        tracking_layout.addLayout(correction_buttons)
        tracking_layout.addWidget(self.cancel_operation_button)
        layout.addWidget(tracking_card)

        framing_card, framing_layout = self._section_card("02 · FRAME", "输出构图")
        ratio_label = QLabel("画幅比例")
        ratio_label.setObjectName("muted")
        self.ratio_combo = SegmentedControl()
        for label, ratio in (
            ("9:16", 9 / 16),
            ("16:9", 16 / 9),
            ("1:1", 1.0),
            ("4:5", 4 / 5),
        ):
            self.ratio_combo.addItem(label, ratio)
        self.ratio_combo.currentIndexAboutToChange.connect(
            lambda _index: self._begin_history_transaction("调整画幅比例")
        )
        self.ratio_combo.currentIndexChanged.connect(self._ratio_changed)

        stabilization_label = QLabel("镜头稳定")
        stabilization_label.setObjectName("muted")
        self.stabilization_combo = SegmentedControl()
        self.stabilization_combo.addItem("稳定", StabilizationPreset.STABLE)
        self.stabilization_combo.addItem("平衡", StabilizationPreset.BALANCED)
        self.stabilization_combo.addItem("灵敏", StabilizationPreset.RESPONSIVE)
        self.stabilization_combo.setCurrentIndex(1)
        self.stabilization_combo.currentIndexChanged.connect(
            self._stabilization_changed
        )
        self.stabilization_combo.currentIndexAboutToChange.connect(
            lambda _index: self._begin_history_transaction("调整镜头稳定")
        )

        smoothing_header = QHBoxLayout()
        smoothing_label = QLabel("镜头缓动")
        smoothing_label.setObjectName("muted")
        self.smoothing_value = QLabel("0.45 s")
        self.smoothing_value.setObjectName("valuePill")
        smoothing_header.addWidget(smoothing_label)
        smoothing_header.addStretch()
        smoothing_header.addWidget(self.smoothing_value)

        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(25, 100)
        self.smoothing_slider.setValue(45)
        self.smoothing_slider.valueChanged.connect(self._smoothing_changed)
        self.smoothing_slider.sliderPressed.connect(
            lambda: self._begin_history_transaction("调整镜头缓动")
        )
        self.smoothing_slider.sliderReleased.connect(self._commit_history_transaction)

        auto_zoom_header = QHBoxLayout()
        auto_zoom_label = QLabel("自动缩放")
        auto_zoom_label.setObjectName("muted")
        self.auto_zoom_combo = SegmentedControl()
        self.auto_zoom_combo.setFixedWidth(138)
        self.auto_zoom_combo.addItem("关闭", False)
        self.auto_zoom_combo.addItem("开启", True)
        self.auto_zoom_combo.setCurrentIndex(1)
        self.auto_zoom_combo.currentIndexChanged.connect(self._framing_changed)
        self.auto_zoom_combo.currentIndexAboutToChange.connect(
            lambda _index: self._begin_history_transaction("切换自动缩放")
        )
        auto_zoom_header.addWidget(auto_zoom_label)
        auto_zoom_header.addStretch()
        auto_zoom_header.addWidget(self.auto_zoom_combo)

        max_zoom_header = QHBoxLayout()
        max_zoom_label = QLabel("最大推近")
        max_zoom_label.setObjectName("muted")
        self.max_zoom_value = QLabel("1.80×")
        self.max_zoom_value.setObjectName("valuePill")
        max_zoom_header.addWidget(max_zoom_label)
        max_zoom_header.addStretch()
        max_zoom_header.addWidget(self.max_zoom_value)
        self.max_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_zoom_slider.setRange(100, 250)
        self.max_zoom_slider.setValue(180)
        self.max_zoom_slider.valueChanged.connect(self._max_zoom_changed)
        self.max_zoom_slider.sliderPressed.connect(
            lambda: self._begin_history_transaction("调整最大推近")
        )
        self.max_zoom_slider.sliderReleased.connect(self._commit_history_transaction)

        framing_layout.addWidget(ratio_label)
        framing_layout.addWidget(self.ratio_combo)
        framing_layout.addWidget(stabilization_label)
        framing_layout.addWidget(self.stabilization_combo)
        framing_layout.addLayout(smoothing_header)
        framing_layout.addWidget(self.smoothing_slider)
        framing_layout.addLayout(auto_zoom_header)
        framing_layout.addLayout(max_zoom_header)
        framing_layout.addWidget(self.max_zoom_slider)
        self.reset_framing_button = MotionButton("恢复构图默认", "ghost")
        self.reset_framing_button.clicked.connect(self._reset_framing_controls)
        framing_layout.addWidget(self.reset_framing_button)
        layout.addWidget(framing_card)

        keyframe_card, keyframe_layout = self._section_card(
            "KEYFRAMES", "镜头关键帧"
        )
        keyframe_copy = QLabel(
            "拖动会立即预览，播放时保持生效；确认后写入对应帧。蓝点会显示在时间轴上。"
        )
        keyframe_copy.setObjectName("muted")
        keyframe_copy.setWordWrap(True)
        keyframe_layout.addWidget(keyframe_copy)

        self.keyframe_x_slider, self.keyframe_x_value = self._labeled_slider(
            keyframe_layout, "水平位置", -240, 240, 0, "0 px"
        )
        self.keyframe_y_slider, self.keyframe_y_value = self._labeled_slider(
            keyframe_layout, "垂直位置", -180, 180, 0, "0 px"
        )
        self.keyframe_zoom_slider, self.keyframe_zoom_value = self._labeled_slider(
            keyframe_layout, "缩放", 100, 250, 100, "1.00×"
        )
        self.keyframe_follow_slider, self.keyframe_follow_value = self._labeled_slider(
            keyframe_layout, "自动跟随", 0, 100, 0, "0%"
        )
        for slider in (
            self.keyframe_x_slider,
            self.keyframe_y_slider,
            self.keyframe_zoom_slider,
            self.keyframe_follow_slider,
        ):
            slider.valueChanged.connect(self._keyframe_editor_changed)
        keyframe_buttons = QHBoxLayout()
        self.add_keyframe_button = MotionButton("写入当前帧", "secondary")
        self.add_keyframe_button.clicked.connect(self._add_camera_keyframe)
        self.add_keyframe_button.setEnabled(False)
        self.remove_keyframe_button = MotionButton("删除当前帧", "secondary")
        self.remove_keyframe_button.clicked.connect(self._remove_camera_keyframe)
        self.remove_keyframe_button.setEnabled(False)
        keyframe_buttons.addWidget(self.add_keyframe_button, 1)
        keyframe_buttons.addWidget(self.remove_keyframe_button)
        keyframe_layout.addLayout(keyframe_buttons)
        self.reset_keyframe_button = MotionButton("恢复关键帧参数", "ghost")
        self.reset_keyframe_button.clicked.connect(self._reset_keyframe_controls)
        self.reset_keyframe_button.setEnabled(False)
        keyframe_layout.addWidget(self.reset_keyframe_button)
        layout.addWidget(keyframe_card)

        output_card, output_layout = self._section_card("03 · OUTPUT", "检查与交付")
        output_copy = QLabel("预览确认后编码为兼容 MP4，并复用原始音频。")
        output_copy.setObjectName("muted")
        output_copy.setWordWrap(True)
        self.export_button = MotionButton("导出跟拍成片", "accent")
        self.export_button.clicked.connect(self._export)
        self.export_button.setEnabled(False)
        self.open_export_button = MotionButton("打开最近导出", "secondary")
        self.open_export_button.clicked.connect(self._open_exported_video)
        self.open_export_button.setEnabled(False)
        output_layout.addWidget(output_copy)
        output_layout.addWidget(self.export_button)
        output_layout.addWidget(self.open_export_button)
        layout.addWidget(output_card)

        status_card = QFrame()
        status_card.setObjectName("statusCard")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(15, 13, 15, 13)
        status_layout.setSpacing(9)
        status_eyebrow = QLabel("SYSTEM STATUS")
        status_eyebrow.setObjectName("eyebrow")
        self.status_label = AnimatedStatusLabel("导入视频后，在第一帧框选要跟随的舞者。")
        self.status_label.setObjectName("muted")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        log_button = MotionButton("查看运行日志", "ghost")
        log_button.clicked.connect(self._open_log)
        status_layout.addWidget(status_eyebrow)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress)
        status_layout.addWidget(log_button)
        layout.addWidget(status_card)
        layout.addStretch()

        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _section_card(number: str, title: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("sectionCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 13, 15, 14)
        layout.setSpacing(10)
        heading = QHBoxLayout()
        number_label = QLabel(number)
        number_label.setObjectName("sectionNumber")
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        heading.addWidget(number_label)
        heading.addStretch()
        heading.addWidget(title_label)
        layout.addLayout(heading)
        return card, layout

    @staticmethod
    def _labeled_slider(
        layout: QVBoxLayout,
        label: str,
        minimum: int,
        maximum: int,
        value: int,
        value_text: str,
    ) -> tuple[QSlider, QLabel]:
        header = QHBoxLayout()
        title = QLabel(label)
        title.setObjectName("muted")
        value_label = QLabel(value_text)
        value_label.setObjectName("valuePill")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(value_label)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        layout.addLayout(header)
        layout.addWidget(slider)
        return slider, value_label

    def _setup_history_actions(self) -> None:
        self.undo_action = QAction("撤销", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self._undo)
        self.addAction(self.undo_action)
        self.redo_action = QAction("重做", self)
        self.redo_action.setShortcuts(QKeySequence.StandardKey.Redo)
        self.redo_action.triggered.connect(self._redo)
        self.addAction(self.redo_action)
        self._update_history_actions()

    def _capture_edit_state(self) -> EditState:
        return EditState(
            subject_prompts=tuple(sorted(self.keyframes.items())),
            tracking=self.tracking_result,
            framing=self._framing_settings(),
            camera_keyframes=tuple(self.camera_keyframes),
            camera_path=self.camera_path,
        )

    def _begin_history_transaction(self, label: str) -> None:
        if self._history_restoring or self._restoring_project or self.busy:
            return
        if self._history_transaction is None:
            self._history_transaction = (label, self._capture_edit_state())

    def _commit_history_transaction(self) -> None:
        if self._history_transaction is None or self._history_restoring:
            return
        label, before = self._history_transaction
        self._history_transaction = None
        self._record_history(label, before)

    def _record_history(self, label: str, before: EditState) -> None:
        if self._history_restoring or self._restoring_project:
            return
        self.edit_history.record(label, before, self._capture_edit_state())
        self._update_history_actions()

    def _update_history_actions(self) -> None:
        undo_label = self.edit_history.undo_label
        redo_label = self.edit_history.redo_label
        enabled = not self.busy
        self.undo_action.setText(f"撤销 {undo_label}" if undo_label else "撤销")
        self.redo_action.setText(f"重做 {redo_label}" if redo_label else "重做")
        self.undo_action.setEnabled(enabled and undo_label is not None)
        self.redo_action.setEnabled(enabled and redo_label is not None)
        self.undo_button.setEnabled(enabled and undo_label is not None)
        self.redo_button.setEnabled(enabled and redo_label is not None)

    def _undo(self) -> None:
        if self.busy:
            return
        if self._preview_keyframe is not None:
            self._discard_keyframe_preview()
            self._update_keyframe_editor()
            self._show_current_frame()
            return
        command = self.edit_history.undo()
        if command is None:
            return
        self._restore_edit_state(command.before)
        self.status_label.setText(f"已撤销：{command.label}")

    def _redo(self) -> None:
        if self.busy:
            return
        command = self.edit_history.redo()
        if command is None:
            return
        self._restore_edit_state(command.after)
        self.status_label.setText(f"已重做：{command.label}")

    def _restore_edit_state(self, state: EditState) -> None:
        self._history_restoring = True
        try:
            self._preview_keyframe = None
            self._preview_camera_path = None
            self.keyframes = dict(state.subject_prompts)
            self.tracking_result = state.tracking
            self.tracked_boxes = state.tracking.boxes if state.tracking else []
            self.camera_keyframes = list(state.camera_keyframes)
            self.camera_path = state.camera_path
            self.crop_path = list(state.camera_path.frames) if state.camera_path else []
            controls = (
                self.ratio_combo,
                self.stabilization_combo,
                self.smoothing_slider,
                self.auto_zoom_combo,
                self.max_zoom_slider,
            )
            for control in controls:
                control.blockSignals(True)
            try:
                ratios = [9 / 16, 16 / 9, 1.0, 4 / 5]
                ratio_index = min(
                    range(len(ratios)),
                    key=lambda index: abs(
                        ratios[index] - state.framing.aspect_ratio
                    ),
                )
                self.ratio_combo.setCurrentIndex(ratio_index)
                self.stabilization_combo.setCurrentIndex(
                    list(StabilizationPreset).index(
                        state.framing.stabilization_preset
                    )
                )
                self.smoothing_slider.setValue(
                    round(state.framing.smoothing_seconds * 100)
                )
                self.auto_zoom_combo.setCurrentIndex(
                    1 if state.framing.auto_zoom else 0
                )
                self.max_zoom_slider.setValue(round(state.framing.max_zoom * 100))
            finally:
                for control in controls:
                    control.blockSignals(False)
            self.smoothing_value.setText(
                f"{state.framing.smoothing_seconds:.2f} s"
            )
            self.max_zoom_value.setText(f"{state.framing.max_zoom:.2f}×")
            if state.tracking is not None:
                self.frame_slider.set_tracking_quality(state.tracking.samples)
            else:
                self.frame_slider.clear_tracking_quality()
            self.frame_slider.set_keyframes(
                keyframe.frame_index for keyframe in self.camera_keyframes
            )
            self.frame_slider.set_subject_corrections(self.keyframes)
            self._update_keyframe_editor()
            self._update_subject_correction_controls()
            self._show_current_frame()
            self._schedule_autosave()
        finally:
            self._history_restoring = False
        self._update_history_actions()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._intro_played:
            return
        self._intro_played = True
        for widget, delay in (
            (self.top_bar, 0),
            (self.stage_card, 70),
            (self.inspector, 130),
        ):
            animation = animate_fade_in(widget, delay)
            if animation is not None:
                self._motion_animations.append(animation)

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择舞蹈视频",
            str(Path.home()),
            "视频文件 (*.mp4 *.mov *.mkv *.avi *.webm);;所有文件 (*)",
        )
        if not path:
            return
        self._load_video(path)

    def _load_video(
        self,
        path: str | Path,
        project: ProjectDocument | None = None,
        restore_autosave: bool = True,
    ) -> None:
        self._pause_playback()
        self.edit_history.clear()
        self._history_transaction = None
        self._pending_edit_before = None
        try:
            info, first_frame = inspect_video(path)
        except Exception as error:
            QMessageBox.critical(self, "无法打开", str(error))
            return

        if self.preview_capture is not None:
            self.preview_capture.release()
        self.preview_capture = cv2.VideoCapture(str(info.path))
        self.media_player.stop()
        self.media_player.setSource(QUrl.fromLocalFile(str(info.path)))
        self.video_info = info
        self.current_frame = first_frame
        self.current_frame_index = 0
        self.keyframes.clear()
        self.camera_keyframes.clear()
        self.tracking_result = None
        self.camera_path = None
        self._preview_keyframe = None
        self._preview_camera_path = None
        self._keyframe_edit_base_center = None
        self._keyframe_edit_frame_index = 0
        self.tracked_boxes.clear()
        self.crop_path.clear()
        self.last_export_path = None
        self.open_export_button.setEnabled(False)
        self.preview_combo.setCurrentIndex(0)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, info.frame_count - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.frame_slider.setEnabled(True)
        self.play_button.setEnabled(True)
        self.analyze_button.setEnabled(False)
        self.correct_subject_button.setEnabled(False)
        self.remove_subject_correction_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.add_keyframe_button.setEnabled(False)
        self.reset_keyframe_button.setEnabled(False)
        self.remove_keyframe_button.setEnabled(False)
        self.previous_issue_button.setEnabled(False)
        self.next_issue_button.setEnabled(False)
        self.save_project_button.setEnabled(True)
        self.frame_slider.clear_tracking_quality()
        self.frame_slider.set_keyframes(())
        self.frame_slider.set_subject_corrections(())

        if project is None and restore_autosave:
            try:
                project = self.project_store.load_for_video(info)
            except Exception as error:
                logging.warning("Could not restore video project: %s", error)
        self.project_document = project or ProjectDocument(source=source_ref(info))

        self.project_label.setText(info.path.name.upper())
        self.video_meta_label.setText(f"{info.width} × {info.height}  /  {info.fps:.2f} FPS")
        self.source_info.setText(
            f"{info.path.name}\n{self._format_time(info.duration)} · "
            f"{info.width}×{info.height} · {info.frame_count} 帧"
        )
        self.workflow.setStep(1)
        if project is not None:
            self._apply_project(project)
            self.status_label.setText("项目已恢复，跟踪结果和镜头编辑无需重新计算。")
        else:
            self.status_label.setText("素材已就绪。请在第一帧拖框，完整框住要跟随的舞者。")
        logging.info(
            "Video opened: %s (%dx%d, %.3f fps, %d frames)",
            path,
            info.width,
            info.height,
            info.fps,
            info.frame_count,
        )
        self._show_current_frame()
        animation = animate_refresh(self.canvas)
        if animation is not None:
            self._motion_animations.append(animation)
        self._update_history_actions()

    def _apply_project(self, document: ProjectDocument) -> None:
        if self.video_info is None:
            return
        self._restoring_project = True
        try:
            self.keyframes = dict(document.subject_prompts)
            self.camera_keyframes = list(document.camera_keyframes)
            self.tracking_result = document.tracking
            self.tracked_boxes = (
                document.tracking.boxes if document.tracking is not None else []
            )
            self.camera_path = document.camera_path
            self.crop_path = (
                list(document.camera_path.frames)
                if document.camera_path is not None
                else []
            )

            ratios = [9 / 16, 16 / 9, 1.0, 4 / 5]
            ratio_index = min(
                range(len(ratios)),
                key=lambda index: abs(ratios[index] - document.framing.aspect_ratio),
            )
            self.ratio_combo.setCurrentIndex(ratio_index)
            self.smoothing_slider.setValue(
                round(document.framing.smoothing_seconds * 100)
            )
            presets = list(StabilizationPreset)
            self.stabilization_combo.setCurrentIndex(
                presets.index(document.framing.stabilization_preset)
            )
            self.auto_zoom_combo.setCurrentIndex(
                1 if document.framing.auto_zoom else 0
            )
            self.max_zoom_slider.setValue(round(document.framing.max_zoom * 100))

            if self.tracking_result is not None:
                self.frame_slider.set_tracking_quality(self.tracking_result.samples)
                self.preview_combo.setCurrentIndex(1)
                self.workflow.setStep(2)
                self.export_button.setEnabled(bool(self.crop_path))
                self.add_keyframe_button.setEnabled(True)
                self.reset_keyframe_button.setEnabled(True)
                has_issues = any(
                    sample.tracking_confidence < 0.42
                    for sample in self.tracking_result.samples
                )
                self.previous_issue_button.setEnabled(has_issues)
                self.next_issue_button.setEnabled(has_issues)
            self.frame_slider.set_keyframes(
                keyframe.frame_index for keyframe in self.camera_keyframes
            )
            self.frame_slider.set_subject_corrections(self.keyframes)
            self.analyze_button.setEnabled(0 in self.keyframes)
            frame_index = min(
                max(document.playhead_frame, 0), self.video_info.frame_count - 1
            )
            self.frame_slider.setValue(frame_index)
            if self.tracking_result is not None:
                self._rebuild_crop_path()
            else:
                self._update_keyframe_editor()
        finally:
            self._restoring_project = False
        self._update_subject_correction_controls()

    def _project_snapshot(self) -> ProjectDocument | None:
        if self.video_info is None or self.project_document is None:
            return None
        prompts = self._pending_prompts_before or self.keyframes
        self.project_document.subject_prompts = dict(prompts)
        self.project_document.tracking = self.tracking_result
        self.project_document.tracking_prompt_hash = (
            subject_prompt_hash(dict(prompts))
            if self.tracking_result is not None
            else None
        )
        self.project_document.framing = self._framing_settings()
        self.project_document.camera_keyframes = list(self.camera_keyframes)
        self.project_document.camera_path = self.camera_path
        self.project_document.playhead_frame = self.current_frame_index
        return self.project_document

    def _schedule_autosave(self) -> None:
        if not self._restoring_project and self.project_document is not None:
            self.autosave_timer.start()

    def _save_project(self) -> None:
        document = self._project_snapshot()
        if document is None:
            return
        try:
            path = self.project_store.save(document)
            logging.info("Project saved: %s", path)
        except Exception:
            logging.exception("Project autosave failed")

    def _save_project_now(self) -> None:
        self._save_project()
        if self.project_document is not None:
            path = self.project_store.autosave_path(self.project_document)
            self.status_label.setText(f"项目已保存：{path.name}")

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开 Dance Focus 项目",
            str(self.project_store.projects_dir),
            "Dance Focus 项目 (*.dancefocus.json);;JSON 文件 (*.json)",
        )
        if not path:
            return
        try:
            document = self.project_store.load(path)
            self._load_video(document.source.path, document, restore_autosave=False)
        except Exception as error:
            self._operation_error(str(error))

    def _selection_changed(self, box: Box) -> None:
        if self.video_info is None:
            return
        self._pause_playback()
        old_prompts = dict(self.keyframes)
        new_prompts = dict(self.keyframes)
        new_prompts[self.current_frame_index] = box
        if self.tracking_result is not None:
            self._pending_edit_before = self._capture_edit_state()
            start, end = affected_correction_interval(
                old_prompts,
                new_prompts,
                self.current_frame_index,
                self.video_info.frame_count,
            )
            self.keyframes = new_prompts
            self.frame_slider.set_subject_corrections(self.keyframes)
            self._show_current_frame()
            self._start_correction_analysis(old_prompts, start, end)
            return

        self.keyframes = new_prompts
        if self.current_frame_index == 0:
            message = "目标框已记录。可以运行 SAM 2 自动跟踪。"
            self.analyze_button.setEnabled(not self.busy)
        else:
            message = f"已在第 {self.current_frame_index + 1} 帧加入修正，请重新运行跟踪。"
        self.tracking_result = None
        self.camera_path = None
        self.tracked_boxes.clear()
        self.crop_path.clear()
        self.camera_keyframes.clear()
        self.frame_slider.clear_tracking_quality()
        self.frame_slider.set_keyframes(())
        self.frame_slider.set_subject_corrections(self.keyframes)
        self.preview_combo.setCurrentIndex(0)
        self.export_button.setEnabled(False)
        self.add_keyframe_button.setEnabled(False)
        self.reset_keyframe_button.setEnabled(False)
        self.previous_issue_button.setEnabled(False)
        self.next_issue_button.setEnabled(False)
        self.workflow.setStep(1)
        self.status_label.setText(message)
        self._show_current_frame()
        self._schedule_autosave()

    def _begin_subject_correction(self) -> None:
        if self.video_info is None or self.busy:
            return
        self._pause_playback()
        self.preview_combo.setCurrentIndex(0)
        self.status_label.setText(
            f"请在第 {self.current_frame_index + 1} 帧重新完整框住目标舞者。"
        )

    def _remove_subject_correction(self) -> None:
        frame = self.current_frame_index
        if self.video_info is None or frame == 0 or frame not in self.keyframes:
            return
        old_prompts = dict(self.keyframes)
        new_prompts = dict(self.keyframes)
        del new_prompts[frame]
        if self.tracking_result is None:
            self.keyframes = new_prompts
            self.frame_slider.set_subject_corrections(self.keyframes)
            self._update_subject_correction_controls()
            self._schedule_autosave()
            return
        start, end = affected_correction_interval(
            old_prompts, new_prompts, frame, self.video_info.frame_count
        )
        self._pending_edit_before = self._capture_edit_state()
        self.keyframes = new_prompts
        self.frame_slider.set_subject_corrections(self.keyframes)
        self._start_correction_analysis(old_prompts, start, end)

    def _start_correction_analysis(
        self,
        old_prompts: dict[int, Box],
        start: int,
        end: int,
    ) -> None:
        if self.video_info is None or self.tracking_result is None:
            return
        self._pending_prompts_before = old_prompts
        self._pending_correction_interval = (start, end)
        self._set_busy(
            True,
            f"正在局部重算第 {start + 1}–{end} 帧，原跟踪结果会保留到成功完成。",
        )
        worker = FunctionWorker(
            reanalyze_subject_interval_isolated,
            self.video_info,
            dict(self.keyframes),
            self.tracking_result,
            start,
            end,
        )
        worker.signals.progress.connect(self.progress.setValue)
        worker.signals.result.connect(self._correction_complete)
        worker.signals.error.connect(self._correction_error)
        worker.signals.finished.connect(lambda: self._set_busy(False))
        self._start_worker(worker)

    def _correction_complete(self, result: TrackingResult) -> None:
        interval = self._pending_correction_interval
        edit_before = self._pending_edit_before
        self._pending_edit_before = None
        self._pending_prompts_before = None
        self._pending_correction_interval = None
        self.tracking_result = result
        self.tracked_boxes = result.boxes
        self.frame_slider.set_tracking_quality(result.samples)
        self._rebuild_crop_path()
        self.preview_combo.setCurrentIndex(1)
        self._update_subject_correction_controls()
        if edit_before is not None:
            self._record_history("修正人物跟踪", edit_before)
        if interval is not None:
            start, end = interval
            self.status_label.setText(
                f"人物修正完成，仅重算了第 {start + 1}–{end} 帧。"
            )
        self._schedule_autosave()

    def _correction_error(self, message: str) -> None:
        if self._pending_prompts_before is not None:
            self.keyframes = self._pending_prompts_before
        self._pending_prompts_before = None
        self._pending_correction_interval = None
        self._pending_edit_before = None
        self.frame_slider.set_subject_corrections(self.keyframes)
        self._show_current_frame()
        if "InterruptedError" in message or "操作已取消" in message:
            self.status_label.setText("已取消局部重新跟踪，原结果保持不变。")
            return
        self._operation_error(message)

    def _update_subject_correction_controls(self) -> None:
        has_tracking = self.tracking_result is not None
        self.correct_subject_button.setEnabled(
            not self.busy and self.video_info is not None and has_tracking
        )
        self.remove_subject_correction_button.setEnabled(
            not self.busy
            and self.current_frame_index != 0
            and self.current_frame_index in self.keyframes
        )

    def _analyze(self) -> None:
        if self.video_info is None or 0 not in self.keyframes or self.busy:
            return
        self.workflow.setStep(1)
        self._set_busy(
            True,
            "SAM 2 正在准备帧缓存并传播人物掩码。首次运行会下载 184 MB 官方模型。",
        )
        worker = FunctionWorker(
            analyze_subject_isolated, self.video_info, dict(self.keyframes)
        )
        worker.signals.progress.connect(self.progress.setValue)
        worker.signals.result.connect(self._analysis_complete)
        worker.signals.error.connect(self._operation_error)
        worker.signals.finished.connect(lambda: self._set_busy(False))
        self._start_worker(worker)

    def _analysis_complete(self, result: TrackingResult) -> None:
        self.tracking_result = result
        self.tracked_boxes = result.boxes
        self.frame_slider.set_tracking_quality(result.samples)
        self._rebuild_crop_path()
        self.preview_combo.setCurrentIndex(1)
        self.workflow.setStep(2)
        flagged = sum(
            sample.tracking_confidence < 0.42 for sample in result.samples
        )
        if flagged:
            self.status_label.setText(
                f"智能分析完成，时间轴标出 {flagged} 个低质量帧。优先检查红色和橙色区域。"
            )
        else:
            self.status_label.setText("姿态与身份分析完成。已进入自动缩放成片预览。")
        self.export_button.setEnabled(True)
        self.add_keyframe_button.setEnabled(True)
        self.reset_keyframe_button.setEnabled(True)
        self.correct_subject_button.setEnabled(True)
        self.frame_slider.set_subject_corrections(self.keyframes)
        self.previous_issue_button.setEnabled(flagged > 0)
        self.next_issue_button.setEnabled(flagged > 0)
        self._show_current_frame()
        self._schedule_autosave()

    def _rebuild_crop_path(self) -> None:
        if self.video_info is None or self.tracking_result is None:
            return
        try:
            self.camera_path = build_camera_path(
                self.tracking_result,
                (self.video_info.width, self.video_info.height),
                self.video_info.fps,
                self._framing_settings(),
                self.camera_keyframes,
            )
            self._preview_keyframe = None
            self._preview_camera_path = None
            self.crop_path = list(self.camera_path.frames)
            self.export_button.setEnabled(not self.busy)
        except Exception as error:
            self._operation_error(str(error))
            return
        self._show_current_frame()
        self.frame_slider.set_keyframes(
            keyframe.frame_index for keyframe in self.camera_keyframes
        )
        self._update_keyframe_editor()
        self._schedule_autosave()

    def _framing_settings(self) -> FramingSettings:
        return FramingSettings(
            aspect_ratio=float(self.ratio_combo.currentData()),
            smoothing_seconds=self.smoothing_slider.value() / 100,
            auto_zoom=bool(self.auto_zoom_combo.currentData()),
            max_zoom=self.max_zoom_slider.value() / 100,
            stabilization_preset=StabilizationPreset(
                self.stabilization_combo.currentData()
            ),
        )

    def _stabilization_changed(self, _index: int) -> None:
        smoothing = {
            StabilizationPreset.STABLE: 75,
            StabilizationPreset.BALANCED: 45,
            StabilizationPreset.RESPONSIVE: 25,
        }[StabilizationPreset(self.stabilization_combo.currentData())]
        self.smoothing_slider.blockSignals(True)
        self.smoothing_slider.setValue(smoothing)
        self.smoothing_slider.blockSignals(False)
        self.smoothing_value.setText(f"{smoothing / 100:.2f} s")
        self._rebuild_crop_path()
        self._commit_history_transaction()

    def _smoothing_changed(self, value: int) -> None:
        self.smoothing_value.setText(f"{value / 100:.2f} s")
        self._rebuild_crop_path()

    def _ratio_changed(self, _index: int) -> None:
        self._rebuild_crop_path()
        self._commit_history_transaction()

    def _framing_changed(self, _index: int = 0) -> None:
        self._rebuild_crop_path()
        self._commit_history_transaction()

    def _max_zoom_changed(self, value: int) -> None:
        self.max_zoom_value.setText(f"{value / 100:.2f}×")
        self._rebuild_crop_path()

    def _reset_framing_controls(self) -> None:
        before = self._capture_edit_state()
        controls = (
            self.ratio_combo,
            self.stabilization_combo,
            self.smoothing_slider,
            self.auto_zoom_combo,
            self.max_zoom_slider,
        )
        for control in controls:
            control.blockSignals(True)
        try:
            self.ratio_combo.setCurrentIndex(0)
            self.stabilization_combo.setCurrentIndex(1)
            self.smoothing_slider.setValue(45)
            self.auto_zoom_combo.setCurrentIndex(1)
            self.max_zoom_slider.setValue(180)
        finally:
            for control in controls:
                control.blockSignals(False)
        self.smoothing_value.setText("0.45 s")
        self.max_zoom_value.setText("1.80×")
        self._rebuild_crop_path()
        self.status_label.setText(
            "已恢复构图默认：9:16、0.45 秒缓动、自动缩放和 1.80× 推近。"
        )
        self._record_history("恢复构图默认", before)

    def _reset_keyframe_controls(self) -> None:
        zoom = 1.0
        base_path = self._camera_path_without_current_keyframe()
        if (
            base_path is not None
            and self._keyframe_edit_frame_index < len(base_path.frames)
        ):
            crop = base_path.frames[self._keyframe_edit_frame_index]
            self._keyframe_edit_base_center = Point(
                crop.x + crop.width / 2,
                crop.y + crop.height / 2,
            )
            zoom = base_path.output_size[1] / max(crop.height, 1)

        self._updating_keyframe_editor = True
        try:
            self.keyframe_x_slider.setValue(0)
            self.keyframe_y_slider.setValue(0)
            self.keyframe_zoom_slider.setValue(round(zoom * 100))
            self.keyframe_follow_slider.setValue(100)
        finally:
            self._updating_keyframe_editor = False
        self._update_keyframe_labels()
        self._keyframe_editor_changed(0)
        self.status_label.setText(
            "关键帧位置、缩放和自动跟随已恢复为当前镜头默认值。"
        )

    def _add_camera_keyframe(self) -> None:
        before = self._capture_edit_state()
        keyframe = self._preview_keyframe or self._keyframe_from_editor()
        if keyframe is None:
            return
        target_frame = keyframe.frame_index
        self.camera_keyframes = [
            item
            for item in self.camera_keyframes
            if item.frame_index != target_frame
        ]
        self.camera_keyframes.append(keyframe)
        self.camera_keyframes.sort(key=lambda item: item.frame_index)
        self._preview_keyframe = None
        self._rebuild_crop_path()
        self._update_keyframe_editor()
        self.status_label.setText(
            f"已在第 {target_frame + 1} 帧写入镜头关键帧。"
        )
        self._record_history("写入镜头关键帧", before)

    def _remove_camera_keyframe(self) -> None:
        if self._preview_keyframe is not None:
            preview_frame = self._preview_keyframe.frame_index
            self._discard_keyframe_preview()
            self._update_keyframe_editor()
            self._show_current_frame()
            self.status_label.setText(
                f"已取消第 {preview_frame + 1} 帧尚未写入的镜头预览。"
            )
            return
        before = self._capture_edit_state()
        self._discard_keyframe_preview()
        before = len(self.camera_keyframes)
        self.camera_keyframes = [
            item
            for item in self.camera_keyframes
            if item.frame_index != self.current_frame_index
        ]
        if len(self.camera_keyframes) == before:
            self._show_current_frame()
            return
        self._rebuild_crop_path()
        self._update_keyframe_editor()
        self.status_label.setText(
            f"已删除第 {self.current_frame_index + 1} 帧的镜头关键帧。"
        )
        self._record_history("删除镜头关键帧", before)

    def _update_keyframe_editor(self) -> None:
        keyframe = next(
            (
                item
                for item in self.camera_keyframes
                if item.frame_index == self.current_frame_index
            ),
            None,
        )
        if self._preview_keyframe is not None:
            preview_frame = self._preview_keyframe.frame_index
            self.add_keyframe_button.setText(f"写入第 {preview_frame + 1} 帧")
            self.remove_keyframe_button.setText("取消预览")
            self.remove_keyframe_button.setEnabled(not self.busy)
            return
        self.remove_keyframe_button.setText("删除当前帧")
        self.remove_keyframe_button.setEnabled(keyframe is not None and not self.busy)
        self.add_keyframe_button.setText(
            "更新当前帧" if keyframe is not None else "写入当前帧"
        )
        self._keyframe_edit_frame_index = self.current_frame_index
        base_path = self._camera_path_without_current_keyframe()
        if base_path is None or self.current_frame_index >= len(base_path.frames):
            self._keyframe_edit_base_center = None
            return
        base_crop = base_path.frames[self.current_frame_index]
        base_center = Point(
            base_crop.x + base_crop.width / 2,
            base_crop.y + base_crop.height / 2,
        )
        self._keyframe_edit_base_center = base_center
        if keyframe is None:
            x_offset = 0
            y_offset = 0
            zoom = base_path.output_size[1] / max(base_crop.height, 1)
            follow = 1.0
        else:
            center = keyframe.center or base_center
            x_offset = round(center.x - base_center.x)
            y_offset = round(center.y - base_center.y)
            zoom = keyframe.zoom if keyframe.zoom is not None else 1.0
            follow = (
                keyframe.follow_strength
                if keyframe.follow_strength is not None
                else 1.0
            )

        self._updating_keyframe_editor = True
        try:
            self.keyframe_x_slider.setValue(x_offset)
            self.keyframe_y_slider.setValue(y_offset)
            self.keyframe_zoom_slider.setValue(round(zoom * 100))
            self.keyframe_follow_slider.setValue(round(follow * 100))
        finally:
            self._updating_keyframe_editor = False
        self._update_keyframe_labels()

    def _camera_path_without_current_keyframe(self) -> CameraPath | None:
        if self.video_info is None or self.tracking_result is None:
            return None
        keyframes = [
            item
            for item in self.camera_keyframes
            if item.frame_index != self._keyframe_edit_frame_index
        ]
        try:
            return build_camera_path(
                self.tracking_result,
                (self.video_info.width, self.video_info.height),
                self.video_info.fps,
                self._framing_settings(),
                keyframes,
            )
        except Exception:
            logging.exception("Could not build keyframe editing baseline")
            return None

    def _keyframe_from_editor(self) -> CameraKeyframe | None:
        if self._keyframe_edit_base_center is None:
            return None
        return CameraKeyframe(
            frame_index=self._keyframe_edit_frame_index,
            center=Point(
                self._keyframe_edit_base_center.x + self.keyframe_x_slider.value(),
                self._keyframe_edit_base_center.y + self.keyframe_y_slider.value(),
            ),
            zoom=self.keyframe_zoom_slider.value() / 100,
            follow_strength=self.keyframe_follow_slider.value() / 100,
        )

    def _update_keyframe_labels(self) -> None:
        self.keyframe_x_value.setText(f"{self.keyframe_x_slider.value():+d} px")
        self.keyframe_y_value.setText(f"{self.keyframe_y_slider.value():+d} px")
        self.keyframe_zoom_value.setText(
            f"{self.keyframe_zoom_slider.value() / 100:.2f}×"
        )
        self.keyframe_follow_value.setText(
            f"{self.keyframe_follow_slider.value()}%"
        )

    def _keyframe_editor_changed(self, _value: int) -> None:
        self._update_keyframe_labels()
        if self._updating_keyframe_editor or self.busy:
            return
        keyframe = self._keyframe_from_editor()
        if (
            keyframe is None
            or self.video_info is None
            or self.tracking_result is None
        ):
            return
        keyframes = [
            item
            for item in self.camera_keyframes
            if item.frame_index != keyframe.frame_index
        ]
        keyframes.append(keyframe)
        try:
            self._preview_camera_path = build_camera_path(
                self.tracking_result,
                (self.video_info.width, self.video_info.height),
                self.video_info.fps,
                self._framing_settings(),
                keyframes,
            )
        except Exception as error:
            self._operation_error(str(error))
            return
        self._preview_keyframe = keyframe
        self.crop_path = list(self._preview_camera_path.frames)
        self.export_button.setEnabled(False)
        self._update_keyframe_editor()
        if self.preview_combo.currentData() != "crop":
            self.preview_combo.setCurrentIndex(1)
        else:
            self._show_current_frame()
        self.status_label.setText("正在预览当前帧调整；点击“写入当前帧”保存。")

    def _discard_keyframe_preview(self) -> None:
        if self._preview_camera_path is None:
            return
        self._preview_keyframe = None
        self._preview_camera_path = None
        self.crop_path = list(self.camera_path.frames) if self.camera_path else []
        self.export_button.setEnabled(not self.busy and self.camera_path is not None)

    def _preview_mode_changed(self) -> None:
        self._update_play_button_text()
        self._show_current_frame()
        animation = animate_refresh(self.canvas)
        if animation is not None:
            self._motion_animations.append(animation)

    def _frame_changed(self, frame_index: int) -> None:
        if self.video_info is None:
            return
        self.current_frame_index = frame_index
        frame = self._read_preview_frame(frame_index)
        if frame is None:
            logging.error("Preview frame decode failed: frame=%d", frame_index)
            self._pause_playback()
            self.status_label.setText(
                f"无法读取第 {frame_index + 1} 帧。请打开运行日志查看详情。"
            )
            return
        self.current_frame = frame
        self._update_keyframe_editor()
        self._update_subject_correction_controls()
        self._show_current_frame()

    def _jump_issue(self, direction: int) -> None:
        if self.tracking_result is None:
            return
        issues = [
            index
            for index, sample in enumerate(self.tracking_result.samples)
            if sample.tracking_confidence < 0.42
        ]
        if not issues:
            return
        self._pause_playback()
        if direction > 0:
            destination = next(
                (index for index in issues if index > self.current_frame_index),
                issues[0],
            )
        else:
            destination = next(
                (index for index in reversed(issues) if index < self.current_frame_index),
                issues[-1],
            )
        self.frame_slider.setValue(destination)
        self.status_label.setText(
            f"已定位到第 {destination + 1} 帧的低质量跟踪区域。"
        )

    def _read_preview_frame(self, frame_index: int):
        if self.video_info is None:
            return None
        frames_dir = cached_frames_dir(self.video_info)
        if frames_dir is not None:
            cached = cv2.imread(str(frames_dir / f"{frame_index:06d}.jpg"))
            if cached is not None:
                return cached
        if self.preview_capture is None or not self.preview_capture.isOpened():
            self.preview_capture = cv2.VideoCapture(str(self.video_info.path))
        expected_position = int(round(self.preview_capture.get(cv2.CAP_PROP_POS_FRAMES)))
        if expected_position != frame_index:
            self.preview_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.preview_capture.read()
        return frame if ok else None

    def _toggle_playback(self) -> None:
        if self.play_timer.isActive():
            self._pause_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        if self.video_info is None or self.busy:
            return
        if self.current_frame_index >= self.video_info.frame_count - 1:
            self.frame_slider.setValue(0)
        self._sync_audio_position()
        self.media_player.play()
        interval = max(1, round(1000 / self.video_info.fps))
        self.play_timer.start(interval)
        self.play_button.setText("暂停")
        mode = "成片" if self.preview_combo.currentData() == "crop" else "原片"
        self.status_label.setText(f"正在播放{mode}。空格键可快速暂停。")
        logging.info("Playback started: mode=%s frame=%d", mode, self.current_frame_index)

    def _pause_playback(self) -> None:
        self.play_timer.stop()
        self.media_player.pause()
        self._update_play_button_text()

    def _update_play_button_text(self) -> None:
        if self.play_timer.isActive():
            self.play_button.setText("暂停")
        elif self.preview_combo.currentData() == "crop" and self.crop_path:
            self.play_button.setText("播放成片")
        else:
            self.play_button.setText("播放原片")

    def _advance_playback(self) -> None:
        if self.video_info is None:
            self._pause_playback()
            return
        next_frame = self.current_frame_index + 1
        if next_frame >= self.video_info.frame_count:
            self._pause_playback()
            if self._preview_keyframe is not None:
                self.status_label.setText(
                    f"播放结束。第 {self._preview_keyframe.frame_index + 1} 帧的"
                    "镜头调整仍在预览中，请写入或取消。"
                )
            else:
                self.status_label.setText("播放结束。可以调整构图参数或直接导出。")
            return
        self.frame_slider.setValue(next_frame)

    def _sync_audio_position(self) -> None:
        if self.video_info is None:
            return
        position_ms = round(self.current_frame_index * 1000 / self.video_info.fps)
        self.media_player.setPosition(position_ms)

    def _media_error(self, _, error_string: str) -> None:
        if not error_string:
            return
        logging.warning("Qt audio playback failed: %s", error_string)
        self.status_label.setText(f"音频播放失败，但画面仍可播放：{error_string}")

    def _show_current_frame(self) -> None:
        if self.video_info is None:
            self.canvas.set_frame(None)
            return
        box = self.keyframes.get(self.current_frame_index)
        if box is None and self.current_frame_index < len(self.tracked_boxes):
            box = self.tracked_boxes[self.current_frame_index]
        crop = (
            self.crop_path[self.current_frame_index]
            if self.current_frame_index < len(self.crop_path)
            else None
        )
        pose_anchor = (
            self.tracking_result.samples[self.current_frame_index].anchor
            if self.tracking_result is not None
            and self.current_frame_index < len(self.tracking_result.samples)
            else None
        )
        self.canvas.set_crop_preview(
            self.preview_combo.currentData() == "crop" and crop is not None
        )
        self.canvas.set_frame(self.current_frame, box, crop, pose_anchor)
        current = self.current_frame_index / self.video_info.fps
        self.time_label.setText(
            f"{self._format_time(current)}  /  {self._format_time(self.video_info.duration)}"
        )

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        remainder = seconds - minutes * 60
        return f"{minutes:02d}:{remainder:04.1f}"

    def _export(self) -> None:
        if self.video_info is None or self.camera_path is None or self.busy:
            return
        default_name = self.video_info.path.with_name(
            f"{self.video_info.path.stem}_focus.mp4"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "导出跟拍视频", str(default_name), "MP4 视频 (*.mp4)"
        )
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"

        self.workflow.setStep(3)
        self._set_busy(True, "正在编码兼容 MP4、复用原音频并验证输出文件。")
        worker = FunctionWorker(export_video, self.video_info, self.camera_path, path)
        worker.signals.progress.connect(self.progress.setValue)
        worker.signals.result.connect(self._export_complete)
        worker.signals.error.connect(self._operation_error)
        worker.signals.finished.connect(lambda: self._set_busy(False))
        self._start_worker(worker)

    def _export_complete(self, output_path: Path) -> None:
        self.last_export_path = Path(output_path)
        self.open_export_button.setEnabled(True)
        self.workflow.setStep(3)
        self.status_label.setText(f"导出完成并通过播放验证：{self.last_export_path.name}")
        self._schedule_autosave()
        QMessageBox.information(
            self,
            "导出完成",
            f"视频已生成并验证可以播放：\n{self.last_export_path}\n\n"
            "点击“打开最近导出”立即播放。",
        )

    def _open_exported_video(self) -> None:
        if self.last_export_path is None or not self.last_export_path.exists():
            self._operation_error("找不到已导出的视频文件")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_export_path))):
            self._operation_error("系统没有可用于打开 MP4 的播放器")

    def _open_log(self) -> None:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _start_worker(self, worker: FunctionWorker) -> None:
        self.active_workers.add(worker)
        self.current_worker = worker
        worker.signals.finished.connect(lambda current=worker: self._worker_finished(current))
        self.thread_pool.start(worker)

    def _worker_finished(self, worker: FunctionWorker) -> None:
        self.active_workers.discard(worker)
        if self.current_worker is worker:
            self.current_worker = None

    def _cancel_active_operation(self) -> None:
        if self.current_worker is None:
            return
        self.current_worker.cancel()
        self.cancel_operation_button.setEnabled(False)
        self.status_label.setText("正在安全取消当前操作…")

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        if busy:
            self._pause_playback()
        self.busy = busy
        self.open_button.setEnabled(not busy)
        self.open_project_button.setEnabled(not busy)
        self.save_project_button.setEnabled(not busy and self.video_info is not None)
        self.analyze_button.setEnabled(not busy and 0 in self.keyframes)
        self.export_button.setEnabled(
            not busy
            and self.camera_path is not None
            and self._preview_keyframe is None
        )
        self.frame_slider.setEnabled(not busy and self.video_info is not None)
        self.play_button.setEnabled(not busy and self.video_info is not None)
        self.ratio_combo.setEnabled(not busy)
        self.stabilization_combo.setEnabled(not busy)
        self.smoothing_slider.setEnabled(not busy)
        self.auto_zoom_combo.setEnabled(not busy)
        self.max_zoom_slider.setEnabled(not busy)
        self.reset_framing_button.setEnabled(not busy)
        self.preview_combo.setEnabled(not busy)
        self.cancel_operation_button.setVisible(busy)
        self.cancel_operation_button.setEnabled(busy)
        self.add_keyframe_button.setEnabled(
            not busy and self.tracking_result is not None
        )
        self.reset_keyframe_button.setEnabled(
            not busy and self.tracking_result is not None
        )
        has_issues = (
            self.tracking_result is not None
            and any(
                sample.tracking_confidence < 0.42
                for sample in self.tracking_result.samples
            )
        )
        self.previous_issue_button.setEnabled(not busy and has_issues)
        self.next_issue_button.setEnabled(not busy and has_issues)
        self._update_subject_correction_controls()
        self._update_keyframe_editor()
        self.open_export_button.setEnabled(
            not busy
            and self.last_export_path is not None
            and self.last_export_path.exists()
        )
        if busy:
            self.progress.setValue(0)
            self.progress.show()
        else:
            self.progress.hide()
        if status:
            self.status_label.setText(status)
        self._update_history_actions()

    def _operation_error(self, message: str) -> None:
        logging.error("Operation failed: %s", message)
        QMessageBox.critical(self, "操作失败", message)
        self.status_label.setText(message)

    def closeEvent(self, event) -> None:
        self._pause_playback()
        for worker in tuple(self.active_workers):
            worker.cancel()
        self.autosave_timer.stop()
        self._save_project()
        self.project_store.clear_last()
        if self.preview_capture is not None:
            self.preview_capture.release()
        super().closeEvent(event)


def main() -> int:
    configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("Dance Focus")
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.black)
    app.setPalette(palette)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
