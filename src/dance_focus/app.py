from __future__ import annotations

from pathlib import Path
import logging
import sys

import cv2
from PySide6.QtCore import QThreadPool, QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPalette
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from dance_focus.canvas import VideoCanvas
from dance_focus.diagnostics import configure_logging, log_path
from dance_focus.exporter import export_video
from dance_focus.geometry import Box, CropRect, build_crop_path, crop_size_for_aspect
from dance_focus.sam2_tracker import cached_frames_dir, runtime_description, track_subject
from dance_focus.video import VideoInfo, inspect_video
from dance_focus.workers import FunctionWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dance Focus - 舞蹈自动跟拍")
        self.resize(1180, 720)
        self.thread_pool = QThreadPool.globalInstance()
        self.active_workers: set[FunctionWorker] = set()
        self.video_info: VideoInfo | None = None
        self.current_frame = None
        self.current_frame_index = 0
        self.keyframes: dict[int, Box] = {}
        self.tracked_boxes: list[Box | None] = []
        self.crop_path: list[CropRect] = []
        self.busy = False
        self.preview_capture = None
        self.last_export_path: Path | None = None

        self.play_timer = QTimer(self)
        self.play_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.play_timer.timeout.connect(self._advance_playback)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.8)
        self.media_player = QMediaPlayer(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.errorOccurred.connect(self._media_error)

        self.canvas = VideoCanvas()
        self.canvas.selection_changed.connect(self._selection_changed)

        self.open_button = QPushButton("打开视频")
        self.open_button.clicked.connect(self._open_video)
        self.analyze_button = QPushButton("使用 SAM 2 自动追踪")
        self.analyze_button.clicked.connect(self._analyze)
        self.analyze_button.setEnabled(False)
        self.export_button = QPushButton("导出视频")
        self.export_button.clicked.connect(self._export)
        self.export_button.setEnabled(False)
        self.open_export_button = QPushButton("打开导出视频")
        self.open_export_button.clicked.connect(self._open_exported_video)
        self.open_export_button.setEnabled(False)

        self.ratio_combo = QComboBox()
        for label, ratio in (
            ("竖屏 9:16", 9 / 16),
            ("横屏 16:9", 16 / 9),
            ("方形 1:1", 1.0),
            ("竖屏 4:5", 4 / 5),
        ):
            self.ratio_combo.addItem(label, ratio)
        self.ratio_combo.currentIndexChanged.connect(self._rebuild_crop_path)

        self.preview_combo = QComboBox()
        self.preview_combo.addItem("原视频 + 裁剪框", "source")
        self.preview_combo.addItem("裁剪结果（成片）", "crop")
        self.preview_combo.currentIndexChanged.connect(self._preview_mode_changed)

        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(10, 100)
        self.smoothing_slider.setValue(45)
        self.smoothing_slider.valueChanged.connect(self._smoothing_changed)
        self.smoothing_value = QLabel("0.45 秒")

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._frame_changed)
        self.frame_slider.sliderPressed.connect(self._pause_playback)
        self.frame_slider.sliderReleased.connect(self._sync_audio_position)
        self.play_button = QPushButton("播放原片")
        self.play_button.setFixedWidth(96)
        self.play_button.setEnabled(False)
        self.play_button.setShortcut("Space")
        self.play_button.clicked.connect(self._toggle_playback)
        self.time_label = QLabel("00:00.0 / 00:00.0")

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.hide()
        self.status_label = QLabel("打开视频后，在第一帧拖框选中自己。")
        self.status_label.setWordWrap(True)

        controls = QFrame()
        controls.setObjectName("controls")
        controls.setFixedWidth(310)
        controls_layout = QVBoxLayout(controls)
        title = QLabel("DANCE FOCUS")
        title.setObjectName("title")
        subtitle = QLabel("把舞者留在画面中心")
        subtitle.setObjectName("subtitle")
        model_label = QLabel(runtime_description())
        model_label.setObjectName("modelStatus")
        model_label.setWordWrap(True)
        controls_layout.addWidget(title)
        controls_layout.addWidget(subtitle)
        controls_layout.addWidget(model_label)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.open_button)

        form = QFormLayout()
        form.setVerticalSpacing(14)
        form.addRow("输出画幅", self.ratio_combo)
        form.addRow("预览模式", self.preview_combo)
        smoothing_row = QHBoxLayout()
        smoothing_row.addWidget(self.smoothing_slider)
        smoothing_row.addWidget(self.smoothing_value)
        form.addRow("跟随平滑", smoothing_row)
        controls_layout.addLayout(form)
        controls_layout.addSpacing(14)
        controls_layout.addWidget(self.analyze_button)
        controls_layout.addWidget(self.export_button)
        controls_layout.addWidget(self.open_export_button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.status_label)
        controls_layout.addWidget(self.progress)
        log_button = QPushButton("查看运行日志")
        log_button.clicked.connect(self._open_log)
        controls_layout.addWidget(log_button)

        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(self.canvas, 1)
        timeline = QHBoxLayout()
        timeline.addWidget(self.play_button)
        timeline.addWidget(self.frame_slider, 1)
        timeline.addWidget(self.time_label)
        preview_layout.addLayout(timeline)

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(18)
        root_layout.addLayout(preview_layout, 1)
        root_layout.addWidget(controls)
        self.setCentralWidget(root)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #15171c; color: #e8e9ed; }
            QFrame#controls { background: #20232a; border: 1px solid #30343d; border-radius: 12px; }
            QLabel#title { color: #65d6ad; font-size: 23px; font-weight: 800; letter-spacing: 2px; }
            QLabel#subtitle { color: #a7abb5; font-size: 13px; }
            QLabel#modelStatus { color: #65d6ad; font-size: 11px; padding-top: 6px; }
            QPushButton { background: #30353f; border: 1px solid #444b58; border-radius: 7px; padding: 10px; font-weight: 600; }
            QPushButton:hover { background: #3b424e; }
            QPushButton:pressed { background: #262a32; }
            QPushButton:disabled { color: #6f747e; background: #282b32; border-color: #32363f; }
            QComboBox { background: #292d35; border: 1px solid #424852; border-radius: 6px; padding: 7px; }
            QSlider::groove:horizontal { height: 4px; background: #383d47; border-radius: 2px; }
            QSlider::handle:horizontal { width: 14px; margin: -5px 0; background: #65d6ad; border-radius: 7px; }
            QProgressBar { border: 1px solid #3b404a; border-radius: 5px; text-align: center; background: #292d35; }
            QProgressBar::chunk { background: #65d6ad; border-radius: 4px; }
            """
        )

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择舞蹈视频",
            str(Path.home()),
            "视频文件 (*.mp4 *.mov *.mkv *.avi *.webm);;所有文件 (*)",
        )
        if not path:
            return
        self._pause_playback()
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
        self.export_button.setEnabled(False)
        self.status_label.setText("在第一帧用鼠标拖框，完整框住自己。")
        logging.info("Video opened: %s (%dx%d, %.3f fps, %d frames)", path, info.width, info.height, info.fps, info.frame_count)
        self._show_current_frame()

    def _selection_changed(self, box: Box) -> None:
        if self.video_info is None:
            return
        self._pause_playback()
        self.keyframes[self.current_frame_index] = box
        if self.current_frame_index == 0:
            message = "已选中目标，可以开始自动追踪。"
            self.analyze_button.setEnabled(not self.busy)
        else:
            message = f"已在第 {self.current_frame_index + 1} 帧添加修正点，请重新追踪。"
        self.tracked_boxes.clear()
        self.crop_path.clear()
        self.preview_combo.setCurrentIndex(0)
        self.export_button.setEnabled(False)
        self.status_label.setText(message)
        self._show_current_frame()

    def _analyze(self) -> None:
        if self.video_info is None or 0 not in self.keyframes or self.busy:
            return
        self._set_busy(True, "SAM 2 正在准备帧缓存并追踪；首次运行会下载 184 MB 官方模型……")
        worker = FunctionWorker(track_subject, self.video_info, dict(self.keyframes))
        worker.signals.progress.connect(self.progress.setValue)
        worker.signals.result.connect(self._analysis_complete)
        worker.signals.error.connect(self._operation_error)
        worker.signals.finished.connect(lambda: self._set_busy(False))
        self._start_worker(worker)

    def _analysis_complete(self, boxes: list[Box | None]) -> None:
        self.tracked_boxes = boxes
        self._rebuild_crop_path()
        self.preview_combo.setCurrentIndex(1)
        lost = sum(box is None for box in boxes)
        if lost:
            self.status_label.setText(
                f"追踪完成，有 {lost} 帧未可靠识别。可拖动时间轴检查，并在跟丢处重新框选。"
            )
        else:
            self.status_label.setText("追踪完成，已切换到裁剪结果预览。点击播放检查成片构图。")
        self.export_button.setEnabled(True)
        self._show_current_frame()

    def _rebuild_crop_path(self) -> None:
        if self.video_info is None or not self.tracked_boxes:
            return
        ratio = float(self.ratio_combo.currentData())
        crop_size = crop_size_for_aspect(
            self.video_info.width, self.video_info.height, ratio
        )
        try:
            self.crop_path = build_crop_path(
                self.tracked_boxes,
                (self.video_info.width, self.video_info.height),
                crop_size,
                self.video_info.fps,
                self.smoothing_slider.value() / 100,
            )
        except Exception as error:
            self._operation_error(str(error))
            return
        self._show_current_frame()

    def _smoothing_changed(self, value: int) -> None:
        self.smoothing_value.setText(f"{value / 100:.2f} 秒")
        self._rebuild_crop_path()

    def _preview_mode_changed(self) -> None:
        self._update_play_button_text()
        self._show_current_frame()

    def _frame_changed(self, frame_index: int) -> None:
        if self.video_info is None:
            return
        self.current_frame_index = frame_index
        frame = self._read_preview_frame(frame_index)
        if frame is None:
            logging.error("Preview frame decode failed: frame=%d", frame_index)
            self._pause_playback()
            self.status_label.setText(
                f"无法读取第 {frame_index + 1} 帧。请点击“查看运行日志”。"
            )
            return
        self.current_frame = frame
        self._show_current_frame()

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
        mode = "裁剪结果" if self.preview_combo.currentData() == "crop" else "原视频"
        self.status_label.setText(f"正在播放{mode}……")
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
        self.status_label.setText(
            f"音频播放失败，但画面仍可播放：{error_string}"
        )

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
        self.canvas.set_crop_preview(
            self.preview_combo.currentData() == "crop" and crop is not None
        )
        self.canvas.set_frame(self.current_frame, box, crop)
        current = self.current_frame_index / self.video_info.fps
        self.time_label.setText(
            f"{self._format_time(current)} / {self._format_time(self.video_info.duration)}"
        )

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        remainder = seconds - minutes * 60
        return f"{minutes:02d}:{remainder:04.1f}"

    def _export(self) -> None:
        if self.video_info is None or not self.crop_path or self.busy:
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

        self._set_busy(True, "正在编码视频并保留原音频……")
        worker = FunctionWorker(export_video, self.video_info, list(self.crop_path), path)
        worker.signals.progress.connect(self.progress.setValue)
        worker.signals.result.connect(self._export_complete)
        worker.signals.error.connect(self._operation_error)
        worker.signals.finished.connect(lambda: self._set_busy(False))
        self._start_worker(worker)

    def _export_complete(self, output_path: Path) -> None:
        self.last_export_path = Path(output_path)
        self.open_export_button.setEnabled(True)
        self.status_label.setText(f"导出完成：{self.last_export_path}")
        QMessageBox.information(
            self,
            "导出完成",
            f"视频已生成并验证可以播放：\n{self.last_export_path}\n\n"
            "点击右侧“打开导出视频”立即播放。",
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
        worker.signals.finished.connect(
            lambda current=worker: self.active_workers.discard(current)
        )
        self.thread_pool.start(worker)

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        if busy:
            self._pause_playback()
        self.busy = busy
        self.open_button.setEnabled(not busy)
        self.analyze_button.setEnabled(not busy and 0 in self.keyframes)
        self.export_button.setEnabled(not busy and bool(self.crop_path))
        self.frame_slider.setEnabled(not busy and self.video_info is not None)
        self.play_button.setEnabled(not busy and self.video_info is not None)
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

    def _operation_error(self, message: str) -> None:
        logging.error("Operation failed: %s", message)
        QMessageBox.critical(self, "操作失败", message)
        self.status_label.setText(message)

    def closeEvent(self, event) -> None:
        self._pause_playback()
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
