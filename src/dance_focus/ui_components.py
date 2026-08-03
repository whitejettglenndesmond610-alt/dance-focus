from __future__ import annotations

import os

from PySide6.QtCore import (
    QEasingCurve,
    QLineF,
    Property,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QEnterEvent, QFont, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QToolButton,
    QWidget,
)


MOTION_ENABLED = os.environ.get("DANCE_FOCUS_REDUCE_MOTION") != "1"


def _mix(start: QColor, end: QColor, amount: float) -> QColor:
    amount = min(max(amount, 0.0), 1.0)
    return QColor(
        round(start.red() + (end.red() - start.red()) * amount),
        round(start.green() + (end.green() - start.green()) * amount),
        round(start.blue() + (end.blue() - start.blue()) * amount),
        round(start.alpha() + (end.alpha() - start.alpha()) * amount),
    )


class MotionButton(QPushButton):
    """A lightweight animated button that does not depend on stylesheet transitions."""

    def __init__(self, text: str, role: str = "secondary", parent=None):
        super().__init__(text, parent)
        self.role = role
        self._hover_progress = 0.0
        self._animation = QPropertyAnimation(self, b"hoverProgress", self)
        self._animation.setDuration(170)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(38)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(max(hint.width() + 24, 96), max(hint.height(), 38))

    def get_hover_progress(self) -> float:
        return self._hover_progress

    def set_hover_progress(self, value: float) -> None:
        self._hover_progress = value
        self.update()

    hoverProgress = Property(float, get_hover_progress, set_hover_progress)

    def _animate(self, target: float) -> None:
        if not MOTION_ENABLED:
            self.set_hover_progress(target)
            return
        self._animation.stop()
        self._animation.setStartValue(self._hover_progress)
        self._animation.setEndValue(target)
        self._animation.start()

    def enterEvent(self, event: QEnterEvent) -> None:
        self._animate(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._animate(0.0)
        super().leaveEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        progress = self._hover_progress
        if self.isDown():
            progress = 0.45

        if self.role == "accent":
            background = _mix(QColor("#42bd91"), QColor("#55cba1"), progress)
            foreground = QColor("#ffffff")
            border = _mix(QColor("#42bd91"), QColor("#269e75"), progress)
        elif self.role == "ghost":
            background = _mix(QColor(255, 255, 255, 0), QColor("#edf7f3"), progress)
            foreground = _mix(QColor("#65766f"), QColor("#268763"), progress)
            border = _mix(QColor("#d7e4df"), QColor("#b9d8cc"), progress)
        else:
            background = _mix(QColor("#ffffff"), QColor("#eef8f4"), progress)
            foreground = _mix(QColor("#344840"), QColor("#21815f"), progress)
            border = _mix(QColor("#d4e2dd"), QColor("#a9d2c2"), progress)

        if not self.isEnabled():
            background = QColor("#f0f3f2")
            foreground = QColor("#a9b3af")
            border = QColor("#e1e7e4")

        painter.setPen(QPen(border, 1))
        painter.setBrush(background)
        painter.drawRoundedRect(rect, 9, 9)
        painter.setPen(foreground)
        font = self.font()
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())

        if self.hasFocus():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#42bd91"), 1))
            painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 8, 8)


class SegmentedControl(QFrame):
    currentIndexAboutToChange = Signal(int)
    currentIndexChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("segmentedControl")
        self.setFixedHeight(36)
        self._buttons: list[QToolButton] = []
        self._data: list[object] = []
        self._current_index = -1
        self._indicator = QFrame(self)
        self._indicator.setObjectName("segmentIndicator")
        self._indicator.lower()
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.idClicked.connect(self.setCurrentIndex)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(3, 3, 3, 3)
        self._layout.setSpacing(2)
        self._animation = QPropertyAnimation(self._indicator, b"geometry", self)
        self._animation.setDuration(210)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def addItem(self, text: str, data=None) -> None:
        index = len(self._buttons)
        button = QToolButton(self)
        button.setText(text)
        button.setCheckable(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setSizePolicy(
            button.sizePolicy().horizontalPolicy(),
            button.sizePolicy().verticalPolicy(),
        )
        self._group.addButton(button, index)
        self._layout.addWidget(button, 1)
        self._buttons.append(button)
        self._data.append(data)
        if self._current_index == -1:
            self._current_index = 0
            button.setChecked(True)
        QTimer.singleShot(0, lambda: self._move_indicator(False))

    def currentData(self):
        if self._current_index < 0:
            return None
        return self._data[self._current_index]

    def currentText(self) -> str:
        if self._current_index < 0:
            return ""
        return self._buttons[self._current_index].text()

    def currentIndex(self) -> int:
        return self._current_index

    def setItemEnabled(self, index: int, enabled: bool) -> None:
        if 0 <= index < len(self._buttons):
            self._buttons[index].setEnabled(enabled)

    def isItemEnabled(self, index: int) -> bool:
        return 0 <= index < len(self._buttons) and self._buttons[index].isEnabled()

    def setCurrentIndex(self, index: int) -> None:
        if not 0 <= index < len(self._buttons):
            return
        changed = index != self._current_index
        if changed:
            self.currentIndexAboutToChange.emit(index)
        self._current_index = index
        self._buttons[index].setChecked(True)
        self._move_indicator(changed)
        if changed:
            self.currentIndexChanged.emit(index)

    def _move_indicator(self, animate: bool) -> None:
        if self._current_index < 0 or not self._buttons:
            return
        target = self._buttons[self._current_index].geometry()
        if target.width() <= 0:
            return
        if animate and MOTION_ENABLED and self.isVisible():
            self._animation.stop()
            self._animation.setStartValue(self._indicator.geometry())
            self._animation.setEndValue(target)
            self._animation.start()
        else:
            self._indicator.setGeometry(target)
        self._indicator.lower()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, lambda: self._move_indicator(False))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, lambda: self._move_indicator(False))


class WorkflowIndicator(QWidget):
    def __init__(self, labels: list[str], parent=None):
        super().__init__(parent)
        self.labels = labels
        self._progress = 0.0
        self._step = 0
        self._animation = QPropertyAnimation(self, b"progress", self)
        self._animation.setDuration(360)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.setFixedHeight(54)

    def get_progress(self) -> float:
        return self._progress

    def set_progress(self, value: float) -> None:
        self._progress = value
        self.update()

    progress = Property(float, get_progress, set_progress)

    def setStep(self, step: int) -> None:
        step = min(max(step, 0), len(self.labels) - 1)
        self._step = step
        if MOTION_ENABLED and self.isVisible():
            self._animation.stop()
            self._animation.setStartValue(self._progress)
            self._animation.setEndValue(float(step))
            self._animation.start()
        else:
            self.set_progress(float(step))

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        count = len(self.labels)
        if count < 2:
            return
        margin = 18.0
        top = 14.0
        spacing = (self.width() - margin * 2) / (count - 1)
        points = [margin + spacing * index for index in range(count)]

        painter.setPen(QPen(QColor("#d7e4df"), 2))
        painter.drawLine(QLineF(points[0], top, points[-1], top))
        progress_x = points[0] + spacing * self._progress
        painter.setPen(QPen(QColor("#42bd91"), 2))
        painter.drawLine(QLineF(points[0], top, min(progress_x, points[-1]), top))

        for index, (x, label) in enumerate(zip(points, self.labels, strict=True)):
            completed = self._progress + 0.02 >= index
            active = index == self._step
            painter.setPen(QPen(QColor("#42bd91") if completed else QColor("#c7d6d0"), 2))
            painter.setBrush(
                QColor("#42bd91")
                if completed
                else QColor("#ffffff")
            )
            radius = 6.5 if active else 5.0
            painter.drawEllipse(QRectF(x - radius, top - radius, radius * 2, radius * 2))
            painter.setPen(QColor("#24362f") if active else QColor("#7a8b85"))
            font = self.font()
            font.setPixelSize(10)
            font.setWeight(
                QFont.Weight.DemiBold if active else QFont.Weight.Normal
            )
            painter.setFont(font)
            painter.drawText(
                QRectF(x - spacing / 2, 29, spacing, 20),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                label,
            )


class AnimatedStatusLabel(QLabel):
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(1.0)
        self.setGraphicsEffect(self._effect)
        self._animation = QPropertyAnimation(self._effect, b"opacity", self)
        self._animation.setDuration(220)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def setText(self, text: str) -> None:
        super().setText(text)
        if not hasattr(self, "_animation") or not MOTION_ENABLED:
            return
        self._animation.stop()
        self._animation.setStartValue(0.42)
        self._animation.setEndValue(1.0)
        self._animation.start()


class ConfidenceTimeline(QSlider):
    """Timeline with compact tracking-quality and keyframe overlays."""

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._quality: list[float] = []
        self._states: list[str] = []
        self._keyframes: set[int] = set()
        self._subject_corrections: set[int] = set()
        self.setMinimumHeight(28)

    def set_tracking_quality(self, samples) -> None:
        self._quality = [float(sample.tracking_confidence) for sample in samples]
        self._states = [sample.state.value for sample in samples]
        self.update()

    def clear_tracking_quality(self) -> None:
        self._quality.clear()
        self._states.clear()
        self.update()

    def set_keyframes(self, frame_indices) -> None:
        self._keyframes = set(frame_indices)
        self.update()

    def set_subject_corrections(self, frame_indices) -> None:
        self._subject_corrections = set(frame_indices)
        self.update()

    def mousePressEvent(self, event) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and (self._keyframes or self._subject_corrections)
            and self.maximum() > self.minimum()
        ):
            left = 8.0
            usable = max(1.0, self.width() - 16.0)
            denominator = self.maximum() - self.minimum()
            nearest = min(
                self._keyframes | self._subject_corrections,
                key=lambda frame: abs(
                    event.position().x()
                    - left
                    - usable * (frame - self.minimum()) / denominator
                ),
            )
            marker_x = (
                left
                + usable * (nearest - self.minimum()) / denominator
            )
            if abs(event.position().x() - marker_x) <= 9:
                self.setValue(nearest)
                event.accept()
                return
        super().mousePressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        left = 8.0
        usable = max(1.0, self.width() - 16.0)
        denominator = max(1, self.maximum() - self.minimum())
        marker_top = 2.0
        marker_bottom = 6.0
        for index, (quality, state) in enumerate(
            zip(self._quality, self._states, strict=True)
        ):
            if state == "lost" or quality < 0.20:
                color = QColor("#e85f69")
            elif state in {"occluded", "reidentified"} or quality < 0.48:
                color = QColor("#e9a347")
            else:
                continue
            x = left + usable * (index - self.minimum()) / denominator
            painter.setPen(QPen(color, 1.5))
            painter.drawLine(QLineF(x, marker_top, x, marker_bottom))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#458fdd"))
        for frame_index in self._keyframes:
            if not self.minimum() <= frame_index <= self.maximum():
                continue
            x = left + usable * (frame_index - self.minimum()) / denominator
            painter.drawEllipse(QRectF(x - 2.5, self.height() - 5.5, 5, 5))

        painter.setBrush(QColor("#32ac83"))
        for frame_index in self._subject_corrections:
            if not self.minimum() <= frame_index <= self.maximum():
                continue
            x = left + usable * (frame_index - self.minimum()) / denominator
            painter.drawRect(QRectF(x - 2.5, 1.0, 5, 5))


def animate_fade_in(widget: QWidget, delay: int = 0, duration: int = 320):
    if not MOTION_ENABLED:
        return None
    effect = QGraphicsOpacityEffect(widget)
    effect.setOpacity(0.0)
    widget.setGraphicsEffect(effect)
    animation = QPropertyAnimation(effect, b"opacity", widget)
    animation.setDuration(duration)
    animation.setStartValue(0.0)
    animation.setEndValue(1.0)
    animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    QTimer.singleShot(delay, animation.start)
    return animation


def animate_refresh(widget: QWidget):
    if not MOTION_ENABLED:
        return None
    effect = widget.graphicsEffect()
    if not isinstance(effect, QGraphicsOpacityEffect):
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
    animation = QPropertyAnimation(effect, b"opacity", widget)
    animation.setDuration(180)
    animation.setStartValue(0.5)
    animation.setEndValue(1.0)
    animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    animation.start()
    return animation
