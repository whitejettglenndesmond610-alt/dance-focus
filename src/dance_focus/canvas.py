from __future__ import annotations

import cv2
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QWidget

from dance_focus.geometry import Box, CropRect, PoseAnchor


class VideoCanvas(QWidget):
    selection_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 420)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._frame = None
        self._image: QImage | None = None
        self._target_box: Box | None = None
        self._crop_rect: CropRect | None = None
        self._pose_anchor: PoseAnchor | None = None
        self._crop_preview = False
        self._drag_start: QPointF | None = None
        self._drag_end: QPointF | None = None

    def set_crop_preview(self, enabled: bool) -> None:
        self._crop_preview = enabled
        self.setCursor(
            Qt.CursorShape.ArrowCursor
            if enabled
            else Qt.CursorShape.CrossCursor
        )
        self.update()

    def set_frame(
        self,
        frame,
        target_box: Box | None = None,
        crop_rect: CropRect | None = None,
        pose_anchor: PoseAnchor | None = None,
    ) -> None:
        self._frame = frame
        self._target_box = target_box
        self._crop_rect = crop_rect
        self._pose_anchor = pose_anchor
        if frame is None:
            self._image = None
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            self._image = QImage(
                rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
            ).copy()
        self.update()

    def _image_rect(self) -> QRectF:
        if self._image is None:
            return QRectF()
        source = self._source_rect()
        scale = min(self.width() / source.width(), self.height() / source.height())
        width = source.width() * scale
        height = source.height() * scale
        return QRectF(
            (self.width() - width) / 2,
            (self.height() - height) / 2,
            width,
            height,
        )

    def _source_rect(self) -> QRectF:
        if self._image is None:
            return QRectF()
        if self._crop_preview and self._crop_rect is not None:
            return QRectF(
                self._crop_rect.x,
                self._crop_rect.y,
                self._crop_rect.width,
                self._crop_rect.height,
            )
        return QRectF(0, 0, self._image.width(), self._image.height())

    def _to_widget_rect(self, x: float, y: float, width: float, height: float) -> QRectF:
        image_rect = self._image_rect()
        if self._image is None:
            return QRectF()
        source = self._source_rect()
        scale_x = image_rect.width() / source.width()
        scale_y = image_rect.height() / source.height()
        return QRectF(
            image_rect.left() + (x - source.left()) * scale_x,
            image_rect.top() + (y - source.top()) * scale_y,
            width * scale_x,
            height * scale_y,
        )

    def _to_frame_point(self, position: QPointF) -> QPointF | None:
        image_rect = self._image_rect()
        if self._image is None or not image_rect.contains(position):
            return None
        source = self._source_rect()
        x = (
            source.left()
            + (position.x() - image_rect.left())
            * source.width()
            / image_rect.width()
        )
        y = (
            source.top()
            + (position.y() - image_rect.top())
            * source.height()
            / image_rect.height()
        )
        return QPointF(x, y)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        background = QLinearGradient(0, 0, 0, self.height())
        background.setColorAt(0, QColor("#18211f"))
        background.setColorAt(1, QColor("#111816"))
        painter.fillRect(self.rect(), background)
        if self._image is None:
            self._paint_empty_state(painter)
            return

        image_rect = self._image_rect()
        frame_path = QPainterPath()
        frame_path.addRoundedRect(image_rect, 8, 8)
        painter.save()
        painter.setClipPath(frame_path)
        painter.drawImage(image_rect, self._image, self._source_rect())
        painter.restore()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#34413d"), 1))
        painter.drawRoundedRect(image_rect, 8, 8)

        if self._crop_rect is not None and not self._crop_preview:
            crop = self._to_widget_rect(
                self._crop_rect.x,
                self._crop_rect.y,
                self._crop_rect.width,
                self._crop_rect.height,
            )
            painter.save()
            painter.setBrush(QColor(0, 0, 0, 135))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(
                QRectF(
                    image_rect.left(),
                    image_rect.top(),
                    image_rect.width(),
                    crop.top() - image_rect.top(),
                )
            )
            painter.drawRect(
                QRectF(
                    image_rect.left(),
                    crop.bottom(),
                    image_rect.width(),
                    image_rect.bottom() - crop.bottom(),
                )
            )
            painter.drawRect(
                QRectF(
                    image_rect.left(),
                    crop.top(),
                    crop.left() - image_rect.left(),
                    crop.height(),
                )
            )
            painter.drawRect(
                QRectF(
                    crop.right(),
                    crop.top(),
                    image_rect.right() - crop.right(),
                    crop.height(),
                )
            )
            painter.setClipRect(crop)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(87, 210, 165, 92), 1))
            painter.drawLine(
                QLineF(
                    crop.left() + crop.width() / 3,
                    crop.top(),
                    crop.left() + crop.width() / 3,
                    crop.bottom(),
                )
            )
            painter.drawLine(
                QLineF(
                    crop.left() + crop.width() * 2 / 3,
                    crop.top(),
                    crop.left() + crop.width() * 2 / 3,
                    crop.bottom(),
                )
            )
            painter.drawLine(
                QLineF(
                    crop.left(),
                    crop.top() + crop.height() / 3,
                    crop.right(),
                    crop.top() + crop.height() / 3,
                )
            )
            painter.drawLine(
                QLineF(
                    crop.left(),
                    crop.top() + crop.height() * 2 / 3,
                    crop.right(),
                    crop.top() + crop.height() * 2 / 3,
                )
            )
            painter.restore()
            self._draw_corner_frame(painter, crop, QColor("#57d2a5"), 2.2, 20)
            self._draw_crop_label(painter, crop)

        if self._target_box is not None and not self._crop_preview:
            painter.save()
            painter.setClipRect(image_rect)
            target = self._to_widget_rect(
                self._target_box.x,
                self._target_box.y,
                self._target_box.width,
                self._target_box.height,
            )
            painter.setBrush(QColor(103, 209, 255, 16))
            painter.setPen(QPen(QColor(103, 209, 255, 74), 1))
            painter.drawRoundedRect(target, 3, 3)
            painter.restore()
            self._draw_corner_frame(painter, target, QColor("#67d1ff"), 2.0, 14)

        if self._pose_anchor is not None and not self._crop_preview:
            anchor = self._to_widget_rect(
                self._pose_anchor.point.x,
                self._pose_anchor.point.y,
                0,
                0,
            ).topLeft()
            painter.save()
            painter.setPen(QPen(QColor("#57d2a5"), 1.5))
            painter.setBrush(QColor(9, 12, 14, 210))
            painter.drawEllipse(anchor, 7, 7)
            painter.drawLine(QLineF(anchor.x() - 12, anchor.y(), anchor.x() - 5, anchor.y()))
            painter.drawLine(QLineF(anchor.x() + 5, anchor.y(), anchor.x() + 12, anchor.y()))
            painter.drawLine(QLineF(anchor.x(), anchor.y() - 12, anchor.x(), anchor.y() - 5))
            painter.drawLine(QLineF(anchor.x(), anchor.y() + 5, anchor.x(), anchor.y() + 12))
            painter.restore()

        if self._drag_start is not None and self._drag_end is not None:
            painter.save()
            painter.setClipRect(image_rect)
            start = self._to_widget_rect(self._drag_start.x(), self._drag_start.y(), 0, 0)
            end = self._to_widget_rect(self._drag_end.x(), self._drag_end.y(), 0, 0)
            selection = QRectF(start.topLeft(), end.topLeft()).normalized()
            painter.setBrush(QColor(87, 210, 165, 28))
            painter.setPen(QPen(QColor(87, 210, 165, 140), 1, Qt.PenStyle.DashLine))
            painter.drawRoundedRect(selection, 3, 3)
            painter.restore()
            self._draw_corner_frame(painter, selection, QColor("#57d2a5"), 2.0, 14)

    def _paint_empty_state(self, painter: QPainter) -> None:
        safe = QRectF(self.rect()).adjusted(64, 64, -64, -64)
        painter.setPen(QPen(QColor(61, 78, 73, 80), 1, Qt.PenStyle.DotLine))
        for fraction in (1 / 3, 2 / 3):
            painter.drawLine(
                QLineF(
                    safe.left() + safe.width() * fraction,
                    safe.top(),
                    safe.left() + safe.width() * fraction,
                    safe.bottom(),
                )
            )
            painter.drawLine(
                QLineF(
                    safe.left(),
                    safe.top() + safe.height() * fraction,
                    safe.right(),
                    safe.top() + safe.height() * fraction,
                )
            )

        center = safe.center()
        painter.setBrush(QColor("#1d2a26"))
        painter.setPen(QPen(QColor("#40514b"), 1))
        painter.drawEllipse(center, 31, 31)
        painter.setPen(QPen(QColor("#57d2a5"), 2))
        painter.drawLine(QLineF(center.x() - 10, center.y(), center.x() + 10, center.y()))
        painter.drawLine(QLineF(center.x(), center.y() - 10, center.x(), center.y() + 10))

        title_rect = QRectF(center.x() - 180, center.y() + 46, 360, 26)
        font = painter.font()
        font.setPixelSize(14)
        font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor("#e7efec"))
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, "导入舞蹈视频")
        copy_rect = QRectF(center.x() - 220, center.y() + 72, 440, 24)
        font.setPixelSize(11)
        font.setWeight(QFont.Weight.Normal)
        painter.setFont(font)
        painter.setPen(QColor("#85958f"))
        painter.drawText(
            copy_rect,
            Qt.AlignmentFlag.AlignCenter,
            "导入视频，从选择舞者开始",
        )

    @staticmethod
    def _draw_corner_frame(
        painter: QPainter,
        rect: QRectF,
        color: QColor,
        width: float,
        length: float,
    ) -> None:
        if rect.width() <= 1 or rect.height() <= 1:
            return
        length = min(length, rect.width() / 3, rect.height() / 3)
        painter.save()
        painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap))
        for x, x_direction in ((rect.left(), 1), (rect.right(), -1)):
            for y, y_direction in ((rect.top(), 1), (rect.bottom(), -1)):
                painter.drawLine(QLineF(x, y, x + length * x_direction, y))
                painter.drawLine(QLineF(x, y, x, y + length * y_direction))
        painter.restore()

    @staticmethod
    def _draw_crop_label(painter: QPainter, crop: QRectF) -> None:
        label = "成片范围"
        font = painter.font()
        font.setPixelSize(9)
        font.setWeight(QFont.Weight.ExtraBold)
        painter.setFont(font)
        text_width = painter.fontMetrics().horizontalAdvance(label)
        label_rect = QRectF(
            crop.left() + 8,
            crop.top() + 8,
            text_width + 18,
            23,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(20, 35, 30, 220))
        painter.drawRoundedRect(label_rect, 6, 6)
        painter.setPen(QColor("#6ce0b6"))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._crop_preview or event.button() != Qt.MouseButton.LeftButton:
            return
        point = self._to_frame_point(event.position())
        if point is not None:
            self._drag_start = point
            self._drag_end = point
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._crop_preview or self._drag_start is None:
            return
        point = self._to_frame_point(event.position())
        if point is not None:
            self._drag_end = point
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._crop_preview or self._drag_start is None or self._drag_end is None:
            return
        start, end = self._drag_start, self._drag_end
        self._drag_start = None
        self._drag_end = None
        x = min(start.x(), end.x())
        y = min(start.y(), end.y())
        width = abs(start.x() - end.x())
        height = abs(start.y() - end.y())
        if width >= 12 and height >= 12:
            box = Box(x, y, width, height)
            self._target_box = box
            self.selection_changed.emit(box)
        self.update()
