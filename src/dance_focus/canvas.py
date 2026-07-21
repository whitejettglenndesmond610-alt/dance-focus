from __future__ import annotations

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from dance_focus.geometry import Box, CropRect


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
        self._crop_preview = False
        self._drag_start: QPointF | None = None
        self._drag_end: QPointF | None = None

    def set_crop_preview(self, enabled: bool) -> None:
        self._crop_preview = enabled
        self.update()

    def set_frame(
        self, frame, target_box: Box | None = None, crop_rect: CropRect | None = None
    ) -> None:
        self._frame = frame
        self._target_box = target_box
        self._crop_rect = crop_rect
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
        painter.fillRect(self.rect(), QColor("#0c0d10"))
        if self._image is None:
            painter.setPen(QColor("#737782"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "打开一个舞蹈视频")
            return

        image_rect = self._image_rect()
        painter.drawImage(image_rect, self._image, self._source_rect())

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
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#f4c95d"), 2))
            painter.drawRect(crop)
            painter.restore()

        if self._target_box is not None and not self._crop_preview:
            painter.save()
            painter.setClipRect(image_rect)
            target = self._to_widget_rect(
                self._target_box.x,
                self._target_box.y,
                self._target_box.width,
                self._target_box.height,
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#65d6ad"), 2))
            painter.drawRect(target)
            painter.restore()

        if self._drag_start is not None and self._drag_end is not None:
            painter.save()
            painter.setClipRect(image_rect)
            start = self._to_widget_rect(self._drag_start.x(), self._drag_start.y(), 0, 0)
            end = self._to_widget_rect(self._drag_end.x(), self._drag_end.y(), 0, 0)
            selection = QRectF(start.topLeft(), end.topLeft()).normalized()
            painter.setBrush(QColor(101, 214, 173, 45))
            painter.setPen(QPen(QColor("#65d6ad"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(selection)
            painter.restore()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self._to_frame_point(event.position())
        if point is not None:
            self._drag_start = point
            self._drag_end = point
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is None:
            return
        point = self._to_frame_point(event.position())
        if point is not None:
            self._drag_end = point
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is None or self._drag_end is None:
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
