from __future__ import annotations

from dataclasses import dataclass

from dance_focus.geometry import (
    Box,
    CameraKeyframe,
    CameraPath,
    FramingSettings,
    TrackingResult,
)


@dataclass(frozen=True)
class EditState:
    subject_prompts: tuple[tuple[int, Box], ...]
    tracking: TrackingResult | None
    framing: FramingSettings
    camera_keyframes: tuple[CameraKeyframe, ...]
    camera_path: CameraPath | None


@dataclass(frozen=True)
class EditCommand:
    label: str
    before: EditState
    after: EditState


class EditHistory:
    def __init__(self) -> None:
        self._undo: list[EditCommand] = []
        self._redo: list[EditCommand] = []

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def record(self, label: str, before: EditState, after: EditState) -> None:
        if before == after:
            return
        self._undo.append(EditCommand(label, before, after))
        self._redo.clear()

    def undo(self) -> EditCommand | None:
        if not self._undo:
            return None
        command = self._undo.pop()
        self._redo.append(command)
        return command

    def redo(self) -> EditCommand | None:
        if not self._redo:
            return None
        command = self._redo.pop()
        self._undo.append(command)
        return command

    @property
    def undo_label(self) -> str | None:
        return self._undo[-1].label if self._undo else None

    @property
    def redo_label(self) -> str | None:
        return self._redo[-1].label if self._redo else None
