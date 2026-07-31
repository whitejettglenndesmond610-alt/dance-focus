from dance_focus.geometry import FramingSettings
from dance_focus.history import EditHistory, EditState


def _state(smoothing: float) -> EditState:
    return EditState((), None, FramingSettings(9 / 16, smoothing), (), None)


def test_history_undo_redo_and_new_edit_clears_redo():
    history = EditHistory()
    first = _state(0.45)
    second = _state(0.75)
    third = _state(0.25)
    history.record("稳定", first, second)

    assert history.undo().before == first
    assert history.redo().after == second
    history.undo()
    history.record("灵敏", first, third)

    assert history.redo() is None
    assert history.undo_label == "灵敏"
