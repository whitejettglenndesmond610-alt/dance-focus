from __future__ import annotations

import signal

import pytest

from dance_focus.analysis_process import (
    _native_crash_message,
    analyze_subject_isolated,
)


def test_native_crash_message_keeps_ui_recoverable():
    message = _native_crash_message(-signal.SIGSEGV)

    assert "SIGSEGV" in message
    assert "主界面和项目数据仍然安全" in message


def test_isolated_runtime_reports_python_errors_without_crashing_parent():
    with pytest.raises(RuntimeError, match="请先在第一帧完整框选自己"):
        analyze_subject_isolated(None, {})
