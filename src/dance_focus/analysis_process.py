from __future__ import annotations

import logging
import multiprocessing
import os
from queue import Empty
import signal
import traceback
from typing import Any


def configure_native_threads() -> None:
    """Keep native runtimes from creating a thread per logical CPU."""
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "4")


def _analysis_entry(messages, operation: str, args) -> None:
    configure_native_threads()
    try:
        import torch

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)

        # Loading Triton's LLVM runtime from an already busy Qt worker thread can
        # segfault before Python can report an exception. Load it here, while this
        # fresh process still has a single application thread.
        import triton._C.libtriton  # noqa: F401

        from dance_focus.analysis import analyze_subject, reanalyze_subject_interval

        function = {
            "full": analyze_subject,
            "interval": reanalyze_subject_interval,
        }[operation]
        result = function(
            *args,
            progress=lambda value: messages.put(("progress", value)),
        )
        messages.put(("result", result))
    except BaseException as error:
        messages.put(
            (
                "error",
                f"{type(error).__name__}: {error}",
                traceback.format_exc(),
            )
        )


def _native_crash_message(exit_code: int) -> str:
    if exit_code < 0:
        try:
            reason = signal.Signals(-exit_code).name
        except ValueError:
            reason = f"signal {-exit_code}"
    else:
        reason = f"exit code {exit_code}"
    return (
        f"AI 分析进程异常退出（{reason}）。主界面和项目数据仍然安全，"
        "请重试；如果持续发生，请打开日志查看详情。"
    )


def _run_analysis_isolated(
    operation: str,
    args,
    progress=None,
    cancelled=None,
):
    """Run native AI libraries out-of-process so a crash cannot kill the UI."""
    context = multiprocessing.get_context("spawn")
    messages = context.Queue()
    process = context.Process(
        target=_analysis_entry,
        args=(messages, operation, args),
        name="dance-focus-analysis",
    )
    process.start()
    result: Any = None
    error: str | None = None
    error_traceback: str | None = None

    try:
        while process.is_alive():
            if cancelled and cancelled():
                process.terminate()
                process.join(timeout=5)
                raise InterruptedError("操作已取消")
            try:
                message = messages.get(timeout=0.2)
            except Empty:
                continue
            kind, *payload = message
            if kind == "progress" and progress:
                progress(payload[0])
            elif kind == "result":
                result = payload[0]
            elif kind == "error":
                error, error_traceback = payload

        process.join()
        while True:
            try:
                message = messages.get_nowait()
            except Empty:
                break
            kind, *payload = message
            if kind == "progress" and progress:
                progress(payload[0])
            elif kind == "result":
                result = payload[0]
            elif kind == "error":
                error, error_traceback = payload
    finally:
        messages.close()
        messages.join_thread()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    if error is not None:
        logging.error("Isolated AI analysis failed: %s\n%s", error, error_traceback)
        raise RuntimeError(error)
    if process.exitcode:
        message = _native_crash_message(process.exitcode)
        logging.error(message)
        raise RuntimeError(message)
    if result is None:
        raise RuntimeError("AI 分析进程没有返回结果，请重试")
    return result


def analyze_subject_isolated(
    info,
    prompts,
    progress=None,
    cancelled=None,
):
    return _run_analysis_isolated(
        "full", (info, prompts), progress=progress, cancelled=cancelled
    )


def reanalyze_subject_interval_isolated(
    info,
    prompts,
    previous,
    start_frame,
    end_frame,
    progress=None,
    cancelled=None,
):
    return _run_analysis_isolated(
        "interval",
        (info, prompts, previous, start_frame, end_frame),
        progress=progress,
        cancelled=cancelled,
    )
