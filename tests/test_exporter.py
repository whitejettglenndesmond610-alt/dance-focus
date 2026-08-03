from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest

import dance_focus.exporter as exporter
from dance_focus.exporter import (
    ExportFrameRate,
    ExportQuality,
    ExportResolution,
    ExportSettings,
    build_ffmpeg_command,
    choose_video_encoder,
    encoder_args,
    export_video,
    resize_interpolation,
    resolve_output_size,
    video_filter,
)
from dance_focus.geometry import CameraPath, CropRect
from dance_focus.video import VideoInfo


def test_export_settings_defaults():
    assert ExportSettings() == ExportSettings(
        ExportQuality.HIGH,
        ExportResolution.NATIVE,
        ExportFrameRate.SOURCE,
        False,
    )


@pytest.mark.parametrize(
    ("native", "resolution", "expected"),
    [
        ((1919, 1079), ExportResolution.NATIVE, (1918, 1078)),
        ((1600, 900), ExportResolution.P720, (1280, 720)),
        ((900, 1600), ExportResolution.P1080, (1080, 1920)),
        ((1000, 1000), ExportResolution.P720, (720, 720)),
        ((800, 1000), ExportResolution.P1080, (1080, 1350)),
    ],
)
def test_resolve_output_size_uses_nearest_common_aspect(native, resolution, expected):
    assert resolve_output_size(native, resolution) == expected
    assert expected[0] % 2 == expected[1] % 2 == 0


def test_encoder_selection_prefers_nvenc_then_software():
    encoder, args = choose_video_encoder(
        ExportQuality.HIGH,
        1920,
        1080,
        60,
        " V..... h264_nvenc\n V..... libx264\n",
    )
    assert encoder == "h264_nvenc"
    assert args == [
        "-preset", "p7", "-tune", "hq", "-rc", "vbr",
        "-cq", "16", "-b:v", "0",
    ]

    encoder, args = choose_video_encoder(
        ExportQuality.SMALL, 1280, 720, 30, " V..... libx264\n V..... mpeg4\n"
    )
    assert encoder == "libx264"
    assert args[-2:] == ["-crf", "23"]


def test_encoder_quality_parameters_and_openh264_bitrate_limits():
    assert encoder_args("h264_nvenc", ExportQuality.STANDARD, 1, 1, 1)[1] == "p5"
    assert encoder_args("libx264", ExportQuality.HIGH, 1, 1, 1)[-1] == "16"
    low = encoder_args("libopenh264", ExportQuality.SMALL, 16, 16, 1)
    high = encoder_args("libopenh264", ExportQuality.HIGH, 7680, 4320, 120)
    assert low[1] == "750000"
    assert high[1] == "50000000"
    assert encoder_args("mpeg4", ExportQuality.SMALL, 1, 1, 1) == ["-q:v", "7"]


def test_frame_rate_filters():
    assert video_filter(24, ExportSettings()) is None
    assert video_filter(
        60, ExportSettings(frame_rate=ExportFrameRate.FPS_30)
    ) == "fps=30"
    assert video_filter(
        24,
        ExportSettings(frame_rate=ExportFrameRate.FPS_60, interpolate=True),
    ) == "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    timed_filter = video_filter(
        24,
        ExportSettings(frame_rate=ExportFrameRate.FPS_60, interpolate=True),
        2.0,
    )
    assert timed_filter.startswith("tpad=stop_mode=clone")
    assert "trim=duration=2.000000000" in timed_filter
    assert video_filter(
        60,
        ExportSettings(frame_rate=ExportFrameRate.FPS_60, interpolate=True),
    ) == "fps=60"


def test_resize_uses_area_for_reduction_and_lanczos_for_enlargement():
    assert resize_interpolation((1920, 1080), (1280, 720)) == cv2.INTER_AREA
    assert resize_interpolation((640, 360), (1280, 720)) == cv2.INTER_LANCZOS4


def test_ffmpeg_command_preserves_raw_fps_and_configures_output(tmp_path: Path):
    info = VideoInfo(tmp_path / "source.mp4", 1920, 1080, 24.0, 48)
    settings = ExportSettings(
        frame_rate=ExportFrameRate.FPS_60, interpolate=True
    )
    command = build_ffmpeg_command(
        info,
        tmp_path / "out.mp4",
        (1280, 720),
        settings,
        "libx264",
        ["-crf", "16"],
    )

    assert command[command.index("-framerate") + 1] == "24.000000"
    assert "minterpolate=fps=60" in command[command.index("-vf") + 1]
    assert command[command.index("-b:a") + 1] == "192k"
    assert command[command.index("-t") + 1] == "2.000000000"
    assert command[-1] == str(tmp_path / "out.mp4")


def test_export_reads_source_directly_and_uses_lanczos_when_enlarging(
    monkeypatch, tmp_path: Path
):
    source = tmp_path / "source.mp4"
    output = tmp_path / "out.mp4"
    info = VideoInfo(source, 320, 180, 24.0, 1)
    path = CameraPath((160, 90), (CropRect(0, 0, 160, 90),))
    captures = []
    resize_interpolations = []

    class FakeCapture:
        def __init__(self, capture_path):
            self.path = capture_path
            self.read_count = 0
            captures.append(self)

        def isOpened(self):
            return True

        def read(self):
            self.read_count += 1
            return True, np.zeros((180, 320, 3), dtype=np.uint8)

        def release(self):
            pass

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.command = command
            self.stdin = BytesIO()
            self.stderr = BytesIO()

        def wait(self):
            output.write_bytes(b"video")
            return 0

        def poll(self):
            return 0

    def fake_resize(frame, size, interpolation):
        resize_interpolations.append(interpolation)
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    monkeypatch.setattr(exporter.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(exporter.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(exporter.cv2, "resize", fake_resize)
    monkeypatch.setattr(exporter.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        exporter,
        "choose_video_encoder",
        lambda *args: ("libx264", ["-crf", "16"]),
    )

    result = export_video(
        info,
        path,
        output,
        settings=ExportSettings(resolution=ExportResolution.P720),
    )

    assert result == output
    assert captures[0].path == str(source)
    assert captures[0].read_count == 1
    assert resize_interpolations == [cv2.INTER_LANCZOS4]
