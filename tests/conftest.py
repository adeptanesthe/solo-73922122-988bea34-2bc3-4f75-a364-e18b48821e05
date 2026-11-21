"""合成空转声 WAV 与 TestClient 夹具。"""
from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.service import SpeedCheckService

SR = 16_000


def build_harmonic_wav(
    rpms,
    duration: float = 4.0,
    orders=(1, 2, 3),
    sr: int = SR,
    noise: float = 0.004,
    seed: int = 7,
) -> bytes:
    """生成空转声 WAV。

    rpms: 单个转速（整段恒定）或长度 3 的序列（三段各自转速，按采样点三等分）。
    orders: 叠加的阶次（基频 = rpm/60 Hz）。
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration * sr))
    t = np.arange(n, dtype=np.float64) / sr
    x = np.zeros(n, dtype=np.float64)

    if np.isscalar(rpms):
        rpm_list = [rpms, rpms, rpms]
    else:
        rpm_list = list(rpms)
    bounds = np.linspace(0, n, 4, dtype=int)
    for seg, rpm in enumerate(rpm_list):
        s, e = bounds[seg], bounds[seg + 1]
        f0 = rpm / 60.0
        for k in orders:
            x[s:e] += (0.6 / k) * np.sin(
                2 * np.pi * f0 * k * t[s:e] + rng.uniform(0, 2 * np.pi)
            )

    x += rng.standard_normal(n) * noise
    x = x / np.max(np.abs(x)) * 0.9

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((x * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def build_silence_wav(duration: float = 4.0, sr: int = SR) -> bytes:
    n = int(round(duration * sr))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.zeros(n, dtype="<i2").tobytes())
    return buf.getvalue()


def rewrap(pcm: bytes, *, nch: int, sw: int, fr: int, nframes: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(sw)
        wf.setframerate(fr)
        wf.writeframes(pcm)
    return buf.getvalue()


@pytest.fixture
def client():
    app.service = SpeedCheckService()
    return TestClient(app)


def post_check(client: TestClient, wav: bytes, rpm_min=300, rpm_max=6000,
               filename: str = "idle.wav"):
    return client.post(
        "/api/check",
        files={"recording": (filename, wav, "audio/wav")},
        data={"rpm_min": str(rpm_min), "rpm_max": str(rpm_max)},
    )
