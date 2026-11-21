"""WAV 解析、时长/格式校验与三段等分测试。"""
import io
import wave

import numpy as np
import pytest

from app.audio import (
    MAX_DURATION_SEC,
    MIN_DURATION_SEC,
    SAMPLE_RATE,
    AudioValidationError,
    parse_wav,
)
from conftest import SR, build_harmonic_wav, build_silence_wav, rewrap


def test_parse_returns_three_equal_segments():
    audio = parse_wav(build_harmonic_wav(1500, duration=4.0))
    assert audio.duration == pytest.approx(4.0, abs=1e-6)
    assert len(audio.segments) == 3
    # 等分：总长 64000 采样，三段长度差至多 1
    lens = [len(s) for s in audio.segments]
    assert sum(lens) == len(audio.samples)
    assert max(lens) - min(lens) <= 1
    # 分段内容与整段对应位置一致
    assert audio.segments[0][0] == audio.samples[0]


def test_segments_for_non_divisible_length():
    # 5 秒 = 80000 采样，不能被 3 整除
    audio = parse_wav(build_harmonic_wav(1500, duration=5.0))
    lens = sorted(len(s) for s in audio.segments)
    assert lens == [26666, 26667, 26667]


def test_reject_too_short():
    with pytest.raises(AudioValidationError, match="不足下限"):
        parse_wav(build_silence_wav(duration=2.0))


def test_reject_too_long():
    # 9 秒录音同时触发体积/时长上限，两种提示均属合规拒绝
    pcm = np.zeros(int((MAX_DURATION_SEC + 1) * SR), dtype="<i2").tobytes()
    with pytest.raises(AudioValidationError, match="过大|超过上限"):
        parse_wav(
            rewrap(pcm, nch=1, sw=2, fr=SR, nframes=int((MAX_DURATION_SEC + 1) * SR))
        )


def test_boundary_durations_accepted():
    parse_wav(build_silence_wav(duration=MIN_DURATION_SEC))
    parse_wav(build_silence_wav(duration=MAX_DURATION_SEC))


def test_reject_wrong_format():
    base = build_harmonic_wav(1500, duration=3.5)
    with wave.open(io.BytesIO(base), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())

    with pytest.raises(AudioValidationError, match="声道"):
        parse_wav(rewrap(pcm, nch=2, sw=2, fr=SR, nframes=len(pcm) // 4))
    with pytest.raises(AudioValidationError, match="位深"):
        parse_wav(rewrap(pcm, nch=1, sw=1, fr=SR, nframes=len(pcm) // 2))
    with pytest.raises(AudioValidationError, match="采样率"):
        parse_wav(rewrap(pcm, nch=1, sw=2, fr=8000, nframes=len(pcm) // 2))
    with pytest.raises(AudioValidationError):
        parse_wav(b"not a wav file")
    with pytest.raises(AudioValidationError, match="为空"):
        parse_wav(b"")


def test_silent_wav_parses_fine():
    # 静音是合法录音，谱分析阶段才会报无有效功率
    audio = parse_wav(build_silence_wav(duration=4.0))
    assert len(audio.segments) == 3
    assert np.all(audio.samples == 0)


def test_sample_rate_constant():
    assert SAMPLE_RATE == 16_000
