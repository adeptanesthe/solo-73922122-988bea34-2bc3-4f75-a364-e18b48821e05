"""WAV 解析与三段等分。

输入约定（沿用现场现有格式）：16 kHz、16-bit PCM、单声道 WAV，
时长 3～8 秒。解析后将录音按采样点等分为三段（长度差至多 1 个采样点），
供谱分析模块逐段提取局部谱峰。
"""
from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # 16-bit PCM
CHANNELS = 1
MIN_DURATION_SEC = 3.0
MAX_DURATION_SEC = 8.0
SEGMENT_COUNT = 3
# 上传体积上限：8 秒 16k/16bit/单声道 PCM + WAV 头余量，再留少量冗余
MAX_UPLOAD_BYTES = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS * int(MAX_DURATION_SEC) + 8192


class AudioValidationError(ValueError):
    """音频文件不符合约定格式或长度。"""


@dataclass
class AudioData:
    samples: np.ndarray          # float32、范围 [-1, 1] 的单声道波形
    duration: float
    segments: list[np.ndarray]   # 等分三段后的波形


def parse_wav(data: bytes) -> AudioData:
    """校验并解析 WAV，返回整段波形与等分三段。"""
    if not data:
        raise AudioValidationError("文件为空，请上传 3～8 秒的 WAV 录音")
    if len(data) > MAX_UPLOAD_BYTES:
        raise AudioValidationError(
            f"文件过大；仅支持 {SAMPLE_RATE}Hz/16bit/单声道、"
            f"{MIN_DURATION_SEC:.0f}～{MAX_DURATION_SEC:.0f} 秒的 WAV"
        )

    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioValidationError(f"无法解析 WAV 文件：{exc}") from exc

    if n_channels != CHANNELS:
        raise AudioValidationError(f"只支持单声道 WAV，当前声道数：{n_channels}")
    if sample_width != SAMPLE_WIDTH:
        raise AudioValidationError(
            f"只支持 16-bit PCM，当前采样位深：{sample_width * 8}-bit"
        )
    if frame_rate != SAMPLE_RATE:
        raise AudioValidationError(
            f"只支持 {SAMPLE_RATE} Hz 采样率，当前：{frame_rate} Hz"
        )
    if n_frames == 0:
        raise AudioValidationError("录音为空，没有任何采样点")

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if samples.shape[0] != n_frames:
        raise AudioValidationError("WAV 数据块长度与声明帧数不一致，文件可能损坏")

    duration = n_frames / frame_rate
    if duration < MIN_DURATION_SEC - 1e-9:
        raise AudioValidationError(
            f"录音时长 {duration:.2f}s 不足下限 {MIN_DURATION_SEC:.0f}s，"
            "请上传 3～8 秒的空转录音"
        )
    if duration > MAX_DURATION_SEC + 1e-9:
        raise AudioValidationError(
            f"录音时长 {duration:.2f}s 超过上限 {MAX_DURATION_SEC:.0f}s"
        )

    # 按采样点等分三段；array_split 保证各段长度差至多 1
    segments = [seg.astype(np.float32, copy=True) for seg in np.array_split(samples, SEGMENT_COUNT)]
    return AudioData(samples=samples, duration=duration, segments=segments)
