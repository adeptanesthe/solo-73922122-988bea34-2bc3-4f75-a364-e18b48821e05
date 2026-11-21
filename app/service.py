"""核对服务：输入校验与谱分析编排。

任何输入错误（WAV 格式/时长、转速范围）都抛出明确原因的异常，
由 API 层转成 400，且不会产生有效结果。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .audio import SAMPLE_RATE, parse_wav
from .matching import analyze

RPM_LIMIT_MIN = 300.0
RPM_LIMIT_MAX = 6000.0


class InputValidationError(ValueError):
    """表单输入（转速范围等）不合法。"""


def parse_rpm(value: Any, label: str) -> float:
    """解析铭牌转速边界，要求是 300～6000 内的有限数值。"""
    if value is None or str(value).strip() == "":
        raise InputValidationError(f"请填写{label}")
    try:
        rpm = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{label}必须是数值（转/分），当前输入：{value}") from exc
    if not (RPM_LIMIT_MIN <= rpm <= RPM_LIMIT_MAX):
        raise InputValidationError(
            f"{label}必须在 {RPM_LIMIT_MIN:.0f}～{RPM_LIMIT_MAX:.0f} 转/分之间，"
            f"当前：{rpm:g}"
        )
    return rpm


def validate_rpm_range(rpm_min_raw: Any, rpm_max_raw: Any) -> tuple[float, float]:
    rpm_min = parse_rpm(rpm_min_raw, "铭牌允许转速下限")
    rpm_max = parse_rpm(rpm_max_raw, "铭牌允许转速上限")
    if rpm_min >= rpm_max:
        raise InputValidationError(
            f"转速下限（{rpm_min:g}）必须严格小于上限（{rpm_max:g}）"
        )
    return rpm_min, rpm_max


class SpeedCheckService:
    def __init__(self) -> None:
        self._last_result: dict[str, Any] | None = None

    def inspect(self, wav_raw: bytes, rpm_min_raw: Any, rpm_max_raw: Any) -> dict[str, Any]:
        # 先完成全部输入校验与解析，任何一步失败都不会产生结果
        rpm_min, rpm_max = validate_rpm_range(rpm_min_raw, rpm_max_raw)
        audio = parse_wav(wav_raw)

        analysis = analyze(
            audio.segments,
            sample_rate=SAMPLE_RATE,
            rpm_min=rpm_min,
            rpm_max=rpm_max,
        )

        result = {
            "id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "audio": {
                "duration": round(audio.duration, 4),
                "sample_rate": SAMPLE_RATE,
                "segments": 3,
            },
            "rpm_range": {"min": rpm_min, "max": rpm_max},
            "verdict": analysis["verdict"],
            "confirmed_interval": analysis["confirmed_interval"],
            "midpoint_orders": analysis["midpoint_orders"],
            "candidate_intervals": analysis["candidate_intervals"],
            "segments": analysis["segments"],
            "warnings": analysis["warnings"],
        }
        self._last_result = result
        return result

    def get_last(self) -> dict[str, Any] | None:
        return None if self._last_result is None else dict(self._last_result)

    def clear(self) -> None:
        self._last_result = None
