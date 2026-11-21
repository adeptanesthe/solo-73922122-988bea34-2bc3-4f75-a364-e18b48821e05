"""空转声谱分析与主轴转速区间推断。

处理流程（严格按核对规则）：
1. 每段录音做加 Hann 窗 FFT，只保留 5～800 Hz 频带；
2. 在该频带内拾取局部谱峰，峰值功率须达到频带最大功率的 1%，
   按功率降序最多保留 8 个；
3. 对 1～8 阶逐个判断：若某转速 r（转/分）的 k 阶倍频 k*r/60
   与某个谱峰频率相差不超过 1 Hz，则该转速得到阶次 k 支持；
4. 一个转速在每段都须有至少 3 个不同阶次支持，且这些阶次的
   最大公约数为 1（排除仅由高阶谐波构成的伪转速）；
5. 取三段共同支持的连续转速区间，单点也算区间；
6. 仅当共同区间只有一个且宽度不超过 30 转/分时确认转速，
   否则只列候选区间并提示无法确认。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import gcd

import numpy as np

BAND_MIN_HZ = 5.0
BAND_MAX_HZ = 800.0
MAX_PEAKS = 8
PEAK_POWER_RATIO = 0.01
MIN_ORDER = 1
MAX_ORDER = 8
FREQ_TOL_HZ = 1.0
MIN_SUPPORT_ORDERS = 3
CONFIRM_MAX_WIDTH_RPM = 30.0
RPM_EPS = 1e-9


@dataclass
class Peak:
    freq: float       # Hz
    power: float      # 相对功率谱值
    power_rel: float  # 相对该段频带最大功率的比例（0～1）


@dataclass
class SegmentAnalysis:
    index: int                  # 段序号，从 1 开始
    peaks: list[Peak]
    band_max_power: float
    supported_intervals: list[tuple[float, float]]  # 本段满足规则的转速区间
    midpoint_orders: list[int]  # 由 analyze 回填（确认区间中点对应的支持阶次）


def extract_peaks(segment: np.ndarray, sample_rate: int) -> tuple[list[Peak], float]:
    """对一段录音做加窗 FFT，返回 5～800Hz 内合格的局部谱峰（至多 8 个）。"""
    n = len(segment)
    x = segment.astype(np.float64, copy=True)
    x -= np.mean(x)  # 去直流，避免 0Hz 泄漏污染低频
    window = np.hanning(n)
    window_energy = float(np.sum(window ** 2))
    spectrum = np.fft.rfft(x * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    # 窗能量归一的单边功率谱，段间功率量纲一致
    power = (np.abs(spectrum) ** 2) / max(window_energy, 1e-300)

    band = (freqs >= BAND_MIN_HZ) & (freqs <= BAND_MAX_HZ)
    band_idx = np.flatnonzero(band)
    if band_idx.size == 0:
        return [], 0.0
    band_power = power[band_idx]
    band_max = float(np.max(band_power))
    if band_max <= 0.0:
        return [], 0.0

    # 局部极大值：严格大于左右相邻频点
    local_max = np.zeros(len(power), dtype=bool)
    local_max[1:-1] = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
    candidates = [
        i for i in band_idx if local_max[i] and power[i] >= PEAK_POWER_RATIO * band_max
    ]
    # 按功率降序，最多取 8 个最强谱峰
    candidates.sort(key=lambda i: float(power[i]), reverse=True)
    kept = candidates[:MAX_PEAKS]
    kept.sort(key=lambda i: float(freqs[i]))
    peaks = [
        Peak(
            freq=float(freqs[i]),
            power=float(power[i]),
            power_rel=float(power[i] / band_max),
        )
        for i in kept
    ]
    return peaks, band_max


def _gcd_of(values: set[int]) -> int:
    return reduce(gcd, sorted(values))


def _qualifies(orders: set[int]) -> bool:
    """至少 3 个不同阶次且阶次最大公约数为 1。"""
    return len(orders) >= MIN_SUPPORT_ORDERS and _gcd_of(orders) == 1


def support_intervals_for_segment(
    peaks: list[Peak], rpm_min: float, rpm_max: float
) -> list[tuple[float, float]]:
    """计算一段内满足「≥3 阶支持且 gcd=1」的全部连续转速区间。"""
    # 每个阶次收集其支持的转速闭区间（由 ±1Hz 容差换算，并夹到铭牌范围）
    per_order: dict[int, list[tuple[float, float]]] = {
        k: [] for k in range(MIN_ORDER, MAX_ORDER + 1)
    }
    half_width = FREQ_TOL_HZ * 60.0
    for peak in peaks:
        for k in range(MIN_ORDER, MAX_ORDER + 1):
            center = peak.freq * 60.0 / k
            lo = max(rpm_min, center - half_width / k)
            hi = min(rpm_max, center + half_width / k)
            if lo <= hi + RPM_EPS:
                per_order[k].append((lo, hi))

    # 扫描线：在每个事件坐标处分别评估「该点」与「坐标间开区间」的支持阶次集合
    starts: dict[float, set[int]] = {}
    ends: dict[float, set[int]] = {}
    for order, intervals in per_order.items():
        for lo, hi in intervals:
            starts.setdefault(lo, set()).add(order)
            ends.setdefault(hi, set()).add(order)
    coords = sorted(set(starts) | set(ends))

    pieces: list[tuple[float, float]] = []
    active: set[int] = set()
    for idx, x in enumerate(coords):
        active |= starts.get(x, set())
        if _qualifies(active):  # 坐标点 x 本身受支持（单点区间来源）
            pieces.append((x, x))
        after = active - ends.get(x, set())
        if idx + 1 < len(coords) and _qualifies(after):
            pieces.append((x, coords[idx + 1]))  # 开区间 (x, next) 内受支持
        active = after

    # 合并相切/浮点抖动相邻的片段，得到互斥的闭区间
    pieces.sort()
    merged: list[tuple[float, float]] = []
    for lo, hi in pieces:
        if merged and lo <= merged[-1][1] + RPM_EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _merge_intervals(
    intervals: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    intervals = sorted(intervals)
    merged: list[tuple[float, float]] = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1] + RPM_EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def intersect_interval_lists(
    lists: list[list[tuple[float, float]]]
) -> list[tuple[float, float]]:
    """多段区间列表求交（双指针），任一为空则结果为空。"""
    if not lists or any(not ivs for ivs in lists):
        return []
    acc = lists[0]
    for ivs in lists[1:]:
        out: list[tuple[float, float]] = []
        i = j = 0
        while i < len(acc) and j < len(ivs):
            lo = max(acc[i][0], ivs[j][0])
            hi = min(acc[i][1], ivs[j][1])
            if lo <= hi + RPM_EPS:
                out.append((lo, hi))
            if acc[i][1] < ivs[j][1]:
                i += 1
            else:
                j += 1
        acc = _merge_intervals(out)
        if not acc:
            return []
    return acc


def orders_at_rpm(peaks: list[Peak], rpm: float) -> list[dict]:
    """返回某转速在该段得到的支持阶次明细（1～8 阶，容差 1Hz）。"""
    details: list[dict] = []
    for k in range(MIN_ORDER, MAX_ORDER + 1):
        expected = k * rpm / 60.0
        best = None
        for peak in peaks:
            err = abs(peak.freq - expected)
            if err <= FREQ_TOL_HZ + 1e-12 and (best is None or err < best["error_hz"]):
                best = {
                    "order": k,
                    "expected_hz": round(expected, 2),
                    "peak_hz": round(peak.freq, 2),
                    "error_hz": round(err, 2),
                    "power_rel": round(peak.power_rel, 4),
                }
        if best is not None:
            details.append(best)
    return details


def analyze(
    segments: list[np.ndarray], sample_rate: int, rpm_min: float, rpm_max: float
) -> dict:
    """完整分析：逐段谱峰 → 逐段支持区间 → 三段共同区间 → 结论。"""
    segment_results: list[SegmentAnalysis] = []
    warnings: list[str] = []

    for idx, seg in enumerate(segments, start=1):
        peaks, band_max = extract_peaks(seg, sample_rate)
        if band_max <= 0.0:
            warnings.append(f"第 {idx} 段频带内无有效功率（疑似静音），无法支持任何转速")
        elif len(peaks) < MAX_PEAKS:
            warnings.append(
                f"第 {idx} 段达到 1% 功率门限的局部谱峰仅 {len(peaks)} 个"
                f"（上限 {MAX_PEAKS} 个）"
            )
        intervals = support_intervals_for_segment(peaks, rpm_min, rpm_max)
        if not intervals:
            warnings.append(f"第 {idx} 段不存在满足 ≥3 阶支持且阶次 gcd=1 的转速")
        segment_results.append(
            SegmentAnalysis(
                index=idx,
                peaks=peaks,
                band_max_power=band_max,
                supported_intervals=intervals,
                midpoint_orders=[],
            )
        )

    common = intersect_interval_lists([s.supported_intervals for s in segment_results])
    candidates = [
        {
            "lo": round(lo, 2),
            "hi": round(hi, 2),
            "mid": round((lo + hi) / 2.0, 2),
            "width": round(hi - lo, 2),
            "is_point": bool(hi - lo <= RPM_EPS),
        }
        for lo, hi in common
    ]

    confirmed = (
        len(common) == 1 and (common[0][1] - common[0][0]) <= CONFIRM_MAX_WIDTH_RPM + RPM_EPS
    )
    confirmed_interval = None
    midpoint_details = None
    if confirmed:
        lo, hi = common[0]
        mid = (lo + hi) / 2.0
        confirmed_interval = {
            "lo": round(lo, 2),
            "hi": round(hi, 2),
            "mid": round(mid, 2),
            "width": round(hi - lo, 2),
        }
        midpoint_details = []
        for seg_res in segment_results:
            details = orders_at_rpm(seg_res.peaks, mid)
            seg_res.midpoint_orders = [d["order"] for d in details]
            midpoint_details.append(
                {"segment": seg_res.index, "orders": seg_res.midpoint_orders, "matches": details}
            )

    if not common:
        warnings.append("三段没有共同支持的转速区间，请检查录音质量或铭牌转速范围")
    elif not confirmed:
        warnings.append(
            "共同支持的候选区间不唯一或宽度超过 30 转/分，无法确认唯一稳定转速"
        )

    return {
        "verdict": "confirmed" if confirmed else "unconfirmed",
        "confirmed_interval": confirmed_interval,
        "midpoint_orders": midpoint_details,
        "candidate_intervals": candidates,
        "segments": [
            {
                "segment": s.index,
                "peak_count": len(s.peaks),
                "peaks": [
                    {
                        "freq_hz": round(p.freq, 2),
                        "power_rel": round(p.power_rel, 4),
                    }
                    for p in s.peaks
                ],
                "supported_intervals": [
                    {"lo": round(lo, 2), "hi": round(hi, 2)}
                    for lo, hi in s.supported_intervals
                ],
            }
            for s in segment_results
        ],
        "warnings": warnings,
    }
