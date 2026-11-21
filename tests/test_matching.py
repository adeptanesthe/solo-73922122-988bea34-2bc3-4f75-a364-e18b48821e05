"""谱峰提取、阶次支持、区间求交与完整分析测试。"""
import numpy as np
import pytest

from app.audio import SAMPLE_RATE, parse_wav
from app.matching import (
    BAND_MAX_HZ,
    BAND_MIN_HZ,
    MAX_PEAKS,
    analyze,
    extract_peaks,
    intersect_interval_lists,
    orders_at_rpm,
    support_intervals_for_segment,
)
from conftest import build_harmonic_wav, build_silence_wav


def _first_segment(wav: bytes) -> np.ndarray:
    return parse_wav(wav).segments[0]


def test_peaks_found_at_harmonic_frequencies():
    # 1500 rpm -> f0 = 25 Hz，1/2/3 阶 = 25/50/75 Hz
    seg = _first_segment(build_harmonic_wav(1500, duration=4.0, orders=(1, 2, 3)))
    peaks, band_max = extract_peaks(seg, SAMPLE_RATE)
    assert band_max > 0
    freqs = [p.freq for p in peaks]
    for target in (25.0, 50.0, 75.0):
        assert min(abs(f - target) for f in freqs) <= 1.0
    # 所有峰在 5～800Hz 内，且相对功率均达到 1%
    assert all(BAND_MIN_HZ <= p.freq <= BAND_MAX_HZ for p in peaks)
    assert all(p.power_rel >= 0.01 - 1e-12 for p in peaks)


def test_peak_count_capped_at_eight():
    # 8 个阶次恰好 8 个主峰；再加入更多谐波也只能保留 8 个
    seg = _first_segment(
        build_harmonic_wav(600, duration=4.0, orders=tuple(range(1, 9)))
    )
    peaks, _ = extract_peaks(seg, SAMPLE_RATE)
    assert len(peaks) <= MAX_PEAKS
    # 按频率升序返回
    assert [p.freq for p in peaks] == sorted(p.freq for p in peaks)


def test_silent_segment_has_no_peaks():
    seg = parse_wav(build_silence_wav(duration=4.0)).segments[0]
    peaks, band_max = extract_peaks(seg, SAMPLE_RATE)
    assert peaks == []
    assert band_max == 0.0


def test_support_intervals_center_on_true_rpm():
    seg = _first_segment(build_harmonic_wav(1500, duration=4.0, orders=(1, 2, 3)))
    peaks, _ = extract_peaks(seg, SAMPLE_RATE)
    intervals = support_intervals_for_segment(peaks, 300.0, 6000.0)
    assert intervals
    # 1500 rpm 必须落在某个区间内
    assert any(lo <= 1500.0 <= hi for lo, hi in intervals)
    # 1Hz 容差换算到最高的 3 阶：半宽 60/3=20 rpm，区间宽度约 40 rpm
    # （区间宽度由满足条件的最高阶容差决定，而非 1 阶）
    cover = [iv for iv in intervals if iv[0] <= 1500.0 <= iv[1]][0]
    assert cover[1] - cover[0] == pytest.approx(40.0, abs=2.0)


def test_even_harmonics_rejected_at_true_speed():
    # 只有 2/4/6/8 阶：在真实转速 1500rpm 处支持阶次 gcd=2，规则拒绝；
    # 半速 750rpm 处可能出现 gcd=1 的别名，故只断言真实转速不受支持
    seg = _first_segment(
        build_harmonic_wav(1500, duration=4.0, orders=(2, 4, 6, 8))
    )
    peaks, _ = extract_peaks(seg, SAMPLE_RATE)
    intervals = support_intervals_for_segment(peaks, 300.0, 6000.0)
    assert not any(lo <= 1500.0 <= hi for lo, hi in intervals)
    details = orders_at_rpm(peaks, 1500.0)
    from functools import reduce
    from math import gcd
    assert reduce(gcd, [d["order"] for d in details]) == 2


def test_fewer_than_three_orders_rejected():
    seg = _first_segment(build_harmonic_wav(1500, duration=4.0, orders=(1, 2)))
    peaks, _ = extract_peaks(seg, SAMPLE_RATE)
    assert support_intervals_for_segment(peaks, 300.0, 6000.0) == []


def test_orders_at_rpm_tolerance_one_hz():
    seg = _first_segment(build_harmonic_wav(1500, duration=4.0, orders=(1, 2, 3)))
    peaks, _ = extract_peaks(seg, SAMPLE_RATE)
    details = orders_at_rpm(peaks, 1500.0)
    got_orders = {d["order"] for d in details}
    assert {1, 2, 3}.issubset(got_orders)
    assert all(d["error_hz"] <= 1.0 for d in details)
    # 偏差 100 rpm 时 1 阶频率差约 1.67 Hz，1 阶不再支持
    far = orders_at_rpm(peaks, 1600.0)
    assert 1 not in {d["order"] for d in far}


def test_intersect_interval_lists():
    a = [(300.0, 500.0), (1000.0, 1200.0)]
    b = [(400.0, 600.0), (1100.0, 1150.0)]
    c = [(450.0, 480.0), (1120.0, 1300.0)]
    assert intersect_interval_lists([a, b, c]) == [(450.0, 480.0), (1120.0, 1150.0)]
    # 单点相交也算
    assert intersect_interval_lists([[(100.0, 200.0)], [(200.0, 300.0)]]) == [(200.0, 200.0)]
    # 任一为空 -> 空
    assert intersect_interval_lists([a, []]) == []


def test_analyze_confirms_stable_speed():
    # 基频弱、2/3/4 阶主导：共同区间宽度由 4 阶容差决定（约 30rpm），
    # 是可被确认的典型机型；含 1 阶时区间宽约 40rpm，按规则不可确认
    audio = parse_wav(build_harmonic_wav(1500, duration=4.0, orders=(2, 3, 4)))
    result = analyze(audio.segments, SAMPLE_RATE, 1400.0, 1600.0)
    assert result["verdict"] == "confirmed"
    iv = result["confirmed_interval"]
    assert iv["lo"] <= 1500.0 <= iv["hi"]
    assert iv["width"] <= 30.0
    # 中点在每段都有 ≥3 个支持阶次且 gcd=1
    assert len(result["midpoint_orders"]) == 3
    for seg in result["midpoint_orders"]:
        assert len(seg["orders"]) >= 3
        from math import gcd
        from functools import reduce
        assert reduce(gcd, seg["orders"]) == 1
        assert all(1 <= o <= 8 for o in seg["orders"])


def test_analyze_unconfirmed_when_segments_drift():
    audio = parse_wav(
        build_harmonic_wav((1500, 1800, 2100), duration=4.5, orders=(1, 2, 3))
    )
    result = analyze(audio.segments, SAMPLE_RATE, 1000.0, 2500.0)
    assert result["verdict"] == "unconfirmed"
    assert result["confirmed_interval"] is None
    assert result["candidate_intervals"] == []
    assert any("没有共同支持" in w for w in result["warnings"])


def test_analyze_wide_interval_is_unconfirmed():
    # 含 1 阶时共同区间宽度约 40 rpm，超过 30 rpm 上限 -> 无法确认
    audio = parse_wav(build_harmonic_wav(600, duration=4.0, orders=(1, 2, 3)))
    result = analyze(audio.segments, SAMPLE_RATE, 300.0, 6000.0)
    assert result["verdict"] == "unconfirmed"
    assert result["confirmed_interval"] is None
    widest = max(c["width"] for c in result["candidate_intervals"])
    assert widest > 30.0
    assert any("30 转/分" in w for w in result["warnings"])


def test_analyze_silence_unconfirmed():
    audio = parse_wav(build_silence_wav(duration=4.0))
    result = analyze(audio.segments, SAMPLE_RATE, 300.0, 6000.0)
    assert result["verdict"] == "unconfirmed"
    assert result["candidate_intervals"] == []
    assert all(s["peak_count"] == 0 for s in result["segments"])
