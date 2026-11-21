"""服务层输入校验测试。"""
import pytest

from app.service import (
    RPM_LIMIT_MAX,
    RPM_LIMIT_MIN,
    InputValidationError,
    SpeedCheckService,
    validate_rpm_range,
)
from conftest import build_harmonic_wav


@pytest.mark.parametrize(
    "lo,hi",
    [(300, 6000), (300.0, 301.0), ("1400", "1600"), (" 1400 ", " 1600 ")],
)
def test_rpm_range_accepted(lo, hi):
    assert validate_rpm_range(lo, hi) == (float(str(lo).strip()), float(str(hi).strip()))


@pytest.mark.parametrize(
    "lo,hi,word",
    [
        (299, 1600, "300"),
        (1400, 6001, "6000"),
        (0, 100, "300"),
        ("abc", 1600, "数值"),
        (None, 1600, "填写"),
        ("", 1600, "填写"),
        (1600, 1400, "小于"),
        (1500, 1500, "小于"),
    ],
)
def test_rpm_range_rejected(lo, hi, word):
    with pytest.raises(InputValidationError, match=word):
        validate_rpm_range(lo, hi)


def test_service_result_shape():
    service = SpeedCheckService()
    result = service.inspect(build_harmonic_wav(1500, orders=(2, 3, 4)), 1400, 1600)
    for key in (
        "id", "timestamp", "audio", "rpm_range", "verdict",
        "confirmed_interval", "midpoint_orders", "candidate_intervals",
        "segments", "warnings",
    ):
        assert key in result
    assert result["audio"]["sample_rate"] == 16000
    assert result["audio"]["segments"] == 3
    assert service.get_last()["id"] == result["id"]
    service.clear()
    assert service.get_last() is None


def test_rpm_limits_constants():
    assert RPM_LIMIT_MIN == 300.0
    assert RPM_LIMIT_MAX == 6000.0
