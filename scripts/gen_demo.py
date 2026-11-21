"""生成空转声演示 WAV（16kHz/16bit/单声道），供手工冒烟测试。

用法：python -I scripts/gen_demo.py <输出目录>
生成三类录音：
- stable_1500.wav   三段均为 1500 rpm、2/3/4 阶主导（基频弱的常见机型），
                    共同区间宽度约 30 转/分，应确认唯一转速；
- drift_rpm.wav     三段转速不同（1500/1800/2100 rpm），无共同区间，无法确认；
- even_harmonics.wav 仅含偶数阶（2/4/6/8），真实转速处阶次 gcd=2，不满足规则。
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

SR = 16_000


def synth_constant(rpm: float, duration: float, orders, seed: int = 7) -> np.ndarray:
    """整段恒定转速，f0 = rpm/60 Hz，叠加指定阶次正弦与轻微本底噪声。"""
    rng = np.random.default_rng(seed)
    n = int(round(duration * SR))
    t = np.arange(n, dtype=np.float64) / SR
    f0 = rpm / 60.0
    x = np.zeros(n, dtype=np.float64)
    for k in orders:
        # 阶次越高幅度略低，保证低阶峰也足够强
        x += (0.6 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
    x += rng.standard_normal(n) * 0.004
    return (x / np.max(np.abs(x)) * 0.9).astype(np.float32)


def synth_piecewise(rpms, duration: float, orders, seed: int = 11) -> np.ndarray:
    """三段分别采用不同转速（边界按采样点三等分），模拟转速漂移。"""
    rng = np.random.default_rng(seed)
    n = int(round(duration * SR))
    x = np.zeros(n, dtype=np.float64)
    bounds = np.linspace(0, n, 4, dtype=int)
    for seg, rpm in enumerate(rpms):
        s, e = bounds[seg], bounds[seg + 1]
        t = np.arange(s, e, dtype=np.float64) / SR
        f0 = rpm / 60.0
        for k in orders:
            x[s:e] += (0.6 / k) * np.sin(
                2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi)
            )
    x += rng.standard_normal(n) * 0.004
    return (x / np.max(np.abs(x)) * 0.9).astype(np.float32)


def save(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((samples * 32767).astype("<i2").tobytes())


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save(out / "stable_1500.wav", synth_constant(1500, 4.0, (2, 3, 4)))
    save(out / "drift_rpm.wav", synth_piecewise((1500, 1800, 2100), 4.5, (2, 3, 4)))
    save(out / "even_harmonics.wav", synth_constant(1500, 4.0, (2, 4, 6, 8)))
    print("written to", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
