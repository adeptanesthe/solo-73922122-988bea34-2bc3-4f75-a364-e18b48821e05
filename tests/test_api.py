"""端到端 API 关键流程测试。"""
from conftest import build_harmonic_wav, build_silence_wav, post_check


def test_health_and_empty_result(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/result/last").status_code == 404


def test_full_confirm_flow(client):
    wav = build_harmonic_wav(1500, duration=4.0, orders=(2, 3, 4))
    resp = post_check(client, wav, rpm_min=1400, rpm_max=1600)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["verdict"] == "confirmed"
    assert body["confirmed_interval"]["lo"] <= 1500 <= body["confirmed_interval"]["hi"]
    assert len(body["midpoint_orders"]) == 3
    assert len(body["segments"]) == 3
    # 结果可再次获取，且与当前录音/范围对应
    last = client.get("/api/result/last").json()
    assert last["id"] == body["id"]
    assert last["rpm_range"] == {"min": 1400.0, "max": 1600.0}


def test_drift_recording_unconfirmed(client):
    wav = build_harmonic_wav((1500, 1800, 2100), duration=4.5)
    body = post_check(client, wav, rpm_min=1000, rpm_max=2500).json()
    assert body["verdict"] == "unconfirmed"
    assert body["confirmed_interval"] is None


def test_invalid_rpm_range_returns_400(client):
    wav = build_harmonic_wav(1500)
    # 低于 300
    r = post_check(client, wav, rpm_min=100, rpm_max=1600)
    assert r.status_code == 400 and "300" in r.json()["detail"]
    # 高于 6000
    r = post_check(client, wav, rpm_min=300, rpm_max=7000)
    assert r.status_code == 400 and "6000" in r.json()["detail"]
    # 下限 >= 上限
    r = post_check(client, wav, rpm_min=1500, rpm_max=1500)
    assert r.status_code == 400 and "小于" in r.json()["detail"]
    # 非数值
    r = client.post(
        "/api/check",
        files={"recording": ("a.wav", wav, "audio/wav")},
        data={"rpm_min": "abc", "rpm_max": "1600"},
    )
    assert r.status_code == 400 and "数值" in r.json()["detail"]
    # 缺字段
    r = client.post(
        "/api/check",
        files={"recording": ("a.wav", wav, "audio/wav")},
        data={"rpm_min": "1400"},
    )
    assert r.status_code == 422


def test_invalid_audio_returns_400(client):
    # 时长不足 3 秒
    short = build_silence_wav(duration=2.0)
    r = post_check(client, short)
    assert r.status_code == 400 and "下限" in r.json()["detail"]
    # 非 WAV
    r = client.post(
        "/api/check",
        files={"recording": ("a.txt", b"hello", "text/plain")},
        data={"rpm_min": "1400", "rpm_max": "1600"},
    )
    assert r.status_code == 400 and ".wav" in r.json()["detail"]
    # 空文件
    r = client.post(
        "/api/check",
        files={"recording": ("a.wav", b"", "audio/wav")},
        data={"rpm_min": "1400", "rpm_max": "1600"},
    )
    assert r.status_code == 400


def test_last_result_not_created_on_invalid_input(client):
    # 先产生一条有效结果
    good = post_check(client, build_harmonic_wav(1500, orders=(2, 3, 4)), 1400, 1600)
    good_id = good.json()["id"]
    # 非法输入被拒
    bad = post_check(client, build_silence_wav(duration=2.0))
    assert bad.status_code == 400
    # 上一次有效结果仍在
    assert client.get("/api/result/last").json()["id"] == good_id


def test_clear_result(client):
    post_check(client, build_harmonic_wav(1500, orders=(2, 3, 4)), 1400, 1600)
    assert client.delete("/api/result/last").json() == {"status": "cleared"}
    assert client.get("/api/result/last").status_code == 404


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "主轴空转转速" in resp.text
