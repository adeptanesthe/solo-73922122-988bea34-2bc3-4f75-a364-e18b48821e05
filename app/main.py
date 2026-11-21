"""FastAPI 入口：上传空转 WAV 与铭牌转速范围，返回转速核对结果。"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .audio import AudioValidationError, MAX_DURATION_SEC, MAX_UPLOAD_BYTES, MIN_DURATION_SEC, SAMPLE_RATE
from .service import InputValidationError, SpeedCheckService

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="主轴空转转速声音核对", version="1.0.0")
service = SpeedCheckService()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/result/last")
def last_result() -> dict:
    result = service.get_last()
    if result is None:
        raise HTTPException(status_code=404, detail="暂无有效核对结果")
    return result


@app.delete("/api/result/last")
def clear_result() -> dict[str, str]:
    service.clear()
    return {"status": "cleared"}


@app.post("/api/check")
async def check_speed(
    recording: UploadFile = File(..., description="空转声 WAV"),
    rpm_min: str = Form(..., description="铭牌允许转速下限（转/分）"),
    rpm_max: str = Form(..., description="铭牌允许转速上限（转/分）"),
) -> dict:
    if not (recording.filename or "").lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="录音必须是 .wav 文件")
    raw = await recording.read()
    if not raw:
        raise HTTPException(status_code=400, detail="录音文件为空")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"录音文件过大；仅支持 {SAMPLE_RATE}Hz/16bit/单声道、"
                f"{MIN_DURATION_SEC:.0f}～{MAX_DURATION_SEC:.0f} 秒的 WAV"
            ),
        )

    try:
        return service.inspect(raw, rpm_min, rpm_max)
    except InputValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AudioValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
