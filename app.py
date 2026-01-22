import gc
import sys
import subprocess
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import librosa
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, Response, JSONResponse

app = FastAPI()


async def save_uploadfile(upload: UploadFile, dst: Path, chunk_size: int = 1024 * 1024):
    with dst.open("wb") as f:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    await upload.close()


def get_audio_duration_sec(audio_path: str) -> float:
    with sf.SoundFile(audio_path) as f:
        return float(len(f) / f.samplerate)


def _peaks_for_waveform(y: np.ndarray, points: int = 1200) -> List[float]:
    """
    Лёгкие пики для waveform (если захочешь рисовать на фронте не по audio blob, а по peaks).
    Сейчас WaveSurfer рисует сам, но peaks отдаём на будущее.
    """
    if y.size == 0:
        return [0.0]
    y = np.abs(y.astype(np.float32))
    n = y.size
    points = max(100, int(points))
    hop = max(1, n // points)
    peaks = []
    for i in range(0, n, hop):
        peaks.append(float(y[i:i + hop].max()))
        if len(peaks) >= points:
            break
    mx = max(peaks) if peaks else 1.0
    if mx > 0:
        peaks = [p / mx for p in peaks]
    return peaks


def aggressive_regions_from_audio(
    audio_path: str,
    sr: int = 11025,
    analyze_seconds: float = 360.0,
    percentile: float = 85.0,
    smooth_sec: float = 0.35,
    min_len_sec: float = 0.60,
    merge_gap_sec: float = 0.25,
    pad_sec: float = 0.15,
) -> Dict[str, Any]:
    """
    Aggressive preset:
    - берём только жирные по энергии куски (верхний перцентиль),
    - сглаживаем,
    - собираем интервалы,
    - чистим короткие/склеиваем близкие,
    - добавляем небольшой pad.
    """
    duration = get_audio_duration_sec(audio_path)

    y, _sr = librosa.load(audio_path, sr=sr, mono=True, duration=float(analyze_seconds))
    if y is None or len(y) == 0:
        return {"duration": duration, "regions": [], "peaks": [0.0]}

    hop_length = 512
    frame_length = 2048

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size == 0:
        return {"duration": duration, "regions": [], "peaks": _peaks_for_waveform(y)}

    # нормализация 0..1
    rms = rms.astype(np.float32)
    rms = rms - float(rms.min())
    mx = float(rms.max())
    if mx > 0:
        rms = rms / mx

    # сглаживание по времени (moving average)
    frames_per_sec = sr / hop_length
    win = max(1, int(round(float(smooth_sec) * frames_per_sec)))
    if win > 1:
        kernel = np.ones(win, dtype=np.float32) / float(win)
        rms = np.convolve(rms, kernel, mode="same")

    # порог по перцентилю
    thr = float(np.percentile(rms, float(percentile)))
    mask = rms >= thr

    # перевод frame -> time
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # собрать интервалы по маске
    regions = []
    in_seg = False
    start_t = 0.0
    for i, on in enumerate(mask):
        t = float(times[i])
        if on and not in_seg:
            in_seg = True
            start_t = t
        if in_seg and (not on or i == len(mask) - 1):
            end_t = t if not on else float(times[i])
            if end_t > start_t:
                regions.append([start_t, end_t])
            in_seg = False

    # паддинг + обрезка в 0..duration
    def clamp(a: float, b: float) -> List[float]:
        a = max(0.0, a - pad_sec)
        b = min(duration, b + pad_sec)
        return [float(a), float(b)]

    regions = [clamp(a, b) for a, b in regions]

    # убрать короткие
    regions = [r for r in regions if (r[1] - r[0]) >= float(min_len_sec)]

    # склеить близкие
    regions.sort(key=lambda x: x[0])
    merged = []
    for r in regions:
        if not merged:
            merged.append(r)
            continue
        prev = merged[-1]
        if r[0] - prev[1] <= float(merge_gap_sec):
            prev[1] = max(prev[1], r[1])
        else:
            merged.append(r)

    # финальная нормализация
    out_regions = [{"start": float(a), "end": float(b)} for a, b in merged if b > a]

    peaks = _peaks_for_waveform(y)

    del y, rms, mask
    gc.collect()

    return {
        "duration": float(duration),
        "regions": out_regions,
        "peaks": peaks,
        "preset": "aggressive",
        "params": {
            "percentile": percentile,
            "smooth_sec": smooth_sec,
            "min_len_sec": min_len_sec,
            "merge_gap_sec": merge_gap_sec,
            "pad_sec": pad_sec,
        },
    }


@app.get("/", response_class=HTMLResponse)
def home():
    # отдаём index.html из файла (красивее и проще поддерживать)
    p = Path(__file__).with_name("index.html")
    if not p.exists():
        return HTMLResponse("<h3>index.html not found рядом с app.py</h3>", status_code=500)
    return HTMLResponse(p.read_text(encoding="utf-8"))


@app.post("/analyze-audio")
async def analyze_audio(music: UploadFile = File(...)):
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_audio = tmp_path / "music_input"
            await save_uploadfile(music, in_audio)

            # анализ
            result = aggressive_regions_from_audio(
                str(in_audio),
                sr=11025,
                analyze_seconds=360.0,
                percentile=85.0,
                smooth_sec=0.35,
                min_len_sec=0.60,
                merge_gap_sec=0.25,
                pad_sec=0.15,
            )
            return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
    finally:
        gc.collect()


@app.post("/process")
async def process(
    timeline: UploadFile = File(...),
    music: UploadFile = File(...),
    fps: int = Form(25),
    settings: str = Form(None),
    beat_regions: str = Form(None),  # ✅ новые зоны (пока просто передаём дальше в settings)
):
    result = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_otio = tmp_path / "input.otio"
            in_mp3 = tmp_path / "music.mp3"
            out_otio = tmp_path / "output.otio"

            await save_uploadfile(timeline, in_otio)
            await save_uploadfile(music, in_mp3)
            gc.collect()

            settings_path = tmp_path / "settings.json"
            settings_obj = {}

            if settings:
                try:
                    settings_obj = json.loads(settings)
                except json.JSONDecodeError:
                    return HTMLResponse("<pre>Ошибка: settings JSON битый</pre>", status_code=400)

            # сохраняем beat_regions (если пришли) в settings, чтобы не терялись
            if beat_regions:
                try:
                    regions_obj = json.loads(beat_regions)
                    # кладём аккуратно в отдельную ветку
                    settings_obj.setdefault("music_analysis", {})
                    settings_obj["music_analysis"]["preset"] = "aggressive"
                    settings_obj["music_analysis"]["regions"] = regions_obj
                except json.JSONDecodeError:
                    return HTMLResponse("<pre>Ошибка: beat_regions JSON битый</pre>", status_code=400)

            if settings_obj:
                settings_path.write_text(json.dumps(settings_obj, ensure_ascii=False), encoding="utf-8")

            cmd = [
                sys.executable,
                "main.py",
                "--timeline", str(in_otio),
                "--music", str(in_mp3),
                "--out", str(out_otio),
                "--fps", str(fps),
                "--grid_offset", "0.02",
            ]

            if settings_obj:
                cmd += ["--settings", str(settings_path)]

            result = subprocess.run(cmd, capture_output=True, text=True)
            gc.collect()

            if result.returncode != 0 or not out_otio.exists():
                msg = "\n".join([
                    (result.stderr or "").strip(),
                    (result.stdout or "").strip(),
                ]) or "Unknown error"
                return HTMLResponse(
                    f"<pre style='white-space:pre-wrap'>Ошибка обработки:\n{msg}</pre>",
                    status_code=500,
                )

            data = out_otio.read_bytes()
            return Response(
                content=data,
                media_type="application/octet-stream",
                headers={"Content-Disposition": 'attachment; filename="output.otio"'},
            )

    except Exception as e:
        return HTMLResponse(
            f"<pre style='white-space:pre-wrap'>SERVER ERROR:\n{type(e).__name__}: {e}</pre>",
            status_code=500,
        )
    finally:
        gc.collect()
