import gc
import sys
import subprocess
import tempfile
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, Response

app = FastAPI()


async def save_uploadfile(upload: UploadFile, dst: Path, chunk_size: int = 1024 * 1024):
    with dst.open("wb") as f:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    await upload.close()


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DaVinci OTIO Beat Cutter</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 18px; }
    label { display:block; margin-top: 12px; font-weight: 600; }
    input, select, button { width: 100%; padding: 10px; margin-top: 6px; }
    button { cursor: pointer; font-weight: 700; }
    .hint { color:#666; font-size: 14px; margin-top: 10px; }
    .row { display:flex; gap:12px; }
    .row > div { flex:1; }
    .small { font-weight: 400; color:#666; font-size: 12px; }
  </style>
</head>
<body>
  <h2>DaVinci OTIO Beat Cutter</h2>
  <div class="card">
    <form id="beatForm" action="/process" method="post" enctype="multipart/form-data">
      <label>OTIO файл (timeline.otio)</label>
      <input type="file" name="timeline" accept=".otio" required />

      <label>MP3 / WAV трек</label>
      <input type="file" name="music" accept=".mp3,.wav,audio/mpeg,audio/wav" required />

      <label>FPS (если нужно)</label>
      <input type="number" name="fps" value="25" min="1" max="120" />

      <hr>

      <h3>Настройки монтажа</h3>

      <label>
        <input type="checkbox" id="ae_enabled" checked />
        Автомонтаж под музыку
      </label>

      <div class="row">
        <div>
          <label>Сдвиг бита (ms)</label>
          <input type="range" id="beat_offset_ms" min="-200" max="200" step="1" value="0" />
          <div class="small">Текущее: <span id="beat_offset_ms_v">0</span></div>
        </div>
        <div>
          <label>Микро-сдвиг сетки (ms)</label>
          <input type="range" id="grid_offset_ms" min="-80" max="80" step="1" value="0" />
          <div class="small">Текущее: <span id="grid_offset_ms_v">0</span></div>
        </div>
      </div>

      <div class="row">
        <div>
          <label>Длина клипа MIN (сек)</label>
          <input type="range" id="cd_min" min="0.2" max="5" step="0.1" value="0.5" />
          <div class="small">Текущее: <span id="cd_min_v">0.5</span></div>
        </div>
        <div>
          <label>Длина клипа MAX (сек)</label>
          <input type="range" id="cd_max" min="0.2" max="5" step="0.1" value="1.9" />
          <div class="small">Текущее: <span id="cd_max_v">1.9</span></div>
        </div>
      </div>

      <div class="row">
        <div>
          <label>Поиск транзиента MIN (сек)</label>
          <input type="range" id="ct_min" min="0.2" max="5" step="0.1" value="0.4" />
          <div class="small">Текущее: <span id="ct_min_v">0.4</span></div>
        </div>
        <div>
          <label>Поиск транзиента MAX (сек)</label>
          <input type="range" id="ct_max" min="0.2" max="5" step="0.1" value="2.1" />
          <div class="small">Текущее: <span id="ct_max_v">2.1</span></div>
        </div>
      </div>

      <label>Аудио-детектор</label>
      <select id="audio_lib">
        <option value="librosa_def" selected>librosa (обычно)</option>
        <option value="aubio_def">aubio (если плохо ловит)</option>
      </select>

      <label>Если клип короткий</label>
      <select id="if_short">
        <option value="delete" selected>удалить</option>
        <option value="keep">оставить</option>
      </select>

      <label>Если клип длинный</label>
      <select id="if_long">
        <option value="cut" selected>порезать</option>
        <option value="delete">удалить</option>
      </select>

      <label>Shuffle (%)</label>
      <input type="range" id="shuffle" min="0" max="100" step="1" value="0" />
      <div class="small">Текущее: <span id="shuffle_v">0</span></div>

      <label>Shuffle после нарезки длинных (%)</label>
      <input type="range" id="shuffle_long" min="0" max="100" step="1" value="0" />
      <div class="small">Текущее: <span id="shuffle_long_v">0</span></div>

      <hr>

      <h3>Ретайм (скорость)</h3>
      <label>
        <input type="checkbox" id="rt_enabled" />
        Включить ретайм
      </label>

      <label>Вероятность ретайма (%)</label>
      <input type="range" id="rt_prob" min="0" max="100" step="1" value="30" />
      <div class="small">Текущее: <span id="rt_prob_v">30</span></div>

      <div class="row">
        <div>
          <label>Скорость MIN (%)</label>
          <input type="range" id="rt_min" min="10" max="200" step="1" value="90" />
          <div class="small">Текущее: <span id="rt_min_v">90</span></div>
        </div>
        <div>
          <label>Скорость MAX (%)</label>
          <input type="range" id="rt_max" min="10" max="200" step="1" value="110" />
          <div class="small">Текущее: <span id="rt_max_v">110</span></div>
        </div>
      </div>

      <label>Алгоритм ретайма</label>
      <select id="rt_algo">
        <option value="optical-flow" selected>optical-flow</option>
        <option value="frame-blending">frame-blending</option>
        <option value="floor">floor</option>
      </select>

      <label>
        <input type="checkbox" id="skip_exist" checked />
        Не трогать клипы с уже заданной скоростью
      </label>

      <label>
        <input type="checkbox" id="apply_audio" checked />
        Применять к аудио
      </label>

      <!-- Скрытое поле, куда кладём JSON -->
      <input type="hidden" name="settings" id="settings_json" />

      <button type="submit">Смонтировать и скачать OTIO</button>
    </form>

    <div class="hint">
      После нажатия подожди: обработка зависит от длины трека.
    </div>
  </div>

<script>
  function bindRange(id, outId) {
    const el = document.getElementById(id);
    const out = document.getElementById(outId);
    const upd = () => out.textContent = el.value;
    el.addEventListener('input', upd);
    upd();
  }

  ["beat_offset_ms","grid_offset_ms","cd_min","cd_max","ct_min","ct_max","shuffle","shuffle_long","rt_prob","rt_min","rt_max"].forEach(id => {
    bindRange(id, id + "_v");
  });

  document.getElementById("beatForm").addEventListener("submit", (e) => {
    const cdMin = parseFloat(cd_min.value), cdMax = parseFloat(cd_max.value);
    const ctMin = parseFloat(ct_min.value), ctMax = parseFloat(ct_max.value);
    const rtMin = parseInt(rt_min.value), rtMax = parseInt(rt_max.value);

    if (cdMin > cdMax || ctMin > ctMax || rtMin > rtMax) {
      e.preventDefault();
      alert("Проверь диапазоны: MIN не должен быть больше MAX");
      return;
    }

    const settings = {
      auto_edit_to_music_1: {
        f_enabled: ae_enabled.checked,
        beat_offset: parseFloat(beat_offset_ms.value) / 1000,
        grid_offset: parseFloat(grid_offset_ms.value) / 1000,
        clip_duration: [cdMin, cdMax],
        clip_transient: [ctMin, ctMax],
        audio_lib: audio_lib.value,
        if_short_clip: if_short.value,
        if_long_clip: if_long.value,
        shuffle: parseInt(shuffle.value, 10),
        shuffle_long: parseInt(shuffle_long.value, 10),
        video_track_name: "vid1",
        audio_track_name: "aud1"
      },
      retime_simple: {
        f_enabled: rt_enabled.checked,
        retime_prob: parseInt(rt_prob.value, 10),
        retime_factor: [rtMin, rtMax],
        algorithm: rt_algo.value,
        skip_exist: skip_exist.checked,
        apply_to_audio: apply_audio.checked,
        video_track_name: "vid1",
        audio_track_name: "aud1"
      }
    };

    settings_json.value = JSON.stringify(settings);
  });
</script>

</body>
</html>
"""


@app.post("/process")
async def process(
    timeline: UploadFile = File(...),
    music: UploadFile = File(...),
    fps: int = Form(25),
    settings: str = Form(None),  # ✅ добавили
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

            # ✅ сохраняем settings в файл (если пришли)
            settings_path = tmp_path / "settings.json"
            settings_obj = {}
            if settings:
                try:
                    settings_obj = json.loads(settings)
                except json.JSONDecodeError:
                    return HTMLResponse("<pre>Ошибка: settings JSON битый</pre>", status_code=400)

                settings_path.write_text(json.dumps(settings_obj, ensure_ascii=False), encoding="utf-8")

            cmd = [
                sys.executable,
                "main.py",
                "--timeline", str(in_otio),
                "--music", str(in_mp3),
                "--out", str(out_otio),
                "--fps", str(fps),
            ]

            # ✅ передаём settings только если они есть
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
