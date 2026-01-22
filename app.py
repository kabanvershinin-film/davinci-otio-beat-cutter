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

    details.help { margin-bottom: 16px; background:#f7f7f7; border:1px solid #e3e3e3; border-radius:12px; padding: 12px 14px; }
    details.help summary { cursor:pointer; font-weight: 800; }
    details.help summary::-webkit-details-marker { display:none; }
    details.help summary::before { content: "▶ "; }
    details.help[open] summary::before { content: "▼ "; }
    details.help h4 { margin: 12px 0 6px; }
    details.help p, details.help li { line-height: 1.4; }
    details.help ul { margin: 6px 0 10px 18px; }
    .tag { display:inline-block; padding: 2px 8px; border-radius: 999px; background:#fff; border:1px solid #ddd; font-size: 12px; margin: 2px 6px 2px 0; }
    .note { color:#555; font-size: 13px; margin: 6px 0; }
    .warn { color:#7a4a00; font-size: 13px; margin: 6px 0; }

  </style>
</head>
<body>
  <h2>DaVinci OTIO Beat Cutter</h2>
  <div class="card">

    <details class="help">
      <summary>Памятка: как это работает и что делают настройки</summary>

      <h4>Как работает сервис</h4>
      <ul>
        <li>Загрузи <b>OTIO</b> (таймлайн из DaVinci Resolve) и <b>MP3/WAV</b> (музыку).</li>
        <li>Сервис анализирует трек, находит точки ритма и пересобирает таймлайн так, чтобы <b>склейки попадали в ритм</b>.</li>
        <li>На выходе ты скачиваешь новый <b>.otio</b> и открываешь его в Resolve.</li>
      </ul>

      <h4>Настройки монтажа</h4>

      <p><span class="tag">Автомонтаж под музыку</span><br>
      Если выключить — сервис почти ничего не будет менять (оставь включённым для нормальной работы).</p>

      <p><span class="tag">Длина клипа MIN / MAX</span><br>
      Ограничивает длительность каждого фрагмента на таймлайне.</p>
      <ul>
        <li><b>MIN</b> — защита от слишком коротких “морганий”.</li>
        <li><b>MAX</b> — защита от слишком длинных кусков.</li>
      </ul>

      <p><span class="tag">Поиск транзиента MIN / MAX</span><br>
      Окно, в котором сервис может “подвинуть” точку склейки к ближайшему транзиенту (резкому удару/акценту).</p>
      <ul>
        <li><b>MIN</b> — насколько рано можно искать транзиент.</li>
        <li><b>MAX</b> — насколько поздно можно искать транзиент.</li>
      </ul>

      <p><span class="tag">Аудио-детектор</span><br>
      Чем анализировать музыку (детектор ударов).</p>
      <p class="warn"><b>Важно:</b> в текущей сборке реально используется режим на базе librosa; вариант “aubio” может не работать, если библиотека не установлена на сервере.</p>

      <p><span class="tag">Если клип короткий</span><br>
      Что делать, если получившийся фрагмент меньше MIN.</p>
      <ul>
        <li><b>удалить</b> — выкинуть слишком короткий кусок.</li>
        <li><b>оставить</b> — оставить как есть (может “мелькать”).</li>
      </ul>

      <p><span class="tag">Если клип длинный</span><br>
      Что делать, если кусок получился слишком длинным.</p>
      <ul>
        <li><b>порезать</b> — разбить на несколько кусков.</li>
        <li><b>удалить</b> — выкинуть длинный кусок.</li>
      </ul>

      <p><span class="tag">Shuffle (%)</span><br>
      Перемешивает порядок клипов (в процентах). 0% — не перемешивать.</p>

      <p><span class="tag">Shuffle после нарезки длинных (%)</span><br>
      Дополнительное перемешивание после того, как длинные куски были порезаны.</p>

      <p class="note">
        Подсказка: если “вроде попадает, но чуть мимо”, обычно помогает тонкая подстройка сетки (grid/beat offset). Если у тебя эти регуляторы включены — начни с ±20–40 мс.
      </p>

      <h4>Ретайм (ускорение/замедление)</h4>
      <p><span class="tag">Включить ретайм</span><br>
      Иногда ускоряет/замедляет клипы, чтобы визуально плотнее ложились в ритм.</p>

      <p><span class="tag">Вероятность ретайма (%)</span><br>
      Не все клипы будут менять скорость. Например, 30% ≈ каждый третий.</p>

      <p><span class="tag">Retime factor MIN / MAX</span><br>
      Диапазон скорости (в процентах):</p>
      <ul>
        <li>50 = 0.5× (замедление)</li>
        <li>100 = 1.0× (норма)</li>
        <li>150 = 1.5× (ускорение)</li>
      </ul>

      <p><span class="tag">Алгоритм ретайма</span><br>
      Способ интерполяции/пересчёта кадров (если поддерживается монтажкой).</p>

      <p><span class="tag">Не трогать клипы с уже заданной скоростью</span><br>
      Если клип уже ускорен/замедлен вручную — пропустить его.</p>

      <p><span class="tag">Применять к аудио</span><br>
      Если ускоряем/замедляем клип — пытаться менять скорость привязанного аудио.</p>

      <p class="warn"><b>Примечание:</b> если в твоей версии сборщик OTIO ещё не применяет ретайм на дорожке, этот блок будет “памяткой на будущее”. Можем включить реальный ретайм в OTIO отдельным апдейтом.</p>
    </details>

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
      <input type="range" id="rt_prob" min="0" max="100" step="1" value="0" />
      <div class="small">Текущее: <span id="rt_prob_v">0</span></div>

      <div class="row">
        <div>
          <label>Retime factor MIN</label>
          <input type="range" id="rt_min" min="10" max="200" step="1" value="40" />
          <div class="small">Текущее: <span id="rt_min_v">40</span></div>
        </div>
        <div>
          <label>Retime factor MAX</label>
          <input type="range" id="rt_max" min="10" max="200" step="1" value="60" />
          <div class="small">Текущее: <span id="rt_max_v">60</span></div>
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

  ["cd_min","cd_max","ct_min","ct_max","shuffle","shuffle_long","rt_prob","rt_min","rt_max"].forEach(id => {
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
                "--grid_offset", "0.02",
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
