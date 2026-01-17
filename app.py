import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse


app = FastAPI()


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
    input, button { width: 100%; padding: 10px; margin-top: 6px; }
    button { cursor: pointer; font-weight: 700; }
    .hint { color:#666; font-size: 14px; margin-top: 10px; }
  </style>
</head>
<body>
  <h2>DaVinci OTIO Beat Cutter</h2>
  <div class="card">
    <form action="/process" method="post" enctype="multipart/form-data">
      <label>OTIO файл (timeline.otio)</label>
      <input type="file" name="timeline" accept=".otio" required />

      <label>MP3 трек (music.mp3)</label>
      <input type="file" name="music" accept=".mp3,audio/mpeg" required />

      <label>FPS (если нужно)</label>
      <input type="number" name="fps" value="25" min="1" max="120" />

      <button type="submit">Смонтировать и скачать OTIO</button>
    </form>

    <div class="hint">
      После нажатия подожди: обработка зависит от длины трека.
    </div>
  </div>
</body>
</html>
"""


@app.post("/process")
async def process(
    timeline: UploadFile = File(...),
    music: UploadFile = File(...),
    fps: int = Form(25),
):
    # Временная папка на сервере
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        in_otio = tmp_path / "input.otio"
        in_mp3 = tmp_path / "music.mp3"
        out_otio = tmp_path / "output.otio"

        # Сохраняем загрузки
        in_otio.write_bytes(await timeline.read())
        in_mp3.write_bytes(await music.read())

        # Запускаем твой CLI-скрипт main.py
        # Он у тебя в README уже описан примерно так:
        # python main.py --timeline input.otio --music track.mp3 --out output.otio --fps 25
        cmd = [
            "python",
            "main.py",
            "--timeline",
            str(in_otio),
            "--music",
            str(in_mp3),
            "--out",
            str(out_otio),
            "--fps",
            str(fps),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 or not out_otio.exists():
        err = (result.stderr or "").strip()
        out = (result.stdout or "").strip()

        # ВАЖНО: чтобы это было видно в Render Logs
        print("=== main.py failed ===")
        print("STDERR:\n", err)
        print("STDOUT:\n", out)

        msg = "\n".join([x for x in [err, out] if x]) or "Unknown error"
        return HTMLResponse(
            f"<pre style='white-space:pre-wrap'>Ошибка обработки:\n{msg}</pre>",
            status_code=500,
        )

    # Если ошибок нет — отдаем файл
    return FileResponse(
        path=str(out_otio),
        filename="output.otio",
        media_type="application/octet-stream",
    )
