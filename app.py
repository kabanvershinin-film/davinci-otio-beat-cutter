import gc
import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, Response  # <-- ВАЖНО: Response


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
    result = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_otio = tmp_path / "input.otio"
            in_mp3 = tmp_path / "music.mp3"
            out_otio = tmp_path / "output.otio"

            in_otio.write_bytes(await timeline.read())
            in_mp3.write_bytes(await music.read())

            print("DEBUG in_otio exists:", in_otio.exists(), in_otio)
            print("DEBUG in_mp3 exists:", in_mp3.exists(), in_mp3)

            cmd = [
                "python",
                "main.py",
                "--timeline", str(in_otio),
                "--music", str(in_mp3),
                "--out", str(out_otio),
                "--fps", str(fps),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            print("=== main.py finished ===")
            print("returncode:", result.returncode)
            print("STDERR:\n", (result.stderr or "").strip())
            print("STDOUT:\n", (result.stdout or "").strip())
            print("DEBUG out_otio exists:", out_otio.exists(), out_otio)

            if result.returncode != 0 or not out_otio.exists():
                msg = "\n".join([
                    (result.stderr or "").strip(),
                    (result.stdout or "").strip(),
                ]) or "Unknown error"
                return HTMLResponse(
                    f"<pre style='white-space:pre-wrap'>Ошибка обработки:\n{msg}</pre>",
                    status_code=500,
                )

            # ВАЖНО: читаем файл в память ДО выхода из TemporaryDirectory
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
        # Освобождаем память всегда (даже если была ошибка)
        gc.collect()
