# DaVinci OTIO Beat Cutter

Автоматически пересобирает таймлайн DaVinci (OTIO) под биты музыкального трека.

## Вход
- timeline.otio — экспорт из DaVinci Resolve
- music.mp3 — отдельный трек

## Выход
- output.otio — новый таймлайн, где склейки совпадают с битами, клипы идут по порядку, хвосты не используются.

## Установка
pip install -r requirements.txt

## Запуск
python main.py --timeline input.otio --music track.mp3 --out output.otio --fps 25

Опции:
--min_cut 0.30
--max_cut 1.20
--beat_offset 0.0
