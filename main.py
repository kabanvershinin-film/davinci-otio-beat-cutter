import sys
print("PYTHON VERSION:", sys.version)

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>DaVinci OTIO Beat Cutter</h1>
    <p>Server is running ✅</p>
    """

import argparse
from pathlib import Path
import numpy as np
import librosa
# --- scipy/librosa compatibility patch ---
import scipy.signal as _sig
if not hasattr(_sig, "hann"):
    from scipy.signal.windows import hann as _hann
    _sig.hann = _hann
# ---------------------------------------

import opentimelineio as otio


def detect_beats(audio_path: str, sr: int = 22050, start_offset: float = 0.0):
    # Возвращает массив времени битов в секундах
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_times = beat_times + float(start_offset)
    # Убираем отрицательные/дубли
    beat_times = beat_times[beat_times > 0]
    beat_times = np.unique(np.round(beat_times, 4))
    return beat_times.tolist(), float(librosa.get_duration(y=y, sr=sr))


def build_cut_times(beat_times, music_duration, min_cut, max_cut):
    """
    Формируем точки резки (в секундах), используя биты.
    Правило:
      - пропускаем слишком частые биты (< min_cut)
      - если между резками очень долго (> max_cut), всё равно режем на ближайшем доступном бите (если есть),
        иначе режем по max_cut (редкий кейс)
    """
    cuts = [0.0]
    last = 0.0

    for t in beat_times:
        if t <= last:
            continue
        dt = t - last
        if dt < min_cut:
            continue

        if dt > max_cut:
            # Попробуем найти ближайший бит <= last + max_cut
            target = last + max_cut
            candidates = [b for b in beat_times if last < b <= target]
            if candidates:
                chosen = candidates[-1]
                cuts.append(chosen)
                last = chosen
            else:
                # крайний случай — режем не по биту
                cuts.append(target)
                last = target
        else:
            cuts.append(t)
            last = t

    # Финальная точка — конец музыки (можно не резать строго по биту)
    if music_duration > cuts[-1] + 0.01:
        cuts.append(music_duration)

    # Уберём слишком короткие хвосты
    cleaned = [cuts[0]]
    for c in cuts[1:]:
        if c - cleaned[-1] >= min_cut:
            cleaned.append(c)
    return cleaned


def find_first_timeline_and_video_track(timeline: otio.schema.Timeline):
    video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
    if not video_tracks:
        raise RuntimeError("В OTIO не найдено видеодорожек (TrackKind.Video).")
    return video_tracks[0]


def collect_clips_in_order(video_track: otio.schema.Track):
    clips = []
    for item in video_track:
        if isinstance(item, otio.schema.Clip):
            clips.append(item)
    if not clips:
        raise RuntimeError("На видеодорожке не найдено клипов (otio.schema.Clip).")
    return clips


def seconds_to_rational_time(sec: float, rate: float) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(sec * rate, rate)


def make_timerange(start_sec: float, dur_sec: float, rate: float) -> otio.opentime.TimeRange:
    return otio.opentime.TimeRange(
        start_time=seconds_to_rational_time(start_sec, rate),
        duration=seconds_to_rational_time(dur_sec, rate)
    )


def get_source_range_or_default(clip: otio.schema.Clip, rate: float):
    # В OTIO у клипа может не быть source_range — тогда считаем, что доступно "всё".
    # Но "всё" без метаданных не узнать. В этом MVP предполагаем, что source_range есть.
    if clip.source_range is None:
        raise RuntimeError(
            f"Клип '{clip.name}' не содержит source_range. "
            "Экспортируй OTIO из Resolve так, чтобы диапазоны сохранялись."
        )
    return clip.source_range


def rebuild_timeline(input_otio: str, music_mp3: str, output_otio: str,
                     fps: float, min_cut: float, max_cut: float, beat_offset: float):
    timeline = otio.adapters.read_from_file(input_otio)
    if not isinstance(timeline, otio.schema.Timeline):
        raise RuntimeError("OTIO-файл не является Timeline.")

    video_track_in = find_first_timeline_and_video_track(timeline)
    source_clips = collect_clips_in_order(video_track_in)

    beat_times, music_duration = detect_beats(music_mp3, start_offset=beat_offset)
    cuts = build_cut_times(beat_times, music_duration, min_cut=min_cut, max_cut=max_cut)

    # Сегменты, под которые будем резать видео
    segments = [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1)]
    segment_durations = [b - a for a, b in segments]

    out_tl = otio.schema.Timeline(name=f"{timeline.name}_beatcut")
    out_tracks = otio.schema.Stack()

    out_video = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    out_audio = otio.schema.Track(name="A1_music", kind=otio.schema.TrackKind.Audio)

    # Аудио клип (музыка)
    music_ref = otio.schema.ExternalReference(target_url=str(Path(music_mp3).resolve().as_posix()))
    music_clip = otio.schema.Clip(name=Path(music_mp3).stem, media_reference=music_ref)
    music_clip.source_range = make_timerange(0.0, music_duration, fps)
    out_audio.append(music_clip)

    # Нарезка видео
    src_i = 0
    src_pos_sec = 0.0  # позиция внутри текущего source_clip (относительно его source_range start)

    for dur in segment_durations:
        if src_i >= len(source_clips):
            break

        remaining = dur
        # Один сегмент может "съесть" несколько исходных клипов, если они короткие
        while remaining > 1e-6:
            if src_i >= len(source_clips):
                remaining = 0
                break

            src_clip = source_clips[src_i]
            sr = get_source_range_or_default(src_clip, fps)
            sr_start = sr.start_time.to_seconds()
            sr_dur = sr.duration.to_seconds()

            available = sr_dur - src_pos_sec
            if available <= 1e-6:
                src_i += 1
                src_pos_sec = 0.0
                continue

            take = min(available, remaining)

            new_clip = otio.schema.Clip(
                name=src_clip.name,
                media_reference=src_clip.media_reference
            )
            # берём кусок из исходника
            new_clip.source_range = make_timerange(
                start_sec=sr_start + src_pos_sec,
                dur_sec=take,
                rate=fps
            )

            out_video.append(new_clip)

            src_pos_sec += take
            remaining -= take

            # если текущий клип закончился — идём к следующему
            if src_pos_sec >= sr_dur - 1e-6:
                src_i += 1
                src_pos_sec = 0.0

    out_tracks.append(out_video)
    out_tracks.append(out_audio)
    out_tl.tracks = out_tracks

    otio.adapters.write_to_file(out_tl, output_otio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", required=True, help="input .otio from DaVinci")
    ap.add_argument("--music", required=True, help="music .mp3")
    ap.add_argument("--out", default="output.otio", help="output .otio")
    ap.add_argument("--fps", type=float, default=25.0, help="timeline fps (default: 25)")
    ap.add_argument("--min_cut", type=float, default=0.30, help="min segment duration seconds")
    ap.add_argument("--max_cut", type=float, default=1.20, help="max segment duration seconds")
    ap.add_argument("--beat_offset", type=float, default=0.0, help="shift beats in seconds (+/-)")
    args = ap.parse_args()

    rebuild_timeline(
        input_otio=args.timeline,
        music_mp3=args.music,
        output_otio=args.out,
        fps=args.fps,
        min_cut=args.min_cut,
        max_cut=args.max_cut,
        beat_offset=args.beat_offset
    )
    print(f"Done. Saved: {args.out}")


if __name__ == "__main__":
    main()
