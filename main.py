import argparse
import gc
import json
from pathlib import Path

import numpy as np
import librosa

# --- scipy/librosa compatibility patch ---
import scipy.signal as _sig
if not hasattr(_sig, "hann"):
    from scipy.signal.windows import hann as _hann
    _sig.hann = _hann
# ---------------------------------------

import soundfile as sf
import opentimelineio as otio


def get_audio_duration_sec(audio_path: str) -> float:
    with sf.SoundFile(audio_path) as f:
        return float(len(f) / f.samplerate)


def detect_beats(
    audio_path: str,
    sr: int = 11025,
    analyze_seconds: float = 120.0,
    start_offset: float = 0.0
):
    y, _sr = librosa.load(audio_path, sr=sr, mono=True, duration=float(analyze_seconds))

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr) + float(start_offset)

    beat_times = beat_times[beat_times > 0]
    beat_times = np.unique(np.round(beat_times, 4)).tolist()

    music_duration = get_audio_duration_sec(audio_path)

    del y, beat_frames
    gc.collect()

    return beat_times, float(music_duration), float(tempo)


def build_cut_times(beat_times, music_duration, min_cut, max_cut):
    cuts = [0.0]
    last = 0.0

    for t in beat_times:
        if t <= last:
            continue
        dt = t - last
        if dt < min_cut:
            continue

        if dt > max_cut:
            target = last + max_cut
            candidates = [b for b in beat_times if last < b <= target]
            chosen = candidates[-1] if candidates else target
            cuts.append(float(chosen))
            last = float(chosen)
        else:
            cuts.append(float(t))
            last = float(t)

    if music_duration > cuts[-1] + 0.01:
        cuts.append(float(music_duration))

    cleaned = [cuts[0]]
    for c in cuts[1:]:
        if c - cleaned[-1] >= min_cut:
            cleaned.append(c)

    cleaned = sorted(set(np.round(cleaned, 4).tolist()))
    if cleaned and cleaned[0] != 0.0:
        cleaned.insert(0, 0.0)
    if cleaned and cleaned[-1] < music_duration - 0.01:
        cleaned.append(float(music_duration))

    return cleaned


def find_first_video_track(timeline: otio.schema.Timeline) -> otio.schema.Track:
    video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
    if not video_tracks:
        raise RuntimeError("В OTIO не найдено видеодорожек (TrackKind.Video).")
    return video_tracks[0]


def collect_clips(track: otio.schema.Track):
    clips = [item for item in track if isinstance(item, otio.schema.Clip)]
    if not clips:
        raise RuntimeError("На видеодорожке не найдено клипов (otio.schema.Clip).")
    return clips


def seconds_to_rational_time(sec: float, rate: float) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(sec * rate, rate)


def make_timerange(start_sec: float, dur_sec: float, rate: float) -> otio.opentime.TimeRange:
    return otio.opentime.TimeRange(
        start_time=seconds_to_rational_time(start_sec, rate),
        duration=seconds_to_rational_time(dur_sec, rate),
    )


def require_source_range(clip: otio.schema.Clip):
    if clip.source_range is None:
        raise RuntimeError(
            f"Клип '{clip.name}' не содержит source_range. "
            "Экспортируй OTIO из Resolve так, чтобы диапазоны сохранялись."
        )
    return clip.source_range


def rebuild_timeline(
    input_otio: str,
    music_mp3: str,
    output_otio: str,
    fps: float,
    min_cut: float,
    max_cut: float,
    beat_offset: float,
    sr: int,
    analyze_seconds: float
):
    timeline = otio.adapters.read_from_file(input_otio)
    if not isinstance(timeline, otio.schema.Timeline):
        raise RuntimeError("OTIO-файл не является Timeline.")

    video_track_in = find_first_video_track(timeline)
    source_clips = collect_clips(video_track_in)

    beat_times, music_duration, tempo = detect_beats(
        music_mp3, sr=sr, analyze_seconds=analyze_seconds, start_offset=beat_offset
    )
    cuts = build_cut_times(beat_times, music_duration, min_cut=min_cut, max_cut=max_cut)

    segments = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    segment_durations = [b - a for a, b in segments]

    out_tl = otio.schema.Timeline(name=f"{timeline.name}_beatcut")
    out_stack = otio.schema.Stack()

    out_video = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    out_audio = otio.schema.Track(name="A1_music", kind=otio.schema.TrackKind.Audio)

    music_ref = otio.schema.ExternalReference(target_url=str(Path(music_mp3).resolve().as_posix()))
    music_clip = otio.schema.Clip(name=Path(music_mp3).stem, media_reference=music_ref)
    music_clip.source_range = make_timerange(0.0, music_duration, fps)
    out_audio.append(music_clip)

    src_i = 0
    for seg_dur in segment_durations:
        remaining = float(seg_dur)

        while remaining > 1e-6 and src_i < len(source_clips):
            src_clip = source_clips[src_i]
            src_i += 1

            srng = require_source_range(src_clip)
            sr_start = float(srng.start_time.to_seconds())
            sr_dur = float(srng.duration.to_seconds())
            if sr_dur <= 1e-6:
                continue

            take = min(sr_dur, remaining)

            new_clip = otio.schema.Clip(
                name=src_clip.name,
                media_reference=src_clip.media_reference,
            )
            new_clip.source_range = make_timerange(sr_start, take, fps)

            out_video.append(new_clip)
            remaining -= take

        if src_i >= len(source_clips):
            break

    out_stack.append(out_video)
    out_stack.append(out_audio)
    out_tl.tracks = out_stack

    otio.adapters.write_to_file(out_tl, output_otio)

    del timeline, out_tl, out_stack, out_video, out_audio, source_clips
    gc.collect()

    return tempo, len(cuts) - 1, music_duration


def _load_settings(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _apply_settings_to_args(settings: dict, args: argparse.Namespace) -> None:
    """
    Берём значения из settings.json и аккуратно маппим их в параметры нашего скрипта.
    Ничего не обязано быть — если нет, оставляем дефолты.
    """
    ae = settings.get("auto_edit_to_music_1", {}) if isinstance(settings, dict) else {}
    rt = settings.get("retime_simple", {}) if isinstance(settings, dict) else {}

    # clip_duration -> min_cut / max_cut
    clip_duration = ae.get("clip_duration")
    if isinstance(clip_duration, (list, tuple)) and len(clip_duration) == 2:
        try:
            mn = float(clip_duration[0])
            mx = float(clip_duration[1])
            if 0.05 <= mn <= mx <= 30.0:
                args.min_cut = mn
                args.max_cut = mx
        except Exception:
            pass

    # beat offset
    if "beat_offset" in ae:
        try:
            args.beat_offset = float(ae["beat_offset"])
        except Exception:
            pass

    # sr / analyze_seconds (RAM saver)
    if "sr" in ae:
        try:
            args.sr = int(ae["sr"])
        except Exception:
            pass
    if "analyze_seconds" in ae:
        try:
            args.analyze_seconds = float(ae["analyze_seconds"])
        except Exception:
            pass

    # ⚠️ retime_simple сейчас в этом main.py НЕ реализован.
    # Мы просто принимаем настройки, но не применяем их, чтобы ничего не ломать.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", required=True, help="input .otio from DaVinci")
    ap.add_argument("--music", required=True, help="music .mp3/.wav")
    ap.add_argument("--out", default="output.otio", help="output .otio")
    ap.add_argument("--fps", type=float, default=25.0, help="timeline fps (default: 25)")
    ap.add_argument("--min_cut", type=float, default=0.30, help="min segment duration seconds")
    ap.add_argument("--max_cut", type=float, default=1.20, help="max segment duration seconds")
    ap.add_argument("--beat_offset", type=float, default=0.0, help="shift beats in seconds (+/-)")
    ap.add_argument("--sr", type=int, default=11025, help="audio sample rate for analysis (lower = less RAM)")
    ap.add_argument("--analyze_seconds", type=float, default=120.0, help="how many seconds of audio to analyze (RAM saver)")

    # ✅ новое: путь к settings.json
    ap.add_argument("--settings", default=None, help="path to settings.json (optional)")

    args = ap.parse_args()

    # ✅ применяем settings.json (если передали)
    settings = _load_settings(args.settings)
    _apply_settings_to_args(settings, args)

    # ✅ финальная страховка
    if args.min_cut <= 0 or args.max_cut <= 0 or args.min_cut > args.max_cut:
        raise ValueError("Неверные значения min_cut/max_cut (проверь настройки).")
    if args.analyze_seconds <= 0:
        args.analyze_seconds = 120.0
    if args.sr <= 0:
        args.sr = 11025

    tempo, n_segments, dur = rebuild_timeline(
        input_otio=args.timeline,
        music_mp3=args.music,
        output_otio=args.out,
        fps=args.fps,
        min_cut=args.min_cut,
        max_cut=args.max_cut,
        beat_offset=args.beat_offset,
        sr=args.sr,
        analyze_seconds=args.analyze_seconds,
    )

    print(f"Done. Saved: {args.out}")
    print(f"Audio duration: {dur:.2f}s | Segments: {n_segments} | Tempo≈{tempo:.2f} BPM")


if __name__ == "__main__":
    main()
