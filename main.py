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


def build_cut_times_by_beats(beat_times, music_duration, min_cut, beat_step=1):
    """
    Режем строго по битам (это то, что тебе нужно, чтобы клипы 'легли' на музыку).
    min_cut: убирает слишком частые биты (иначе будет мельтешить).
    beat_step: 1 = каждый бит, 2 = каждый второй, 4 = каждый четвертый.
    """
    beat_step = max(1, int(beat_step))

    cuts = [0.0]
    last = 0.0
    kept = 0

    for t in beat_times:
        t = float(t)
        if t <= last:
            continue
        if t - last < float(min_cut):
            continue

        kept += 1
        if kept % beat_step != 0:
            continue

        cuts.append(t)
        last = t

    if music_duration > cuts[-1] + 0.01:
        cuts.append(float(music_duration))

    # уникальность + сортировка
    cuts = sorted(set(np.round(cuts, 4).tolist()))
    if cuts[0] != 0.0:
        cuts.insert(0, 0.0)
    if cuts[-1] < music_duration - 0.01:
        cuts.append(float(music_duration))

    return cuts


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
    max_cut: float,      # оставляем в сигнатуре для совместимости
    beat_offset: float,
    sr: int,
    analyze_seconds: float,
    beat_step: int
):
    timeline = otio.adapters.read_from_file(input_otio)
    if not isinstance(timeline, otio.schema.Timeline):
        raise RuntimeError("OTIO-файл не является Timeline.")

    video_track_in = find_first_video_track(timeline)
    source_clips = collect_clips(video_track_in)

    beat_times, music_duration, tempo = detect_beats(
        music_mp3, sr=sr, analyze_seconds=analyze_seconds, start_offset=beat_offset
    )

    # ✅ ВАЖНО: теперь режем строго по битам
    cuts = build_cut_times_by_beats(
        beat_times,
        music_duration,
        min_cut=min_cut,
        beat_step=beat_step
    )

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
    ae = settings.get("auto_edit_to_music_1", {}) if isinstance(settings, dict) else {}

    # clip_duration -> min_cut (а max_cut оставим как есть, он больше не правит длины)
    clip_duration = ae.get("clip_duration")
    if isinstance(clip_duration, (list, tuple)) and len(clip_duration) == 2:
        try:
            mn = float(clip_duration[0])
            mx = float(clip_duration[1])
            # min_cut = минимальный шаг, чтобы не резало слишком часто
            if 0.05 <= mn <= mx <= 30.0:
                args.min_cut = mn
        except Exception:
            pass

    # beat_step (если захочешь добавить в settings — поддержим)
    if "beat_step" in ae:
        try:
            args.beat_step = int(ae["beat_step"])
        except Exception:
            pass

    if "beat_offset" in ae:
        try:
            args.beat_offset = float(ae["beat_offset"])
        except Exception:
            pass

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", required=True, help="input .otio from DaVinci")
    ap.add_argument("--music", required=True, help="music .mp3/.wav")
    ap.add_argument("--out", default="output.otio", help="output .otio")
    ap.add_argument("--fps", type=float, default=25.0, help="timeline fps (default: 25)")

    # раньше это управляло длиной сегментов; теперь min_cut = фильтр "слишком часто"
    ap.add_argument("--min_cut", type=float, default=0.18, help="min interval between cuts (sec)")
    ap.add_argument("--max_cut", type=float, default=1.20, help="(legacy) not used in beat-cut mode")

    ap.add_argument("--beat_offset", type=float, default=0.0, help="shift beats in seconds (+/-)")
    ap.add_argument("--sr", type=int, default=11025, help="audio sample rate for analysis (lower = less RAM)")
    ap.add_argument("--analyze_seconds", type=float, default=120.0, help="how many seconds of audio to analyze (RAM saver)")

    # ✅ новое: резать каждый N-й бит
    ap.add_argument("--beat_step", type=int, default=1, help="1=every beat, 2=every 2nd beat, 4=every 4th beat")

    ap.add_argument("--settings", default=None, help="path to settings.json (optional)")
    args = ap.parse_args()

    settings = _load_settings(args.settings)
    _apply_settings_to_args(settings, args)

    if args.min_cut <= 0:
        args.min_cut = 0.18
    if args.analyze_seconds <= 0:
        args.analyze_seconds = 120.0
    if args.sr <= 0:
        args.sr = 11025
    if args.beat_step <= 0:
        args.beat_step = 1

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
        beat_step=args.beat_step,
    )

    print(f"Done. Saved: {args.out}")
    print(f"Audio duration: {dur:.2f}s | Segments: {n_segments} | Tempo≈{tempo:.2f} BPM")
    print(f"Beat mode: step={args.beat_step} | min_cut={args.min_cut:.2f}s")


if __name__ == "__main__":
    main()
