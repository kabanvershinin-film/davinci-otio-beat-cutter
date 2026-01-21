import argparse
import gc
import json
from pathlib import Path
import random

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


def _filter_beats_min_gap(beat_times, min_cut):
    """Убираем слишком частые биты (чтобы не мельтешило)."""
    filtered = []
    last = 0.0
    for t in beat_times:
        t = float(t)
        if t <= last:
            continue
        if t - last < float(min_cut):
            continue
        filtered.append(t)
        last = t
    return filtered


def build_cut_times_by_beats(beat_times, music_duration, min_cut, beat_step=1):
    """
    Режем строго по битам, но равномерно: каждый N-й бит.
    (старый режим, оставляем)
    """
    beat_step = max(1, int(beat_step))
    filtered = _filter_beats_min_gap(beat_times, min_cut)

    cuts = [0.0]
    kept = 0
    for t in filtered:
        kept += 1
        if kept % beat_step != 0:
            continue
        cuts.append(float(t))

    if music_duration > cuts[-1] + 0.01:
        cuts.append(float(music_duration))

    cuts = sorted(set(np.round(cuts, 4).tolist()))
    if cuts[0] != 0.0:
        cuts.insert(0, 0.0)
    if cuts[-1] < music_duration - 0.01:
        cuts.append(float(music_duration))
    return cuts


# ---------- NEW: tempo grid (чтобы НЕ плывёт) ----------
def make_tempo_grid(beat_times, music_duration, tempo, grid_offset=0.0):
    """
    Строим ровную сетку по tempo (BPM) и якорим от первого бита.
    Это сильно стабилизирует попадание (в отличие от "сырого" beat_times).
    grid_offset — мелкая подстройка сетки (+/- секунды), если надо (1-2 кадра).
    """
    beat_times = [float(t) for t in beat_times if float(t) > 0]
    beat_times.sort()

    if tempo <= 0 or len(beat_times) < 1:
        # fallback: что есть
        grid = [0.0] + beat_times + [float(music_duration)]
        grid = sorted(set(np.round(grid, 4).tolist()))
        if grid[0] != 0.0:
            grid.insert(0, 0.0)
        if grid[-1] < music_duration - 0.01:
            grid.append(float(music_duration))
        return grid

    spb = 60.0 / float(tempo)  # seconds per beat
    t0 = beat_times[0] + float(grid_offset)

    grid = [0.0]
    t = t0
    # строим сетку до конца трека
    while t < music_duration - 0.01:
        if t > 0:
            grid.append(float(t))
        t += spb
    grid.append(float(music_duration))

    grid = sorted(set(np.round(grid, 4).tolist()))
    if grid[0] != 0.0:
        grid.insert(0, 0.0)
    if grid[-1] < music_duration - 0.01:
        grid.append(float(music_duration))
    return grid


def build_cuts_from_grid(grid_times, min_cut, beat_weights=(55, 30, 12, 3), seed=None):
    """
    Режем по ровной tempo-сетке и выбираем шаг 1..4 бита по весам.
    Это даёт разную длину клипов и сохраняет попадание в темп.
    """
    rng = np.random.default_rng(seed)

    steps = np.array([1, 2, 3, 4], dtype=int)
    w = np.array(beat_weights, dtype=float)
    w = w / w.sum()

    # grid_times содержит 0 и конец
    beats = [float(t) for t in grid_times[1:-1]]  # внутренние биты
    end = float(grid_times[-1])

    cuts = [0.0]
    last = 0.0
    i = 0

    while i < len(beats):
        t = beats[i]
        if t - last >= float(min_cut):
            cuts.append(float(t))
            last = float(t)
        step = int(rng.choice(steps, p=w))
        i += step

    if end > cuts[-1] + 0.01:
        cuts.append(end)

    cuts = sorted(set(np.round(cuts, 4).tolist()))
    if cuts[0] != 0.0:
        cuts.insert(0, 0.0)
    if cuts[-1] < end - 0.01:
        cuts.append(end)
    return cuts
# ------------------------------------------------------


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
    max_cut: float,      # legacy
    beat_offset: float,
    sr: int,
    analyze_seconds: float,
    beat_step: int,
    mode: str,
    beat_weights: tuple[int, int, int, int],
    seed: int | None,
    grid_offset: float,
    # --- retime ---
    retime_enabled: bool = False,
    retime_prob: int = 0,
    # retime factor in percent, where 100 = normal speed, 50 = 0.5x (slow), 150 = 1.5x (fast)
    retime_factor_min: int = 100,
    retime_factor_max: int = 100,
    skip_retime_if_exists: bool = True,
):
    timeline = otio.adapters.read_from_file(input_otio)
    if not isinstance(timeline, otio.schema.Timeline):
        raise RuntimeError("OTIO-файл не является Timeline.")

    video_track_in = find_first_video_track(timeline)
    source_clips = collect_clips(video_track_in)

    beat_times, music_duration, tempo = detect_beats(
        music_mp3, sr=sr, analyze_seconds=analyze_seconds, start_offset=beat_offset
    )

    # ✅ выбор режима
    if mode == "fixed":
        cuts = build_cut_times_by_beats(
            beat_times, music_duration, min_cut=min_cut, beat_step=beat_step
        )
    else:
        # ✅ новый правильный var: сначала строим tempo grid, потом режем по ней
        grid = make_tempo_grid(beat_times, music_duration, tempo, grid_offset=grid_offset)
        cuts = build_cuts_from_grid(
            grid, min_cut=min_cut, beat_weights=beat_weights, seed=seed
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

            # Timeline duration we want to fill for this piece
            take_out = min(sr_dur, remaining)

            # --- retime: keep timeline duration == take_out, but consume source duration = take_out * speed
            time_scalar = 1.0
            take_src = take_out

            if retime_enabled and int(retime_prob) > 0:
                already_timewarped = any(
                    isinstance(eff, otio.schema.LinearTimeWarp) for eff in (src_clip.effects or [])
                )

                if not (skip_retime_if_exists and already_timewarped):
                    # Probability gate
                    if random.randint(1, 100) <= int(retime_prob):
                        # Convert factor range to speed scalars
                        mn = max(10, int(retime_factor_min)) / 100.0
                        mx = max(10, int(retime_factor_max)) / 100.0
                        if mn > mx:
                            mn, mx = mx, mn

                        # Pick desired speed scalar and adapt to source availability.
                        desired = random.uniform(mn, mx)

                        # Constraint: take_src = take_out * time_scalar <= sr_dur
                        max_scalar_for_take = sr_dur / max(take_out, 1e-9)

                        if desired > max_scalar_for_take and max_scalar_for_take < mn:
                            # Not enough source to apply at least mn speed for this take_out.
                            # Reduce the timeline chunk so we can still apply mn.
                            take_out = min(remaining, sr_dur / mn)
                            max_scalar_for_take = sr_dur / max(take_out, 1e-9)

                        time_scalar = float(min(desired, max_scalar_for_take))
                        if time_scalar < 0.05:
                            time_scalar = 1.0

                        take_src = min(sr_dur, take_out * time_scalar)
                        # Keep exact mapping: output duration = source / scalar
                        take_out = take_src / time_scalar

            new_clip = otio.schema.Clip(
                name=src_clip.name,
                media_reference=src_clip.media_reference,
            )
            new_clip.source_range = make_timerange(sr_start, take_src, fps)

            # Apply time effect so that source duration maps to timeline duration
            if abs(time_scalar - 1.0) > 1e-3:
                new_clip.effects.append(
                    otio.schema.LinearTimeWarp(name="LinearTimeWarp", time_scalar=float(time_scalar))
                )

            out_video.append(new_clip)
            remaining -= take_out

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
    rt = settings.get("retime_simple", {}) if isinstance(settings, dict) else {}

    clip_duration = ae.get("clip_duration")
    if isinstance(clip_duration, (list, tuple)) and len(clip_duration) == 2:
        try:
            mn = float(clip_duration[0])
            mx = float(clip_duration[1])
            if 0.05 <= mn <= mx <= 30.0:
                args.min_cut = mn
        except Exception:
            pass

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

    if "beat_weights" in ae and isinstance(ae["beat_weights"], (list, tuple)) and len(ae["beat_weights"]) == 4:
        try:
            args.beat_weights = tuple(int(x) for x in ae["beat_weights"])
        except Exception:
            pass

    if "mode" in ae:
        if str(ae["mode"]).lower() in ("fixed", "var"):
            args.mode = str(ae["mode"]).lower()

    if "grid_offset" in ae:
        try:
            args.grid_offset = float(ae["grid_offset"])
        except Exception:
            pass

    # --- retime_simple ---
    if isinstance(rt, dict):
        if "f_enabled" in rt:
            try:
                args.retime_enabled = bool(rt["f_enabled"])
            except Exception:
                pass

        if "retime_prob" in rt:
            try:
                args.retime_prob = int(rt["retime_prob"])
            except Exception:
                pass

        rf = rt.get("retime_factor")
        if isinstance(rf, (list, tuple)) and len(rf) == 2:
            try:
                args.retime_factor_min = int(rf[0])
                args.retime_factor_max = int(rf[1])
            except Exception:
                pass

        if "skip_exist" in rt:
            try:
                args.skip_retime_if_exists = bool(rt["skip_exist"])
            except Exception:
                pass


def _parse_weights(s: str):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("beat_weights должен быть в формате: a,b,c,d (4 числа)")
    w = tuple(int(x) for x in parts)
    if any(x < 0 for x in w) or sum(w) == 0:
        raise ValueError("beat_weights: числа должны быть >=0 и сумма > 0")
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", required=True, help="input .otio from DaVinci")
    ap.add_argument("--music", required=True, help="music .mp3/.wav")
    ap.add_argument("--out", default="output.otio", help="output .otio")
    ap.add_argument("--fps", type=float, default=25.0, help="timeline fps (default: 25)")

    ap.add_argument("--min_cut", type=float, default=0.18, help="min interval between cuts (sec)")
    ap.add_argument("--max_cut", type=float, default=1.20, help="(legacy) not used")

    ap.add_argument("--beat_offset", type=float, default=0.0, help="shift beat detection in seconds (+/-)")
    ap.add_argument("--sr", type=int, default=11025, help="audio sample rate for analysis (lower = less RAM)")
    ap.add_argument("--analyze_seconds", type=float, default=120.0, help="how many seconds of audio to analyze (RAM saver)")

    # fixed mode
    ap.add_argument("--beat_step", type=int, default=1, help="fixed mode: cut each N-th beat")

    # var mode
    ap.add_argument("--mode", default="var", choices=["fixed", "var"], help="fixed=each N-th beat, var=variable steps")
    ap.add_argument("--beat_weights", default="55,30,12,3", help="var mode weights for steps 1..4 beats")
    ap.add_argument("--seed", type=int, default=7, help="var mode: set seed for repeatable results")

    # ✅ NEW: offset for tempo grid (подгонка на 1–2 кадра)
    ap.add_argument("--grid_offset", type=float, default=0.0, help="var mode: shift tempo grid in seconds (+/-)")

    # --- retime (speed) ---
    ap.add_argument("--retime_enabled", action="store_true", help="enable retime (speed changes)")
    ap.add_argument("--retime_prob", type=int, default=0, help="retime probability per clip (0..100)")
    ap.add_argument(
        "--retime_factor_min",
        type=int,
        default=100,
        help="min retime factor percent (100=normal, 50=0.5x slow, 150=1.5x fast)",
    )
    ap.add_argument(
        "--retime_factor_max",
        type=int,
        default=100,
        help="max retime factor percent (100=normal, 50=0.5x slow, 150=1.5x fast)",
    )
    ap.add_argument(
        "--skip_retime_if_exists",
        action="store_true",
        help="skip clips that already have LinearTimeWarp in OTIO",
    )

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

    beat_weights = _parse_weights(args.beat_weights)

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
        mode=args.mode,
        beat_weights=beat_weights,
        seed=args.seed,
        grid_offset=args.grid_offset,
        retime_enabled=bool(getattr(args, "retime_enabled", False)),
        retime_prob=int(getattr(args, "retime_prob", 0)),
        retime_factor_min=int(getattr(args, "retime_factor_min", 100)),
        retime_factor_max=int(getattr(args, "retime_factor_max", 100)),
        skip_retime_if_exists=bool(getattr(args, "skip_retime_if_exists", True)),
    )

    print(f"Done. Saved: {args.out}")
    print(f"Audio duration: {dur:.2f}s | Segments: {n_segments} | Tempo≈{tempo:.2f} BPM")
    print(f"Mode: {args.mode} | min_cut={args.min_cut:.2f}s | weights={beat_weights} | seed={args.seed} | grid_offset={args.grid_offset}")


if __name__ == "__main__":
    main()
