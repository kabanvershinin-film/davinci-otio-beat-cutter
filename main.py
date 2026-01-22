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


# ---------- tempo grid (чтобы НЕ плывёт) ----------
def make_tempo_grid(beat_times, music_duration, tempo, grid_offset=0.0):
    """
    Строим ровную сетку по tempo (BPM) и якорим от первого бита.
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
    """
    rng = np.random.default_rng(seed)

    steps = np.array([1, 2, 3, 4], dtype=int)
    w = np.array(beat_weights, dtype=float)
    w = w / w.sum()

    beats = [float(t) for t in grid_times[1:-1]]
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


# ---------- NEW: transient peaks + snap ----------
def detect_transient_peaks(
    audio_path: str,
    sr: int,
    start_offset: float = 0.0,
    hop_length: int = 512,
    use_hpss_percussive: bool = True,
):
    """
    Возвращает:
      peak_times_sec: np.array shape (N,)
      peak_strengths: np.array shape (N,)
    Детектим пики транзиентов (onset) по перкуссионной составляющей.
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    if use_hpss_percussive:
        # перкуссия лучше соответствует "пикам" на waveform (удары)
        _, y = librosa.effects.hpss(y)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # peak picking: достаточно устойчивые настройки под разные треки
    peaks = librosa.util.peak_pick(
        onset_env,
        pre_max=3, post_max=3,
        pre_avg=6, post_avg=6,
        delta=0.2,
        wait=3
    )

    peak_strengths = onset_env[peaks] if len(peaks) else np.array([], dtype=float)
    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length) + float(start_offset)

    # чистим
    peak_times = peak_times[peak_times > 0]
    if len(peak_times):
        # синхронизируем strengths по тем же индексам
        # (peaks уже по onset_env; frame->time не меняет длину)
        pass

    del y, onset_env, peaks
    gc.collect()

    return np.asarray(peak_times, dtype=float), np.asarray(peak_strengths, dtype=float)


def snap_cuts_to_peaks(
    cuts: list[float],
    peak_times: np.ndarray,
    peak_strengths: np.ndarray,
    win_before: float,
    win_after: float,
    strength_percentile: float = 65.0,
    keep_min_gap: float | None = None
):
    """
    Сдвигаем каждый рез (кроме 0 и конца) к самому сильному пику в окне:
      [cut - win_before, cut + win_after]
    Если пики слабые — не двигаем (порог по перцентилю силы).
    """
    if not cuts or len(cuts) < 3:
        return cuts

    if peak_times is None or len(peak_times) == 0:
        return cuts

    win_before = max(0.0, float(win_before))
    win_after = max(0.0, float(win_after))
    if win_before == 0.0 and win_after == 0.0:
        return cuts

    # порог силы (чтобы не липло к шуму)
    thr = None
    if peak_strengths is not None and len(peak_strengths) == len(peak_times) and len(peak_strengths) > 0:
        p = float(np.clip(strength_percentile, 0.0, 100.0))
        thr = float(np.percentile(peak_strengths, p))
    else:
        thr = None  # нет сил — снапаем без порога

    snapped = [float(cuts[0])]
    end = float(cuts[-1])

    # для быстрого поиска по окну
    pt = peak_times
    ps = peak_strengths if (peak_strengths is not None and len(peak_strengths) == len(pt)) else None

    for c in cuts[1:-1]:
        c = float(c)
        lo = c - win_before
        hi = c + win_after

        # индексы пиков в окне
        i0 = int(np.searchsorted(pt, lo, side="left"))
        i1 = int(np.searchsorted(pt, hi, side="right"))

        if i1 <= i0:
            snapped.append(c)
            continue

        window_times = pt[i0:i1]
        if ps is not None:
            window_strengths = ps[i0:i1]
            j = int(np.argmax(window_strengths))
            best_t = float(window_times[j])
            best_s = float(window_strengths[j])
            if thr is None or best_s >= thr:
                snapped.append(best_t)
            else:
                snapped.append(c)
        else:
            # нет strengths — берём ближайший по времени
            j = int(np.argmin(np.abs(window_times - c)))
            snapped.append(float(window_times[j]))

    snapped.append(end)

    # нормализуем, сортируем, убираем дубли
    snapped = sorted(set(np.round(snapped, 4).tolist()))

    # защита от слишком близких резов после снапа
    if keep_min_gap is not None and keep_min_gap > 0 and len(snapped) >= 3:
        filtered = [snapped[0]]
        last = snapped[0]
        for t in snapped[1:-1]:
            if float(t) - float(last) >= float(keep_min_gap):
                filtered.append(float(t))
                last = float(t)
        filtered.append(snapped[-1])
        snapped = filtered

    # гарантируем 0 и end
    if snapped[0] != 0.0:
        snapped.insert(0, 0.0)
    if snapped[-1] < end - 0.01:
        snapped.append(end)

    return snapped
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
    transient_before: float,
    transient_after: float,
    transient_strength_percentile: float
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
        grid = make_tempo_grid(beat_times, music_duration, tempo, grid_offset=grid_offset)
        cuts = build_cuts_from_grid(
            grid, min_cut=min_cut, beat_weights=beat_weights, seed=seed
        )

    # ✅ NEW: snap cuts to transient peaks (если окно задано)
    if (transient_before > 0.0 or transient_after > 0.0) and len(cuts) >= 3:
        peak_times, peak_strengths = detect_transient_peaks(
            music_mp3, sr=sr, start_offset=beat_offset, hop_length=512, use_hpss_percussive=True
        )
        # keep_min_gap чуть меньше min_cut, чтобы снап не рушил ритм
        keep_gap = max(0.04, float(min_cut) * 0.6)
        cuts = snap_cuts_to_peaks(
            cuts,
            peak_times=peak_times,
            peak_strengths=peak_strengths,
            win_before=transient_before,
            win_after=transient_after,
            strength_percentile=transient_strength_percentile,
            keep_min_gap=keep_gap
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

    clip_duration = ae.get("clip_duration")
    if isinstance(clip_duration, (list, tuple)) and len(clip_duration) == 2:
        try:
            mn = float(clip_duration[0])
            mx = float(clip_duration[1])
            if 0.05 <= mn <= mx <= 30.0:
                args.min_cut = mn
        except Exception:
            pass

    # ✅ NEW: transient window from UI: [ct_min, ct_max]  (оба положительные)
    clip_transient = ae.get("clip_transient")
    if isinstance(clip_transient, (list, tuple)) and len(clip_transient) == 2:
        try:
            tmin = float(clip_transient[0])
            tmax = float(clip_transient[1])
            # адекватные клампы, чтобы не улетало
            tmin = float(np.clip(tmin, 0.0, 1.0))
            tmax = float(np.clip(tmax, 0.0, 1.5))
            args.transient_before = tmin
            args.transient_after = tmax
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

    # опционально: порог силы (если захочешь менять из UI потом)
    if "transient_strength_percentile" in ae:
        try:
            args.transient_strength_percentile = float(ae["transient_strength_percentile"])
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

    # tempo grid offset
    ap.add_argument("--grid_offset", type=float, default=0.0, help="var mode: shift tempo grid in seconds (+/-)")

    # ✅ NEW: transient snap window (seconds)
    ap.add_argument("--transient_before", type=float, default=0.08, help="snap window BEFORE cut (sec)")
    ap.add_argument("--transient_after", type=float, default=0.12, help="snap window AFTER cut (sec)")
    ap.add_argument("--transient_strength_percentile", type=float, default=65.0,
                    help="snap only to peaks above this strength percentile (0..100)")

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

    # клампы окон снапа
    args.transient_before = float(np.clip(args.transient_before, 0.0, 1.0))
    args.transient_after = float(np.clip(args.transient_after, 0.0, 1.5))
    args.transient_strength_percentile = float(np.clip(args.transient_strength_percentile, 0.0, 100.0))

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
        transient_before=args.transient_before,
        transient_after=args.transient_after,
        transient_strength_percentile=args.transient_strength_percentile
    )

    print(f"Done. Saved: {args.out}")
    print(f"Audio duration: {dur:.2f}s | Segments: {n_segments} | Tempo≈{tempo:.2f} BPM")
    print(f"Mode: {args.mode} | min_cut={args.min_cut:.2f}s | weights={beat_weights} | seed={args.seed} | grid_offset={args.grid_offset}")
    print(f"Transient snap: before={args.transient_before:.3f}s after={args.transient_after:.3f}s thrP={args.transient_strength_percentile:.1f}")


if __name__ == "__main__":
    main()
