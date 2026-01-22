#!/usr/bin/env python3
# fix_retime_zoom_otio.py
# Убирает Dynamic Zoom из клипов с ReTime,
# чтобы retime + optical flow НЕ применяли zoom к изображению.

import json
import sys
from typing import Any, Dict, Tuple


SPATIAL_EFFECT_NAMES = {
    "Dynamic Zoom",
}


def _effect_name(effect: Dict[str, Any]) -> str:
    md = effect.get("metadata") or {}
    ro = md.get("Resolve_OTIO") or {}
    return (ro.get("Effect Name") or ro.get("Name") or "").strip()


def _is_time_effect(effect: Dict[str, Any]) -> bool:
    schema = effect.get("OTIO_SCHEMA") or ""
    return (
        "LinearTimeWarp" in schema
        or "TimeEffect" in schema
        or "FreezeFrame" in schema
    )


def _clip_has_retime(clip: Dict[str, Any]) -> bool:
    for e in clip.get("effects", []) or []:
        if _is_time_effect(e):
            return True
    return False


def _remove_dynamic_zoom(clip: Dict[str, Any]) -> int:
    effects = clip.get("effects") or []
    if not effects:
        return 0

    new_effects = []
    removed = 0

    for e in effects:
        if _effect_name(e) in SPATIAL_EFFECT_NAMES:
            removed += 1
            continue
        new_effects.append(e)

    clip["effects"] = new_effects
    return removed


def _walk(node: Dict[str, Any]) -> Tuple[int, int]:
    retime_clips = 0
    removed_total = 0

    schema = node.get("OTIO_SCHEMA") or ""
    if schema.startswith("Clip."):
        if _clip_has_retime(node):
            retime_clips += 1
            removed_total += _remove_dynamic_zoom(node)

    for key in ("tracks", "children", "clips", "stack", "track"):
        v = node.get(key)
        if isinstance(v, dict):
            a, b = _walk(v)
            retime_clips += a
            removed_total += b
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    a, b = _walk(item)
                    retime_clips += a
                    removed_total += b

    return retime_clips, removed_total


def fix_otio(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    retime_clips, removed = _walk(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("✔ OTIO fixed")
    print(f"Retime clips found: {retime_clips}")
    print(f"Dynamic Zoom removed: {removed}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("python fix_retime_zoom_otio.py input.otio output.otio")
        sys.exit(1)

    fix_otio(sys.argv[1], sys.argv[2])
