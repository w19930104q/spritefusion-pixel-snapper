#!/usr/bin/env python3
"""Prototype for deterministic grid-pitch scoring.

Mirrors the Rust pipeline's profile math ([-1,0,1] gradient projections,
transparent pixels treated as gray 0) and evaluates an explicit,
fully deterministic candidate scoring + harmonic pruning design.
"""
import hashlib
import json
import math
import os
import sys

import numpy as np
from PIL import Image

RNG = np.random.default_rng(20261118)


# ---------------------------------------------------------------- synthetic
def make_grid(cells_x=16, cells_y=12, pitch=8, mode="sharp", seed=0):
    rng = np.random.default_rng(seed)
    w, h = cells_x * pitch, cells_y * pitch
    img = np.zeros((h, w, 4), dtype=np.uint8)

    if mode == "low_contrast":
        base = rng.integers(90, 120, size=(cells_y, cells_x, 3))
        cell_colors = np.clip(base + rng.integers(-12, 13, size=base.shape), 0, 255)
    else:
        cell_colors = rng.integers(0, 256, size=(cells_y, cells_x, 3))

    if mode == "diagonal45":
        # diagonal stripes: structure at 45 deg, but cells stay grid aligned
        for j in range(cells_y):
            for i in range(cells_x):
                cell_colors[j, i] = (40 if (i + j) % 2 == 0 else 220,
                                     40 if (i + j) % 2 == 0 else 220,
                                     90 if (i + j) % 2 == 0 else 200)

    for j in range(cells_y):
        for i in range(cells_x):
            y0, x0 = j * pitch, i * pitch
            alpha = 255
            if mode == "transparent_edges" and rng.random() < 0.4:
                continue  # fully transparent cell
            img[y0:y0 + pitch, x0:x0 + pitch, :3] = cell_colors[j, i]
            img[y0:y0 + pitch, x0:x0 + pitch, 3] = alpha

    if mode == "transparent_edges":
        # semi-transparent fringe on some opaque cell borders
        fringe = rng.random((cells_y, cells_x)) < 0.3
        for j in range(cells_y):
            for i in range(cells_x):
                if fringe[j, i] and img[j * pitch, i * pitch, 3] != 0:
                    img[j * pitch, i * pitch:i * pitch + pitch, 3] = 128

    if mode == "noisy":
        # per-pixel salt noise inside cells, edges stay sharp
        mask = rng.random((h, w)) < 0.08
        img[mask, :3] = rng.integers(0, 256, size=(mask.sum(), 3))
        img[mask, 3] = 255

    if mode == "blurred":
        # gaussian-like blur via repeated 3x3 box blur, pure numpy
        work = img.astype(np.float64)
        for _ in range(3):
            padded = np.pad(work, ((1, 1), (1, 1), (0, 0)), mode="edge")
            work = (
                padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
                + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
                + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
            ) / 9.0
        img = np.clip(work.round(), 0, 255).astype(np.uint8)

    return Image.fromarray(img, "RGBA")


def make_nogrid(kind="smooth", size=128):
    rng = np.random.default_rng(7)
    if kind == "smooth":
        # horizontal gradient + soft circle: no periodic edges
        yy, xx = np.mgrid[0:size, 0:size]
        gray = (xx * 255 // size).astype(np.float32)
        circle = ((xx - 64) ** 2 + (yy - 64) ** 2) < 30 ** 2
        gray[circle] = np.clip(gray[circle] - 80, 0, 255)
        img = np.stack([gray, gray, gray, np.full_like(gray, 255)], axis=-1)
        return Image.fromarray(img.astype(np.uint8), "RGBA")
    # independent per-pixel noise: no periodicity
    img = np.zeros((size, size, 4), dtype=np.uint8)
    img[..., :3] = rng.integers(0, 256, size=(size, size, 3))
    img[..., 3] = 255
    return Image.fromarray(img, "RGBA")


# ---------------------------------------------------------------- profiling
def profiles(img):
    a = np.asarray(img.convert("RGBA"), dtype=np.float64)
    alpha = a[..., 3]
    gray = np.where(alpha == 0, 0.0,
                    0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])
    h, w = gray.shape
    col = np.zeros(w)
    row = np.zeros(h)
    col[1:w - 1] = np.abs(gray[:, 2:w] - gray[:, 0:w - 2]).sum(axis=0)
    row[1:h - 1] = np.abs(gray[2:h, :] - gray[0:h - 2, :]).sum(axis=1)
    return col, row


# ---------------------------------------------------------------- scoring
def axis_scores(profile):
    """Deterministic score for every integer pitch d in [2, n//2].

    Primary score: normalized autocorrelation at lag d (periodicity).
    A harmonic-decay term rewards the base pitch, whose ACF stays high across
    multiples while the profile's overall correlation decays.
    """
    n = len(profile)
    p = np.asarray(profile, dtype=np.float64)
    total = p.sum()
    max_p = p.max()
    out = {}
    if total <= 0 or max_p <= 0:
        return out, {"mean_over_max": 0.0, "peak_fraction": 0.0, "energy": 0.0}
    mean = p.mean()
    z = p - mean
    denom = float(np.dot(z, z))

    def acf(d):
        return max(0.0, float(np.dot(z[:n - d], z[d:])) / denom) if denom > 0 else 0.0

    base = max(acf(1), 1e-12)
    min_pitch = max(4, n // 64)
    for d in range(2, n // 2 + 1):
        periods = n // d
        if periods < 4 or d < min_pitch:  # >=4 periods, ignore short-range ramps
            continue
        primary = acf(d)
        # base-pitch bonus: 2d correlation nearly as high as d => d is base
        decay = min(1.0, primary / max(acf(2 * d), 1e-12)) if 2 * d < n else 1.0
        # smooth ramps inflate lags close to min_pitch via lag-1 correlation
        short_range = min(1.0, primary / base) if d < min_pitch * 2 else 0.0
        out[d] = primary * (0.75 + 0.25 * decay) * (1.0 - 0.7 * short_range)
    stats = {
        "mean_over_max": float(mean / max_p),
        "peak_fraction": float((p > 0.2 * max_p).sum() / n),
        "energy": float(total),
    }
    return out, stats


def combine(sx, sy):
    keys = set(sx) & set(sy)
    return {d: math.sqrt(sx[d] * sy[d]) for d in keys}


def prune_harmonics(scored, tol=0.025, drop_ratio=0.25):
    """Deterministic harmonic/sub-harmonic pruning.

    Pitches are grouped by harmonic/near-equivalent relation. The base
    (smallest) pitch in a group is preferred unless another member beats it
    decisively (>drop_ratio relative), in which case that member wins.
    Ties resolve to the smaller pitch.
    """
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    groups: list[list[tuple[int, float]]] = []

    def related(p, q):
        lo, hi = min(p, q), max(p, q)
        return (hi % lo == 0 and (hi // lo) <= 4) or abs(hi - lo) / lo <= tol

    for pitch, score in ranked:
        for group in groups:
            if any(related(pitch, gp) for gp, _ in group):
                group.append((pitch, score))
                break
        else:
            groups.append([(pitch, score)])

    representatives = []
    for group in groups:
        base_pitch = min(p for p, _ in group)
        base_score = dict(group)[base_pitch]
        winner = (base_pitch, base_score)
        for pitch, score in group:
            if score > base_score * (1.0 + drop_ratio) and score > winner[1]:
                winner = (pitch, score)
        representatives.append(winner)
    representatives.sort(key=lambda kv: (-kv[1], kv[0]))
    return representatives


def analyze(img):
    col, row = profiles(img)
    sx, stx = axis_scores(col)
    sy, sty = axis_scores(row)
    combined = combine(sx, sy)
    ranked = prune_harmonics(combined)
    signal = math.sqrt(max(stx["mean_over_max"], 0) * max(sty["mean_over_max"], 0))
    peak = math.sqrt(stx["peak_fraction"] * sty["peak_fraction"])
    best_d, best_s = ranked[0]
    second_s = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = (best_s - second_s) / best_s if best_s > 0 else 0.0
    confidence = min(1.0, 0.55 * min(1.0, best_s / 0.45)
                     + 0.30 * min(1.0, margin / 0.25)
                     + 0.15 * min(1.0, signal / 0.12))
    # gate: a real grid has a strong, non-ambiguous periodic peak well above
    # the noise floor; smooth ramps and random textures fail here
    no_grid = signal < 0.03 or best_s < 0.25 or margin < 0.10
    return {
        "ranked": ranked[:6],
        "best": best_d,
        "best_score": best_s,
        "margin": margin,
        "signal": signal,
        "peak": peak,
        "confidence": 0.0 if no_grid else confidence,
        "no_grid": no_grid,
        "stats_x": stx,
        "stats_y": sty,
    }


def main():
    out_dir = "/tmp/ps-synth"
    os.makedirs(out_dir, exist_ok=True)
    cases = [
        ("grid_sharp", make_grid(mode="sharp", seed=1), 8),
        ("grid_blurred", make_grid(mode="blurred", seed=2), 8),
        ("grid_low_contrast", make_grid(mode="low_contrast", seed=3), 8),
        ("grid_transparent_edges", make_grid(mode="transparent_edges", seed=4), 8),
        ("grid_diagonal45", make_grid(mode="diagonal45", seed=5), 8),
        ("grid_noisy", make_grid(mode="noisy", seed=6), 8),
        ("nogrid_smooth", make_nogrid("smooth"), None),
        ("nogrid_noise", make_nogrid("noise"), None),
    ]
    report = {}
    for name, img, truth in cases:
        path = os.path.join(out_dir, name + ".png")
        img.save(path)
        r = analyze(img)
        r["truth"] = truth
        r["sha256_input"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        report[name] = r
    for name in ("repo_od", "hero", "details"):
        img = Image.open(f"/home/liiing/feedback/uid21717/static/{name}.png")
        r = analyze(img)
        r["truth"] = None
        report[name] = r
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
