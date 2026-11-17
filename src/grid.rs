//! Deterministic grid-pitch detection, candidate pruning and cut transforms.
//!
//! The pipeline in this module is fully deterministic: every step uses fixed
//! integer arithmetic orderings and explicit tie-breaking rules, so the same
//! input always yields the same ranked candidates, confidence and output
//! transform. No randomness or map iteration order influences the result.

use image::RgbaImage;

/// Tunable detection thresholds. Centralized so the scoring behavior is
/// auditable in one place and cannot drift between call sites.
#[derive(Debug, Clone)]
pub struct DetectionParams {
    /// Minimum number of pitch periods required to claim a grid.
    pub min_periods: usize,
    /// Smallest considered pitch as a fraction of the axis length.
    pub min_pitch_divisor: usize,
    /// Absolute floor for the smallest considered pitch.
    pub min_pitch_floor: usize,
    /// Hard gate: combined score below this means "no grid".
    pub score_gate: f64,
    /// Hard gate: winning margin below this means "ambiguous, no grid".
    pub margin_gate: f64,
    /// Hard gate: profile signal (mean/max) below this means "no structure".
    pub signal_gate: f64,
    /// Relative score gap that lets a harmonic replace its base pitch.
    pub harmonic_drop: f64,
    /// Relative distance treating two pitches as near-equivalent.
    pub equivalent_tol: f64,
    /// Number of candidates retained after pruning.
    pub max_candidates: usize,
    /// Fraction of a cell searched when snapping uniform cuts.
    pub cut_search_window_ratio: f64,
    /// Minimum cut search window in pixels.
    pub cut_min_search_window: f64,
    /// Cut snap accepted only above mean profile strength times this.
    pub cut_strength_threshold: f64,
}

impl Default for DetectionParams {
    fn default() -> Self {
        Self {
            min_periods: 4,
            min_pitch_divisor: 64,
            min_pitch_floor: 4,
            score_gate: 0.25,
            margin_gate: 0.10,
            signal_gate: 0.03,
            harmonic_drop: 0.25,
            equivalent_tol: 0.025,
            max_candidates: 8,
            cut_search_window_ratio: 0.35,
            cut_min_search_window: 2.0,
            cut_strength_threshold: 0.5,
        }
    }
}

/// Outcome of grid detection for one image.
#[derive(Debug, Clone, PartialEq)]
pub enum GridVerdict {
    /// A credible grid was found.
    Detected,
    /// No periodic grid could be established; the transform is identity.
    NoGrid,
    /// The pitch was supplied by the caller; scoring is skipped.
    Override,
}

/// One ranked pitch candidate with its deterministic score components.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub pitch: usize,
    pub score: f64,
    pub score_x: f64,
    pub score_y: f64,
}

/// Axis-level profile statistics used for confidence and no-grid gating.
#[derive(Debug, Clone, Default)]
pub struct AxisStats {
    pub mean_over_max: f64,
    pub peak_fraction: f64,
    pub energy: f64,
}

/// Full, machine-readable detection report for a single image.
#[derive(Debug, Clone)]
pub struct GridReport {
    pub verdict: GridVerdict,
    /// Winning pitch in input pixels (square grid); 0 when no grid.
    pub pitch: usize,
    /// Winning pitch as a float (mirrors legacy `pixel_size` reporting).
    pub pixel_size: f64,
    /// Confidence in `[0, 1]`; exactly 0 for `NoGrid`.
    pub confidence: f64,
    pub signal: f64,
    pub margin: f64,
    pub best_score: f64,
    pub candidates: Vec<Candidate>,
    pub stats_x: AxisStats,
    pub stats_y: AxisStats,
    pub output_width: u32,
    pub output_height: u32,
}

impl GridReport {
    /// Output-to-input dimension ratio for the winning transform.
    pub fn scale_x(&self, input_width: u32) -> f64 {
        if input_width == 0 {
            0.0
        } else {
            self.output_width as f64 / input_width as f64
        }
    }

    pub fn scale_y(&self, input_height: u32) -> f64 {
        if input_height == 0 {
            0.0
        } else {
            self.output_height as f64 / input_height as f64
        }
    }
}

/// Compute gradient projections along each axis.
///
/// Mirrors the original implementation: a `[-1, 0, 1]` kernel on luma, with
/// fully transparent pixels treated as luma 0. The column profile accumulates
/// horizontal gradients (edges aligned to column boundaries), the row profile
/// vertical ones.
pub fn compute_profiles(img: &RgbaImage) -> (Vec<f64>, Vec<f64>) {
    let (w, h) = img.dimensions();
    let mut col_proj = vec![0.0f64; w as usize];
    let mut row_proj = vec![0.0f64; h as usize];

    let luma = |x: u32, y: u32| -> f64 {
        let p = img.get_pixel(x, y);
        if p[3] == 0 {
            0.0
        } else {
            0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64
        }
    };

    for y in 0..h {
        for x in 1..w.saturating_sub(1) {
            let grad = (luma(x + 1, y) - luma(x - 1, y)).abs();
            col_proj[x as usize] += grad;
        }
    }
    for x in 0..w {
        for y in 1..h.saturating_sub(1) {
            let grad = (luma(x, y + 1) - luma(x, y - 1)).abs();
            row_proj[y as usize] += grad;
        }
    }

    (col_proj, row_proj)
}

fn axis_stats(profile: &[f64]) -> AxisStats {
    let n = profile.len();
    if n == 0 {
        return AxisStats::default();
    }
    let mut max_p = 0.0f64;
    let mut sum = 0.0f64;
    for &v in profile {
        if v > max_p {
            max_p = v;
        }
        sum += v;
    }
    if max_p <= 0.0 {
        return AxisStats::default();
    }
    let mean = sum / n as f64;
    let threshold = 0.2 * max_p;
    let peaks = profile.iter().filter(|&&v| v > threshold).count();
    AxisStats {
        mean_over_max: mean / max_p,
        peak_fraction: peaks as f64 / n as f64,
        energy: sum,
    }
}

/// Score every eligible integer pitch on one axis.
///
/// The primary signal is the normalized autocorrelation of the mean-subtracted
/// profile at lag `d` — a pitch-unbiased measure of periodicity. A
/// harmonic-decay term rewards the base pitch (its double-lag correlation stays
/// nearly as high), and short lags inheriting the lag-1 correlation of smooth
/// ramps are penalized.
fn score_axis(profile: &[f64], params: &DetectionParams) -> std::collections::BTreeMap<usize, f64> {
    let n = profile.len();
    let mut scores = std::collections::BTreeMap::new();
    if n < 8 {
        return scores;
    }

    let stats = axis_stats(profile);
    if stats.energy <= 0.0 {
        return scores;
    }

    let mean = stats.energy / n as f64;
    let centered: Vec<f64> = profile.iter().map(|v| v - mean).collect();
    let denom: f64 = centered.iter().map(|z| z * z).sum();
    if denom <= 0.0 {
        return scores;
    }

    let autocorr = |d: usize| -> f64 {
        if d == 0 || d >= n {
            return 0.0;
        }
        let num: f64 = centered[..n - d]
            .iter()
            .zip(&centered[d..])
            .map(|(a, b)| a * b)
            .sum();
        (num / denom).max(0.0)
    };

    let lag1 = autocorr(1).max(1e-12);
    let min_pitch = (n / params.min_pitch_divisor)
        .max(params.min_pitch_floor)
        .max(2);

    for d in min_pitch..=n / 2 {
        if n / d < params.min_periods {
            continue;
        }
        let primary = autocorr(d);
        let decay = if 2 * d < n {
            (primary / autocorr(2 * d).max(1e-12)).min(1.0)
        } else {
            1.0
        };
        let short_range = if d < min_pitch * 2 {
            (primary / lag1).min(1.0)
        } else {
            0.0
        };
        let score = primary * (0.75 + 0.25 * decay) * (1.0 - 0.7 * short_range);
        scores.insert(d, score);
    }
    scores
}

/// Deterministic harmonic / sub-harmonic pruning.
///
/// Pitches are grouped by a harmonic or near-equivalent relation. The smallest
/// pitch in a group (the base frequency) represents it unless another member
/// beats it decisively by `harmonic_drop`. All comparisons use explicit
/// thresholds; ties resolve toward the smaller pitch.
pub fn prune_harmonics(scored: &[(usize, f64)], params: &DetectionParams) -> Vec<(usize, f64)> {
    let mut ranked: Vec<(usize, f64)> = scored.to_vec();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    let related = |p: usize, q: usize| -> bool {
        let (lo, hi) = (p.min(q), p.max(q));
        let harmonic = hi % lo == 0 && (hi / lo) <= 4;
        let near = (hi - lo) as f64 / lo as f64 <= params.equivalent_tol;
        harmonic || near
    };

    let mut groups: Vec<Vec<(usize, f64)>> = Vec::new();
    for candidate in ranked {
        let mut placed = false;
        for group in &mut groups {
            if group.iter().any(|(pitch, _)| related(candidate.0, *pitch)) {
                group.push(candidate);
                placed = true;
                break;
            }
        }
        if !placed {
            groups.push(vec![candidate]);
        }
    }

    let mut representatives: Vec<(usize, f64)> = groups
        .iter()
        .map(|group| {
            let base_pitch = group.iter().map(|(pitch, _)| *pitch).min().unwrap();
            let base_score = group
                .iter()
                .find(|(pitch, _)| *pitch == base_pitch)
                .map(|(_, score)| *score)
                .unwrap();
            let mut winner = (base_pitch, base_score);
            for (pitch, score) in group {
                if *score > base_score * (1.0 + params.harmonic_drop) && *score > winner.1 {
                    winner = (*pitch, *score);
                }
            }
            winner
        })
        .collect();

    representatives.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    representatives.truncate(params.max_candidates);
    representatives
}

fn combine_axis(
    sx: &std::collections::BTreeMap<usize, f64>,
    sy: &std::collections::BTreeMap<usize, f64>,
) -> Vec<(usize, f64, f64, f64)> {
    let mut combined = Vec::new();
    for (pitch, x) in sx {
        if let Some(y) = sy.get(pitch) {
            // geometric mean: a pitch must be periodic on both axes to win
            let score = (x * y).sqrt();
            combined.push((*pitch, score, *x, *y));
        }
    }
    combined
}

/// Deterministic uniform cut grid: `desired_cells` cells spanning `[0, limit]`.
///
/// Each interior cut snaps to the strongest profile position inside a fixed
/// search window around its uniform target; if nothing beats the mean-strength
/// threshold, the cut lands on the rounded target. Ties take the lower index,
/// so the transform is reproducible bit-for-bit.
pub fn uniform_cuts(
    profile: &[f64],
    limit: usize,
    desired_cells: usize,
    params: &DetectionParams,
) -> Vec<usize> {
    if limit == 0 {
        return vec![0];
    }
    if limit == 1 || desired_cells == 0 {
        return vec![0, limit];
    }
    let desired_cells = desired_cells.min(limit).max(1);
    let cell_width = limit as f64 / desired_cells as f64;
    let search_window =
        (cell_width * params.cut_search_window_ratio).max(params.cut_min_search_window);
    let mean = if profile.is_empty() {
        0.0
    } else {
        profile.iter().sum::<f64>() / profile.len() as f64
    };
    let strength_threshold = mean * params.cut_strength_threshold;

    let mut cuts = Vec::with_capacity(desired_cells + 1);
    cuts.push(0);
    for idx in 1..desired_cells {
        let target = cell_width * idx as f64;
        let prev = *cuts.last().unwrap();
        if prev + 1 >= limit {
            break;
        }
        let start = ((target - search_window).floor() as isize)
            .max(prev as isize + 1)
            .max(0) as usize;
        let end = ((target + search_window).ceil() as isize).min(limit as isize - 1) as usize;
        let (start, end) = if end < start {
            (prev + 1, prev + 1)
        } else {
            (start, end)
        };

        let mut best_idx = start.min(profile.len().saturating_sub(1));
        let mut best_val = f64::NEG_INFINITY;
        for (i, &v) in profile
            .iter()
            .enumerate()
            .take(end.min(profile.len().saturating_sub(1)) + 1)
            .skip(start)
        {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }

        if best_val < strength_threshold {
            let mut fallback = target.round() as isize;
            if fallback <= prev as isize {
                fallback = prev as isize + 1;
            }
            if fallback >= limit as isize {
                fallback = (limit as isize - 1).max(prev as isize + 1);
            }
            best_idx = fallback as usize;
        }
        cuts.push(best_idx);
    }
    if *cuts.last().unwrap() != limit {
        cuts.push(limit);
    }
    cuts.sort_unstable();
    cuts.dedup();
    cuts
}

/// Identity cuts: each output pixel maps to exactly one input pixel.
pub fn identity_cuts(limit: usize) -> Vec<usize> {
    (0..=limit).collect()
}

/// Result of the full detection + transform-planning stage.
pub struct Detection {
    pub report: GridReport,
    pub col_cuts: Vec<usize>,
    pub row_cuts: Vec<usize>,
}

/// Run the deterministic detection pipeline on already-computed profiles.
pub fn detect(
    col_profile: &[f64],
    row_profile: &[f64],
    width: usize,
    height: usize,
    params: &DetectionParams,
) -> Detection {
    let stats_x = axis_stats(col_profile);
    let stats_y = axis_stats(row_profile);

    let sx = score_axis(col_profile, params);
    let sy = score_axis(row_profile, params);
    let combined = combine_axis(&sx, &sy);
    let scored: Vec<(usize, f64)> = combined.iter().map(|(p, s, _, _)| (*p, *s)).collect();
    let pruned = prune_harmonics(&scored, params);

    let candidates: Vec<Candidate> = pruned
        .iter()
        .map(|(pitch, score)| Candidate {
            pitch: *pitch,
            score: *score,
            score_x: sx.get(pitch).copied().unwrap_or(0.0),
            score_y: sy.get(pitch).copied().unwrap_or(0.0),
        })
        .collect();

    let signal = (stats_x.mean_over_max.max(0.0) * stats_y.mean_over_max.max(0.0)).sqrt();

    let Some((best_pitch, best_score)) = pruned.first().copied() else {
        return no_grid_detection(width, height, candidates, stats_x, stats_y, signal);
    };
    let second_score = pruned.get(1).map(|(_, s)| *s).unwrap_or(0.0);
    let margin = if best_score > 0.0 {
        (best_score - second_score) / best_score
    } else {
        0.0
    };

    let confidence = (0.55 * (best_score / 0.45).min(1.0)
        + 0.30 * (margin / 0.25).min(1.0)
        + 0.15 * (signal / 0.12).min(1.0))
    .min(1.0);

    let no_grid = signal < params.signal_gate
        || best_score < params.score_gate
        || margin < params.margin_gate;

    if no_grid {
        return no_grid_detection(width, height, candidates, stats_x, stats_y, signal);
    }

    let col_cells = (width as f64 / best_pitch as f64).round() as usize;
    let row_cells = (height as f64 / best_pitch as f64).round() as usize;
    let col_cuts = uniform_cuts(col_profile, width, col_cells, params);
    let row_cuts = uniform_cuts(row_profile, height, row_cells, params);

    let report = GridReport {
        verdict: GridVerdict::Detected,
        pitch: best_pitch,
        pixel_size: best_pitch as f64,
        confidence,
        signal,
        margin,
        best_score,
        candidates,
        stats_x,
        stats_y,
        output_width: (col_cuts.len() - 1) as u32,
        output_height: (row_cuts.len() - 1) as u32,
    };
    Detection {
        report,
        col_cuts,
        row_cuts,
    }
}

fn no_grid_detection(
    width: usize,
    height: usize,
    candidates: Vec<Candidate>,
    stats_x: AxisStats,
    stats_y: AxisStats,
    signal: f64,
) -> Detection {
    let best_score = candidates.first().map(|c| c.score).unwrap_or(0.0);
    let margin = match (candidates.first(), candidates.get(1)) {
        (Some(first), Some(second)) if first.score > 0.0 => {
            (first.score - second.score) / first.score
        }
        _ => 0.0,
    };
    let report = GridReport {
        verdict: GridVerdict::NoGrid,
        pitch: 0,
        pixel_size: 0.0,
        confidence: 0.0,
        signal,
        margin,
        best_score,
        candidates,
        stats_x,
        stats_y,
        output_width: width as u32,
        output_height: height as u32,
    };
    Detection {
        report,
        col_cuts: identity_cuts(width),
        row_cuts: identity_cuts(height),
    }
}

/// Build the transform for a caller-supplied pitch override (no scoring).
pub fn override_detection(
    col_profile: &[f64],
    row_profile: &[f64],
    width: usize,
    height: usize,
    pitch: f64,
    params: &DetectionParams,
) -> Detection {
    let stats_x = axis_stats(col_profile);
    let stats_y = axis_stats(row_profile);
    let col_cells = (width as f64 / pitch).round() as usize;
    let row_cells = (height as f64 / pitch).round() as usize;
    let col_cuts = uniform_cuts(col_profile, width, col_cells, params);
    let row_cuts = uniform_cuts(row_profile, height, row_cells, params);
    let report = GridReport {
        verdict: GridVerdict::Override,
        pitch: pitch.round() as usize,
        pixel_size: pitch,
        confidence: 1.0,
        signal: (stats_x.mean_over_max * stats_y.mean_over_max).sqrt(),
        margin: 1.0,
        best_score: 1.0,
        candidates: Vec::new(),
        stats_x,
        stats_y,
        output_width: (col_cuts.len() - 1) as u32,
        output_height: (row_cuts.len() - 1) as u32,
    };
    Detection {
        report,
        col_cuts,
        row_cuts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prune_prefers_base_pitch_over_harmonics() {
        let params = DetectionParams::default();
        // Base pitch 8 slightly weaker than its harmonic 16; base must win.
        let scored = vec![(16, 0.80), (8, 0.75), (24, 0.70), (5, 0.30)];
        let pruned = prune_harmonics(&scored, &params);
        assert_eq!(pruned[0].0, 8);
        // 16 and 24 are harmonics of 8 and collapse into its group.
        assert!(!pruned.iter().any(|(p, _)| *p == 16 || *p == 24));
        // Non-harmonic pitch 5 survives as a separate candidate.
        assert!(pruned.iter().any(|(p, _)| *p == 5));
    }

    #[test]
    fn prune_lets_decisively_stronger_harmonic_win() {
        let params = DetectionParams::default();
        // Harmonic beats the base by more than harmonic_drop.
        let scored = vec![(16, 0.95), (8, 0.50), (3, 0.20)];
        let pruned = prune_harmonics(&scored, &params);
        assert_eq!(pruned[0].0, 16);
    }

    #[test]
    fn prune_ranking_is_deterministic_on_ties() {
        let params = DetectionParams::default();
        let scored = vec![(10, 0.5), (7, 0.5), (13, 0.5)];
        let a = prune_harmonics(&scored, &params);
        let b = prune_harmonics(&scored, &params);
        assert_eq!(a, b);
        // Equal scores resolve to the smaller pitch first.
        assert_eq!(a[0].0, 7);
    }

    #[test]
    fn prune_respects_candidate_cap() {
        let params = DetectionParams::default();
        let scored: Vec<(usize, f64)> = (2..40).map(|d| (d, 1.0 / d as f64)).collect();
        let pruned = prune_harmonics(&scored, &params);
        assert!(pruned.len() <= params.max_candidates);
    }

    #[test]
    fn uniform_cuts_are_sorted_complete_and_deterministic() {
        let params = DetectionParams::default();
        let profile: Vec<f64> = (0..128)
            .map(|i| if i % 8 == 0 { 10.0 } else { 1.0 })
            .collect();
        let a = uniform_cuts(&profile, 128, 16, &params);
        let b = uniform_cuts(&profile, 128, 16, &params);
        assert_eq!(a, b);
        assert_eq!(*a.first().unwrap(), 0);
        assert_eq!(*a.last().unwrap(), 128);
        assert!(a.windows(2).all(|w| w[1] > w[0]));
        assert_eq!(a.len(), 17);
    }

    #[test]
    fn identity_cuts_cover_every_pixel() {
        let cuts = identity_cuts(4);
        assert_eq!(cuts, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn flat_profile_yields_no_grid() {
        let params = DetectionParams::default();
        let profile = vec![0.0f64; 64];
        let detection = detect(&profile, &profile, 64, 64, &params);
        assert_eq!(detection.report.verdict, GridVerdict::NoGrid);
        assert_eq!(detection.report.confidence, 0.0);
        assert_eq!(detection.col_cuts.len(), 65);
    }

    #[test]
    fn periodic_profile_detects_pitch() {
        let params = DetectionParams::default();
        // Strong sharp peak every 8 positions on both axes.
        let profile: Vec<f64> = (0..128)
            .map(|i| if i % 8 == 0 { 100.0 } else { 0.0 })
            .collect();
        let detection = detect(&profile, &profile, 128, 128, &params);
        assert_eq!(detection.report.verdict, GridVerdict::Detected);
        assert_eq!(detection.report.pitch, 8);
        assert!(detection.report.confidence > 0.9);
    }

    #[test]
    fn override_detection_uses_supplied_pitch() {
        let params = DetectionParams::default();
        let profile = vec![1.0f64; 64];
        let detection = override_detection(&profile, &profile, 64, 64, 8.0, &params);
        assert_eq!(detection.report.verdict, GridVerdict::Override);
        assert_eq!(detection.report.pitch, 8);
        assert_eq!(detection.report.output_width, 8);
    }
}
