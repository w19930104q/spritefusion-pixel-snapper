//! Deterministic detection matrix: synthetic patterns + repository images.
//!
//! Covers the hard cases called out for this pipeline — low contrast,
//! transparent edges, 45-degree edges, blur and noise — plus explicit
//! no-grid inputs, determinism (identical hashes across runs) and the CLI
//! surface (report JSON, pixel-size override).

mod support;

use spritefusion_pixel_snapper::{process_image_common, report_to_json, Config, GridVerdict};
use support::{png_bytes, GridMode, NoGridKind};

const PITCH: usize = 8;
const CELLS_X: usize = 16;
const CELLS_Y: usize = 12;

#[test]
fn synthetic_grid_matrix_detects_base_pitch() {
    let modes = [
        ("sharp", GridMode::Sharp, 0.90),
        ("blurred", GridMode::Blurred, 0.90),
        ("low_contrast", GridMode::LowContrast, 0.75),
        ("transparent_edges", GridMode::TransparentEdges, 0.75),
        ("diagonal45", GridMode::Diagonal45, 0.90),
        ("noisy", GridMode::Noisy, 0.90),
    ];

    for (name, mode, min_confidence) in modes {
        let img = support::synthetic_grid(CELLS_X, CELLS_Y, PITCH, mode, 1);
        let (w, h) = img.dimensions();
        let processed = process_image_common(&png_bytes(&img), Some(Config::default()))
            .unwrap_or_else(|e| panic!("{name} failed: {e}"));
        let report = processed.report();

        // Report JSON is part of the CLI contract; validate it in the loop.
        let json = report_to_json(report, name, "out.png", w, h, processed.output_sha256());
        assert!(json.contains("\"verdict\": \"detected\""), "{name}: {json}");
        assert!(json.contains("\"pitch\": 8"), "{name}: {json}");

        assert_eq!(report.verdict, GridVerdict::Detected, "{name}");
        assert_eq!(report.pitch, PITCH, "{name}");
        assert!(
            report.confidence >= min_confidence,
            "{name} confidence {} below {}",
            report.confidence,
            min_confidence
        );
        assert_eq!(report.output_width as usize, CELLS_X, "{name} width");
        assert_eq!(report.output_height as usize, CELLS_Y, "{name} height");

        // Top candidate is the base pitch; harmonic multiples were pruned.
        assert_eq!(report.candidates.first().unwrap().pitch, PITCH, "{name}");
        assert!(report.candidates.len() <= 8, "{name} candidate cap");
        assert!(processed.output_sha256().len() == 64, "{name} hash");
    }
}

#[test]
fn no_grid_inputs_get_identity_transform_and_zero_confidence() {
    let cases = [
        ("nogrid_smooth", NoGridKind::Smooth),
        ("nogrid_noise", NoGridKind::Noise),
    ];
    for (name, kind) in cases {
        let img = support::synthetic_no_grid(128, kind);
        let (w, h) = img.dimensions();
        let input_bytes = png_bytes(&img);
        let processed = process_image_common(&input_bytes, Some(Config::default())).expect(name);
        let report = processed.report();

        assert_eq!(report.verdict, GridVerdict::NoGrid, "{name}");
        assert_eq!(report.confidence, 0.0, "{name}");
        assert_eq!(report.pitch, 0, "{name}");
        assert_eq!(report.output_width, w, "{name} identity width");
        assert_eq!(report.output_height, h, "{name} identity height");
        assert!(!report.candidates.is_empty() || report.best_score == 0.0);

        // True identity: decoded output pixels equal the input pixels.
        let decoded = image::load_from_memory(processed.output_bytes())
            .unwrap()
            .to_rgba8();
        assert_eq!(decoded.dimensions(), img.dimensions(), "{name} dims");
        assert!(
            decoded
                .enumerate_pixels()
                .all(|(x, y, p)| p == img.get_pixel(x, y)),
            "{name} output pixels must be identical to the input"
        );
    }
}

#[test]
fn pipeline_is_deterministic_across_runs() {
    let img = support::synthetic_grid(CELLS_X, CELLS_Y, PITCH, GridMode::Blurred, 2);
    let bytes = png_bytes(&img);
    let first = process_image_common(&bytes, Some(Config::default())).unwrap();
    let second = process_image_common(&bytes, Some(Config::default())).unwrap();

    assert_eq!(first.output_bytes(), second.output_bytes());
    assert_eq!(first.output_sha256(), second.output_sha256());
    assert_eq!(first.report().pitch, second.report().pitch);
    assert_eq!(first.report().confidence, second.report().confidence);
}

#[test]
fn pixel_size_override_skips_scoring_and_is_reported() {
    let img = support::synthetic_no_grid(128, NoGridKind::Smooth);
    let mut cfg = Config::default();
    cfg.pixel_size_override = Some(8.0);
    let processed = process_image_common(&png_bytes(&img), Some(cfg)).unwrap();
    let report = processed.report();

    assert_eq!(report.verdict, GridVerdict::Override);
    assert_eq!(report.pixel_size, 8.0);
    assert_eq!(report.confidence, 1.0);
    assert_eq!(report.output_width, 16);
    assert_eq!(report.output_height, 16);
}

#[test]
fn repository_images_are_classified_and_stably_hashed() {
    // The repo images are large composite screenshots without one dominant
    // sprite grid; the pipeline must stay honest (no_grid, identity output)
    // rather than inventing a pitch, and hash identically across runs.
    for name in ["repo_od", "hero", "details"] {
        let path = format!("static/{name}.png");
        let input = std::fs::read(&path).expect("repo image present");
        let processed = process_image_common(&input, Some(Config::default())).unwrap();
        let report = processed.report();

        assert_eq!(report.verdict, GridVerdict::NoGrid, "{name}");
        assert_eq!(report.confidence, 0.0, "{name}");
        assert_eq!(processed.output_sha256().len(), 64, "{name} hash");

        let again = process_image_common(&input, Some(Config::default())).unwrap();
        assert_eq!(
            processed.output_sha256(),
            again.output_sha256(),
            "{name} hash drift"
        );
    }
}

#[test]
fn report_json_is_deterministic_and_complete() {
    let img = support::synthetic_grid(CELLS_X, CELLS_Y, PITCH, GridMode::Sharp, 3);
    let (w, h) = img.dimensions();
    let processed = process_image_common(&png_bytes(&img), Some(Config::default())).unwrap();
    let report = processed.report();

    let json_a = report_to_json(report, "in.png", "out.png", w, h, processed.output_sha256());
    let json_b = report_to_json(report, "in.png", "out.png", w, h, processed.output_sha256());
    assert_eq!(json_a, json_b);

    for required in [
        "\"verdict\": \"detected\"",
        "\"pitch\": 8",
        "\"confidence\":",
        "\"scale\":",
        "\"output_sha256\":",
        "\"candidates\":",
        "\"stats_x\":",
        "\"stats_y\":",
    ] {
        assert!(json_a.contains(required), "missing {required}");
    }
}
