//! Deterministic matrix runner.
//!
//! Regenerates the synthetic pattern set (fixed seeds), runs the full
//! pipeline over the synthetic patterns plus the repository images, and
//! writes a JSON matrix report containing the best pitch (ratio), confidence,
//! output scale and SHA-256 hash for every case, including the no-grid
//! controls.
//!
//! Usage: cargo run --release --bin matrix-runner -- <OUT_DIR>

// Shares the deterministic synthetic generators with the integration tests.
#[path = "../support/mod.rs"]
mod support;

use image::GenericImageView;
use spritefusion_pixel_snapper::{process_image_common, report_to_json, Config, GridVerdict};
use support::{png_bytes, GridMode, NoGridKind};

const PITCH: usize = 8;
const CELLS_X: usize = 16;
const CELLS_Y: usize = 12;

struct Case {
    name: String,
    group: &'static str,
    expected_verdict: &'static str,
    expected_pitch: usize,
    input: Vec<u8>,
    width: u32,
    height: u32,
}

fn build_cases() -> Vec<Case> {
    let grid_modes = [
        ("grid_sharp", GridMode::Sharp),
        ("grid_blurred", GridMode::Blurred),
        ("grid_low_contrast", GridMode::LowContrast),
        ("grid_transparent_edges", GridMode::TransparentEdges),
        ("grid_diagonal45", GridMode::Diagonal45),
        ("grid_noisy", GridMode::Noisy),
    ];

    let mut cases = Vec::new();
    for (idx, (name, mode)) in grid_modes.iter().enumerate() {
        let img = support::synthetic_grid(CELLS_X, CELLS_Y, PITCH, *mode, (idx as u64) + 1);
        let (w, h) = img.dimensions();
        cases.push(Case {
            name: (*name).to_string(),
            group: "synthetic_grid",
            expected_verdict: "detected",
            expected_pitch: PITCH,
            input: png_bytes(&img),
            width: w,
            height: h,
        });
    }

    for (name, kind) in [
        ("nogrid_smooth", NoGridKind::Smooth),
        ("nogrid_noise", NoGridKind::Noise),
    ] {
        let img = support::synthetic_no_grid(128, kind);
        let (w, h) = img.dimensions();
        cases.push(Case {
            name: name.to_string(),
            group: "no_grid_control",
            expected_verdict: "no_grid",
            expected_pitch: 0,
            input: png_bytes(&img),
            width: w,
            height: h,
        });
    }

    for name in ["repo_od", "hero", "details"] {
        let path = format!("static/{name}.png");
        let bytes = std::fs::read(&path).expect("repo image present");
        let (w, h) = image::load_from_memory(&bytes)
            .expect("decode repo image")
            .dimensions();
        cases.push(Case {
            name: name.to_string(),
            group: "repository_image",
            expected_verdict: "no_grid",
            expected_pitch: 0,
            input: bytes,
            width: w,
            height: h,
        });
    }
    cases
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out_dir = args
        .next()
        .unwrap_or_else(|| "artifacts/matrix".to_string());
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let cases = build_cases();
    let config = Config::default();

    let mut rows = Vec::new();
    let mut all_pass = true;

    for case in &cases {
        let processed = process_image_common(&case.input, Some(config.clone()))
            .unwrap_or_else(|e| panic!("case {} failed: {e}", case.name));
        let report = processed.report();

        let input_path = format!("{out_dir}/{}.input.png", case.name);
        let output_path = format!("{out_dir}/{}.output.png", case.name);
        std::fs::write(&input_path, &case.input).expect("write input");
        std::fs::write(&output_path, processed.output_bytes()).expect("write output");

        let verdict = match report.verdict {
            GridVerdict::Detected => "detected",
            GridVerdict::NoGrid => "no_grid",
            GridVerdict::Override => "override",
        };
        let pass = verdict == case.expected_verdict
            && (case.expected_pitch == 0 || report.pitch == case.expected_pitch);
        all_pass &= pass;

        // Per-case machine-readable report (same JSON contract as --report).
        let case_json = report_to_json(
            report,
            &input_path,
            &output_path,
            case.width,
            case.height,
            processed.output_sha256(),
        );
        std::fs::write(format!("{out_dir}/{}.report.json", case.name), case_json)
            .expect("write case report");

        rows.push(serde_row(
            case,
            verdict,
            report,
            processed.output_sha256(),
            pass,
        ));

        println!(
            "{:26} {:18} verdict={:8} pitch={:4} conf={:.3} scale={:.4},{:.4} hash={} {}",
            case.name,
            case.group,
            verdict,
            report.pitch,
            report.confidence,
            report.scale_x(case.width),
            report.scale_y(case.height),
            processed.output_sha256(),
            if pass { "PASS" } else { "FAIL" }
        );
    }

    let summary = format!(
        "{{\n  \"tool\": \"spritefusion-pixel-snapper\",\n  \"version\": \"{}\",\n  \"total\": {},\n  \"passed\": {},\n  \"cases\": [\n{}  ]\n}}\n",
        env!("CARGO_PKG_VERSION"),
        rows.len(),
        rows.iter().filter(|r| r.contains("\"pass\": true")).count(),
        rows.join(",\n")
    );
    std::fs::write(format!("{out_dir}/matrix_report.json"), summary).expect("write summary");
    println!("\nWrote matrix report to {out_dir}/matrix_report.json");

    if !all_pass {
        std::process::exit(1);
    }
}

fn serde_row(
    case: &Case,
    verdict: &str,
    report: &spritefusion_pixel_snapper::GridReport,
    hash: &str,
    pass: bool,
) -> String {
    format!(
        "    {{\n      \
         \"name\": \"{}\", \
         \"group\": \"{}\", \
         \"verdict\": \"{}\", \
         \"expected_verdict\": \"{}\", \
         \"pitch\": {}, \
         \"expected_pitch\": {}, \
         \"confidence\": {:.6}, \
         \"score\": {:.6}, \
         \"margin\": {:.6}, \
         \"signal\": {:.6}, \
         \"scale_x\": {:.6}, \
         \"scale_y\": {:.6}, \
         \"output_size\": [{}, {}], \
         \"output_sha256\": \"{}\", \
         \"pass\": {} \
         }}",
        case.name,
        case.group,
        verdict,
        case.expected_verdict,
        report.pitch,
        case.expected_pitch,
        report.confidence,
        report.best_score,
        report.margin,
        report.signal,
        report.scale_x(case.width),
        report.scale_y(case.height),
        report.output_width,
        report.output_height,
        hash,
        pass
    )
}
