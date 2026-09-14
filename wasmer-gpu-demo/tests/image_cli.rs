//! Drives the `gpu-image` binary over a small deterministic image. Expected
//! output fingerprints (FNV-1a 64) were computed by an independent Python
//! implementation of the same integer kernels (see the aisec-6th bead notes),
//! so a shared slip in the WGSL and the Rust reference cannot pass here.

use std::process::Command;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 6;
const INPUT_FNV: &str = "cd6031a4ea8506b5";
const EXPECTED: [(&str, Option<&str>, &str, &str, &str); 5] = [
    // filter, explicit param, output fnv1a64, first two pixels, last two pixels
    (
        "blur",
        None,
        "385eb960b6a84741",
        "1a070eff320c16ff",
        "6f8926ff4d9e28ff",
    ),
    (
        "blur",
        Some("2"),
        "b7381cca0f70089b",
        "12040aff2b0812ff",
        "658d21ff3ca526ff",
    ),
    (
        "blur",
        Some("1"),
        "1991432d90a73411",
        "0c0306ff280610ff",
        "528f20ff2eab26ff",
    ),
    (
        "edges",
        None,
        "c95789d363f847b5",
        "151515ff282828ff",
        "101010ff282828ff",
    ),
    (
        "emboss",
        None,
        "f75ca4684b7d9a92",
        "9f9f9fffbdbdbdff",
        "7d7d7dffbdbdbdff",
    ),
];

/// Same RGB pattern either way; `varied_alpha` gives every pixel its own
/// alpha so a kernel that drops or replaces alpha cannot match.
fn pattern(varied_alpha: bool) -> Vec<u8> {
    let mut rgba = Vec::new();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let alpha = if varied_alpha {
                ((x * 53 + y * 29 + 7) % 256) as u8
            } else {
                255
            };
            rgba.extend_from_slice(&[
                ((x * 37 + y * 11) % 256) as u8,
                ((x * x * 3 + y * 7) % 256) as u8,
                (((x ^ y) * 16) % 256) as u8,
                alpha,
            ]);
        }
    }
    rgba
}

fn input_path(varied_alpha: bool) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "gpu-image-test-{}-{}.rgba",
        std::process::id(),
        varied_alpha
    ));
    std::fs::write(&path, pattern(varied_alpha)).expect("write raw input");
    path
}

fn run(args: &[&str]) -> (bool, serde_json::Value) {
    run_on(false, args)
}

fn run_on(varied_alpha: bool, args: &[&str]) -> (bool, serde_json::Value) {
    let path = input_path(varied_alpha);
    let output = Command::new(env!("CARGO_BIN_EXE_gpu-image"))
        .args([
            "--input",
            path.to_str().unwrap(),
            "--width",
            "8",
            "--height",
            "6",
            "--json",
        ])
        .args(args)
        .output()
        .expect("gpu-image runs");
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|_| {
        panic!(
            "gpu-image emitted no JSON\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output.status.success(), report)
}

fn decode_base64(text: &str) -> Vec<u8> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::new();
    let mut acc = 0u32;
    let mut bits = 0;
    for byte in text.bytes().filter(|&b| b != b'=') {
        let value = TABLE
            .iter()
            .position(|&t| t == byte)
            .expect("base64 alphabet") as u32;
        acc = (acc << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((acc >> bits) & 0xff) as u8);
        }
    }
    out
}

/// FNV-1a 64 of the committed guest file, so the report's embedded-guest
/// fingerprint must equal the source the reviewer can read.
fn committed_guest_fnv() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("guest/image_filter.wat");
    let bytes = std::fs::read(path).expect("read committed guest");
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, &b| {
        (hash ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    });
    format!("{hash:016x}")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn every_filter_matches_the_independent_reference() {
    for (filter, param, fnv, head, tail) in EXPECTED {
        let mut args = vec!["--filter", filter];
        if let Some(param) = param {
            args.extend(["--param", param]);
        }
        let (success, report) = run(&args);
        assert!(success, "{filter} exited nonzero: {report}");
        assert_eq!(report["matches"], true, "{filter}: {report}");
        assert_eq!(report["mismatched_pixels"], 0);
        assert_eq!(
            report["changed_pixels"], 48,
            "{filter} must change every pixel"
        );
        assert_eq!(report["input_fnv1a64"], INPUT_FNV);
        assert_eq!(report["output_fnv1a64"], fnv, "{filter} output fingerprint");
        assert_eq!(report["runtime"], "wasix");
        assert_eq!(report["source"], "file");
        assert_eq!(
            report["guest_fnv1a64"],
            committed_guest_fnv(),
            "{filter}: embedded guest differs from guest/image_filter.wat"
        );
        assert_eq!(report["resources_reclaimed"], true);
        assert_eq!(report["dispatch_us"].as_array().map(Vec::len), Some(1));
        assert!(report["adapter"]
            .as_str()
            .is_some_and(|name| !name.is_empty()));
        let output = decode_base64(report["output_rgba_b64"].as_str().unwrap());
        assert_eq!(output.len(), 8 * 6 * 4);
        assert_eq!(hex(&output[..8]), head, "{filter} first pixels");
        assert_eq!(
            hex(&output[output.len() - 8..]),
            tail,
            "{filter} last pixels"
        );
        assert_eq!(
            decode_base64(report["input_rgba_b64"].as_str().unwrap()),
            pattern(false)
        );
    }
}

const VARIED_ALPHA_INPUT_FNV: &str = "6807866dd6a2995d";
const VARIED_ALPHA_EXPECTED: [(&str, Option<&str>, &str, &str, &str); 5] = [
    (
        "blur",
        None,
        "059bbb06fd8d1329",
        "1a070e07320c163c",
        "6f8926d64d9e280b",
    ),
    (
        "blur",
        Some("2"),
        "6675238b619ba66f",
        "12040a072b08123c",
        "658d21d63ca5260b",
    ),
    (
        "blur",
        Some("1"),
        "89e50f6b63885c05",
        "0c0306072806103c",
        "528f20d62eab260b",
    ),
    (
        "edges",
        None,
        "c12099770ba7853d",
        "151515072828283c",
        "101010d62828280b",
    ),
    (
        "emboss",
        None,
        "7b67d53089001bb2",
        "9f9f9f07bdbdbd3c",
        "7d7d7dd6bdbdbd0b",
    ),
];

#[test]
fn every_filter_carries_source_alpha_through() {
    for (filter, param, fnv, head, tail) in VARIED_ALPHA_EXPECTED {
        let mut args = vec!["--filter", filter];
        if let Some(param) = param {
            args.extend(["--param", param]);
        }
        let (success, report) = run_on(true, &args);
        assert!(success, "{filter} exited nonzero: {report}");
        assert_eq!(report["matches"], true, "{filter}: {report}");
        assert_eq!(report["input_fnv1a64"], VARIED_ALPHA_INPUT_FNV);
        assert_eq!(
            report["output_fnv1a64"], fnv,
            "{filter} varied-alpha fingerprint"
        );
        let output = decode_base64(report["output_rgba_b64"].as_str().unwrap());
        assert_eq!(hex(&output[..8]), head, "{filter} first pixels");
        assert_eq!(
            hex(&output[output.len() - 8..]),
            tail,
            "{filter} last pixels"
        );
        let input = pattern(true);
        for (i, chunk) in output.chunks_exact(4).enumerate() {
            assert_eq!(chunk[3], input[i * 4 + 3], "{filter} alpha at pixel {i}");
        }
    }
}

#[test]
fn wrong_shader_control_is_reported_as_a_mismatch() {
    let (success, report) = run(&["--filter", "blur", "--control", "wrong-shader"]);
    assert!(!success, "the identity kernel must not pass as blur");
    assert_eq!(report["matches"], false);
    assert_eq!(report["exit_code"], 0, "the guest itself ran cleanly");
    assert_eq!(
        report["changed_pixels"], 0,
        "identity leaves the input intact"
    );
    assert!(report["mismatched_pixels"].as_u64().unwrap() > 0);
    assert_eq!(report["resources_reclaimed"], true);
}

#[test]
fn no_dispatch_control_is_reported_as_a_mismatch() {
    let (success, report) = run(&["--filter", "edges", "--control", "no-dispatch"]);
    assert!(!success);
    assert_eq!(report["matches"], false);
    assert_eq!(report["dispatch_us"].as_array().map(Vec::len), Some(0));
    assert!(report["mismatched_pixels"].as_u64().unwrap() > 0);
    let output = decode_base64(report["output_rgba_b64"].as_str().unwrap());
    assert!(
        output.iter().all(|&b| b == 0),
        "untouched output region stays zero"
    );
}

#[test]
fn blur_radius_past_five_is_refused_before_any_gpu_work() {
    let output = Command::new(env!("CARGO_BIN_EXE_gpu-image"))
        .args(["--filter", "blur", "--param", "6", "--json"])
        .output()
        .expect("gpu-image runs");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("1..=5"));
}

#[test]
fn oversized_dimensions_are_refused_before_any_gpu_work() {
    let output = Command::new(env!("CARGO_BIN_EXE_gpu-image"))
        .args(["--width", "513", "--height", "8", "--json"])
        .output()
        .expect("gpu-image runs");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("1..=512"));
}
