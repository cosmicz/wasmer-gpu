use std::process::Command;

/// Expected values are computed independently of the crate (see the Python
/// reference in the bead notes), so a shared bug in the CPU reference and
/// the WGSL kernel cannot pass this test.
const COLLATZ_EXPECTED: [u32; 4] = [0, 1, 7, 2];
const LOWBIAS32_8_ROUNDS_EXPECTED: [u32; 8] = [
    0,
    2_545_901_461,
    2_022_763_327,
    2_777_816_046,
    2_697_318_930,
    3_120_225_037,
    1_226_734_786,
    2_562_130_367,
];

#[test]
fn smoke_binary_runs_guest_kernels_on_core_and_wasix_and_all_controls() {
    let output = Command::new(env!("CARGO_BIN_EXE_gpu-smoke"))
        .arg("--json")
        .output()
        .expect("gpu-smoke runs");

    assert!(
        output.status.success(),
        "gpu-smoke failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("gpu-smoke emits JSON");
    assert_eq!(report["all_kernels_match"], true);
    assert_eq!(report["all_controls_pass"], true);
    assert_eq!(report["resources_reclaimed"], true);
    assert!(report["build_host_triple"]
        .as_str()
        .is_some_and(|host| !host.is_empty()));

    let kernels = report["kernels"].as_array().expect("kernels array");
    assert_eq!(kernels.len(), 3);

    let collatz = &kernels[0];
    assert_eq!(collatz["guest"], "collatz_valid.wat");
    assert_eq!(collatz["runtime"], "core");
    assert_eq!(collatz["gpu_result"], serde_json::json!(COLLATZ_EXPECTED));
    assert_eq!(collatz["cpu_result"], serde_json::json!(COLLATZ_EXPECTED));
    assert_eq!(collatz["matches"], true);
    assert_eq!(collatz["dispatch_us"].as_array().map(Vec::len), Some(1));

    let hash = &kernels[1];
    assert_eq!(hash["guest"], "lowbias32_warm.wat");
    assert_eq!(hash["rounds"], 8);
    assert_eq!(
        hash["gpu_result"],
        serde_json::json!(LOWBIAS32_8_ROUNDS_EXPECTED)
    );
    assert_eq!(
        hash["cpu_result"],
        serde_json::json!(LOWBIAS32_8_ROUNDS_EXPECTED)
    );
    assert_eq!(hash["matches"], true);
    assert_eq!(hash["dispatch_us"].as_array().map(Vec::len), Some(8));

    let wasix = &kernels[2];
    assert_eq!(wasix["guest"], "wasix_collatz.wat");
    assert_eq!(wasix["runtime"], "wasix");
    assert_eq!(wasix["status"], 0);
    assert_eq!(wasix["gpu_result"], serde_json::json!(COLLATZ_EXPECTED));
    assert_eq!(wasix["matches"], true);
    assert_eq!(wasix["dispatch_us"].as_array().map(Vec::len), Some(1));

    let controls = report["controls"].as_array().expect("controls array");
    let names: Vec<&str> = controls
        .iter()
        .map(|c| c["guest"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "invalid_range.wat",
            "invalid_handle.wat",
            "invalid_shader.wat",
            "unaligned_read.wat",
            "quota.wat",
            "early_exit.wat",
            "leak.wat",
            "trap_after_create.wat",
            "wasix_leak.wat",
        ]
    );
    for control in controls {
        assert_eq!(control["passed"], true, "control {}", control["guest"]);
        assert_eq!(control["live_buffers_after"], 0);
        assert_eq!(control["live_pipelines_after"], 0);
        if control["guest"] == "trap_after_create.wat" {
            assert_eq!(control["trapped"], true);
        } else {
            assert_eq!(control["status"], 0);
        }
    }
}
