use std::{path::Path, process::Command};

/// Runs a short real training job when the MNIST files are present. The
/// dataset is not committed, so the test states clearly when it cannot run
/// instead of passing vacuously.
#[test]
fn mnist_trainer_learns_and_passes_its_cpu_check() {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mnist/train-images-idx3-ubyte");
    if !data.is_file() {
        eprintln!(
            "SKIPPED: MNIST files missing at {}; see docs/MNIST.org",
            data.display()
        );
        return;
    }
    let output = Command::new(env!("CARGO_BIN_EXE_gpu-mnist"))
        .args([
            "--steps",
            "20",
            "--eval-every",
            "20",
            "--progress-every",
            "5",
            "--log-dir",
        ])
        .arg(std::env::temp_dir().join("wasmer-gpu-demo-mnist-test"))
        .output()
        .expect("gpu-mnist runs");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "gpu-mnist failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let events: Vec<serde_json::Value> = stdout
        .lines()
        .map(|line| serde_json::from_str(line).expect("JSON line"))
        .collect();
    let find = |name: &str| {
        events
            .iter()
            .find(|e| e["event"] == name)
            .unwrap_or_else(|| panic!("no {name} event"))
    };
    let start = find("start");
    assert_eq!(start["train_samples"], 12000);
    assert_eq!(start["test_samples"], 2000);
    assert!(start["execution"]
        .as_str()
        .unwrap()
        .contains("InstantiationHook"));
    let check = find("check");
    assert_eq!(check["passed"], true);
    assert_eq!(check["kind"], "learning_signals");
    assert_eq!(start["model_kind"], "cnn");
    assert!(check["final_vs_initial_max_abs_diff"].as_f64().unwrap() > 0.0);
    assert_eq!(check["all_finite"], true);
    let done = find("done");
    assert_eq!(done["checks_passed"], true);
    let first = done["first_loss"].as_f64().unwrap();
    let last = done["final_loss"].as_f64().unwrap();
    assert!(
        first > 2.0 && last < first,
        "loss did not fall: {first} -> {last}"
    );
    assert!(done["final_test_accuracy"].as_f64().unwrap() > 0.5);
    assert_eq!(done["live_buffers_after"], 0);
    assert_eq!(done["live_pipelines_after"], 0);
    let evals = events.iter().filter(|e| e["event"] == "eval").count();
    assert_eq!(evals, 1);
    assert_eq!(
        find("weights")["values"].as_array().map(Vec::len),
        Some(11_738)
    );
}

/// The softmax path keeps its exact f64 parity check.
#[test]
fn softmax_trainer_still_matches_its_cpu_reference() {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mnist/train-images-idx3-ubyte");
    if !data.is_file() {
        eprintln!("SKIPPED: MNIST files missing");
        return;
    }
    let output = Command::new(env!("CARGO_BIN_EXE_gpu-mnist"))
        .args([
            "--model",
            "softmax",
            "--steps",
            "5",
            "--eval-every",
            "0",
            "--progress-every",
            "5",
            "--log-dir",
        ])
        .arg(std::env::temp_dir().join("wasmer-gpu-demo-mnist-test"))
        .output()
        .expect("gpu-mnist runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let check: serde_json::Value = stdout
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("JSON"))
        .find(|e| e["event"] == "check")
        .expect("check event");
    assert_eq!(check["kind"], "cpu_parity_first_update");
    assert!(check["first_update_max_abs_diff"].as_f64().unwrap() <= 1e-4);
    assert_eq!(check["passed"], true);
}
