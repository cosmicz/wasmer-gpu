"use strict";
const byId = (id) => document.getElementById(id);
const controls = {
  "invalid_range.wat": "Invalid ranges refused", "invalid_handle.wat": "Invalid handles refused",
  "invalid_shader.wat": "Invalid shaders refused", "unaligned_read.wat": "Unaligned reads refused",
  "quota.wat": "Allocation quotas enforced", "leak.wat": "Leaked resources reclaimed",
  "early_exit.wat": "Early-exit cleanup", "trap_after_create.wat": "Guest trap and cleanup",
  "wasix_leak.wat": "WASIX guest leak reclaimed"
};
const kernelNames = {
  "collatz_valid.wat": "Collatz iterations", "lowbias32_warm.wat": "Lowbias32 hash × 8",
  "wasix_collatz.wat": "Collatz via WASIX guest"
};
const runtimeNames = {core: "core module, hand-built imports", wasix: "WASIX command guest via runtime hook"};
const sameGuests = (rows, names) => rows.length === Object.keys(names).length
  && rows.every(row => Object.hasOwn(names, row.guest)) && new Set(rows.map(row => row.guest)).size === rows.length;
let currentReport = null;
const timingOk = (value) => Number.isInteger(value) && value >= 0 && value <= 1e12;
const timing = (value) => timingOk(value) ? `${(value / 1000).toFixed(2)} ms` : "Not reported";
const timingFields = ["cpu_reference_ns", "upload_us", "pipeline_create_us", "readback_us"];
function showKernel() {
  if (!currentReport) return;
  const kernel = currentReport.kernels.find(row => row.guest === byId("kernel").value);
  if (!kernel) return;
  byId("gpu").textContent = `[${kernel.gpu_result.join(", ")}]`;
  byId("cpu").textContent = `[${kernel.cpu_result.join(", ")}]`;
  byId("setup").textContent = timing(currentReport.setup_us);
  byId("upload").textContent = timing(kernel.upload_us);
  byId("pipeline").textContent = timing(kernel.pipeline_create_us);
  byId("compute").textContent = timing(kernel.dispatch_us.reduce((sum, value) => sum + value, 0));
  byId("readback").textContent = timing(kernel.readback_us);
  byId("kernel-detail").textContent = `${runtimeNames[kernel.runtime] || kernel.runtime} · ${kernel.elements} values · ${kernel.rounds} dispatch(es) on one pipeline · ${kernel.dispatch_us.map(timing).join(", ")}`;
}
byId("kernel").addEventListener("change", showKernel);
function resetResults() {
  currentReport = null;
  byId("kernel").disabled = true;
  byId("kernel").replaceChildren(new Option("Waiting for all guests", ""));
  byId("kernel-detail").textContent = "One host API, guest-owned shaders, core and WASIX guests. Each run checks all three.";
  for (const id of ["gpu", "cpu", "setup", "upload", "pipeline", "compute", "readback"]) byId(id).textContent = "—";
  byId("receipt").textContent = "Pending";
  byId("duration").textContent = "Waiting for this invocation.";
  byId("adapter").textContent = "Your native GPU";
  byId("backend").textContent = "Adapter identified on execution";
  const pending = document.createElement("li");
  pending.className = "pending";
  pending.textContent = "Checks pending for this run.";
  byId("checks").replaceChildren(pending);
}
byId("run").addEventListener("click", async () => {
  byId("run").disabled = true;
  document.body.classList.add("running");
  byId("run-state").textContent = "Executing guest…";
  byId("verdict").className = "verdict";
  byId("verdict").textContent = "Wasmer is running the guest on your native GPU.";
  byId("raw").textContent = "Waiting for a fresh report.";
  resetResults();
  try {
    const response = await fetch("api/run", { method: "POST", headers: {"Content-Type": "application/json"}, body: "{}", signal: AbortSignal.timeout(35000) });
    const data = await response.json();
    byId("raw").textContent = JSON.stringify(data, null, 2);
    if (!response.ok || data.ok !== true) throw new Error(data.error || `Local server returned HTTP ${response.status}.`);
    const report = data.report;
    if (!report || report.all_kernels_match !== true || report.all_controls_pass !== true || report.resources_reclaimed !== true
      || !Array.isArray(report.kernels) || !sameGuests(report.kernels, kernelNames)
      || !Array.isArray(report.controls) || !sameGuests(report.controls, controls)
      || report.controls.some(row => row.passed !== true)
      || report.kernels.some(row => row.matches !== true || !Array.isArray(row.gpu_result) || !row.gpu_result.length
        || JSON.stringify(row.gpu_result) !== JSON.stringify(row.cpu_result))) throw new Error("The report did not establish matching results and passing checks.");
    if (!timingOk(report.setup_us) || report.kernels.some(row => !Array.isArray(row.dispatch_us) || !row.dispatch_us.length
      || !row.dispatch_us.every(timingOk) || timingFields.some(key => !timingOk(row[key])))) throw new Error("The report did not carry complete timing evidence, so the run is not shown as verified.");
    currentReport = report;
    byId("kernel").replaceChildren(...report.kernels.map(row => new Option(kernelNames[row.guest], row.guest)));
    byId("kernel").disabled = false;
    showKernel();
    byId("adapter").textContent = report.adapter;
    byId("backend").textContent = `${report.backend} backend · native host`;
    byId("verdict").textContent = "All three kernel runs, core and WASIX, match their independent CPU references. All resources reclaimed.";
    byId("verdict").className = "verdict success";
    byId("run-state").textContent = "Verified result";
    byId("receipt").textContent = data.run_id;
    byId("duration").textContent = `${data.elapsed_ms} ms end-to-end · ${report.build_host_triple || "Build target not reported"}`;
    byId("checks").replaceChildren(...report.controls.map(row => {
      const item = document.createElement("li"); item.textContent = controls[row.guest]; return item;
    }));
  } catch (error) {
    byId("verdict").className = "verdict error";
    byId("verdict").textContent = error.name === "TimeoutError"
      ? "The local request timed out. Check the server terminal; no successful result was recorded."
      : error.message;
    byId("run-state").textContent = "Run failed";
    byId("receipt").textContent = "No successful receipt";
  } finally {
    document.body.classList.remove("running");
    byId("run").disabled = false;
  }
});
