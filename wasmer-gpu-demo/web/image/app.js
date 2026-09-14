"use strict";
const byId = (id) => document.getElementById(id);
const MAX_SIDE = 512;
const PARAMS = {
  blur: {name: "Radius", default: 4, max: 5},
  edges: {name: "Gain", default: 2, max: 16},
  emboss: {name: "Relief", default: 1, max: 16}
};
const state = {filter: "edges", running: false, file: null, params: {}};

const compare = byId("compare");
const before = byId("before");
const after = byId("after");
const scrub = byId("scrub");
const param = byId("param");
compare.classList.add("empty");

const fmt = new Intl.NumberFormat("en-US");
const timing = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0
  ? `${(value / 1000).toFixed(2)} ms` : "Not reported";

function setReveal(percent) {
  compare.style.setProperty("--reveal", `${percent}%`);
}
scrub.addEventListener("input", () => setReveal(scrub.value));

function selectFilter(name) {
  state.filter = name;
  for (const button of document.querySelectorAll(".preset")) {
    button.setAttribute("aria-pressed", String(button.dataset.filter === name));
  }
  const spec = PARAMS[name];
  byId("param-name").textContent = spec.name;
  param.max = spec.max;
  const remembered = state.params[name];
  param.value = remembered ?? spec.default;
  byId("param-out").value = param.value;
}
param.addEventListener("input", () => {
  state.params[state.filter] = Number(param.value);
  byId("param-out").value = param.value;
});

function decodePixels(encoded, width, height) {
  const binary = atob(encoded);
  const pixels = new Uint8ClampedArray(width * height * 4);
  if (binary.length !== pixels.length) throw new Error("pixel payload has the wrong length");
  for (let i = 0; i < binary.length; i++) pixels[i] = binary.charCodeAt(i);
  return new ImageData(pixels, width, height);
}

function encodePixels(bytes) {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

function paint(canvas, image) {
  canvas.width = image.width;
  canvas.height = image.height;
  canvas.getContext("2d").putImageData(image, 0, 0);
}

function showReport(result) {
  const report = result.report;
  paint(before, decodePixels(report.input_rgba_b64, report.width, report.height));
  paint(after, decodePixels(report.output_rgba_b64, report.width, report.height));
  compare.style.aspectRatio = `${report.width} / ${report.height}`;
  compare.classList.remove("empty");
  scrub.disabled = false;
  compare.classList.add("revealing");
  setReveal(0);
  scrub.value = 50;
  requestAnimationFrame(() => requestAnimationFrame(() => setReveal(50)));
  setTimeout(() => compare.classList.remove("revealing"), 1000);

  const verdict = byId("verdict");
  verdict.className = "verdict pass";
  verdict.textContent = `GPU output equals the CPU reference on all ${fmt.format(report.pixels)} pixels; ${fmt.format(report.changed_pixels)} changed from the input.`;
  byId("adapter").textContent = `${report.adapter} · ${report.backend}`;
  byId("guest").innerHTML = "";
  byId("guest").append(`${report.guest} · WASIX · kernel `, Object.assign(document.createElement("code"), {textContent: report.filter}),
    ` · embedded ${fmt.format(report.guest_bytes)} B, fnv ${report.guest_fnv1a64.slice(0, 12)}`);
  byId("pixels").textContent = `${report.width} × ${report.height} · ${report.source === "file" ? "your file" : "generated scene"} · fnv ${report.output_fnv1a64.slice(0, 12)}`;
  byId("binary").innerHTML = "";
  byId("binary").append("sha256 ", Object.assign(document.createElement("code"), {textContent: result.binary_sha256.slice(0, 16)}), ` · run ${result.run_id}`);
  byId("t-setup").textContent = timing(report.setup_us);
  byId("t-upload").textContent = timing(report.upload_us);
  byId("t-pipeline").textContent = timing(report.pipeline_create_us);
  byId("t-dispatch").textContent = timing(report.dispatch_us[0]);
  byId("t-readback").textContent = timing(report.readback_us);
  byId("t-cpu").textContent = timing(report.cpu_reference_us);
}

function showError(message) {
  const verdict = byId("verdict");
  verdict.className = "verdict fail";
  verdict.textContent = message;
}

async function run(filter) {
  if (state.running) return;
  selectFilter(filter);
  const sourceChoice = document.querySelector("input[name=source]:checked").value;
  let source = "generated";
  if (sourceChoice === "file") {
    if (!state.file) {
      const hint = byId("source-hint");
      hint.textContent = "Choose an image file first, or switch back to the generated scene.";
      hint.classList.add("warn");
      return;
    }
    source = state.file;
  }
  state.running = true;
  compare.classList.add("running");
  for (const button of document.querySelectorAll(".preset")) button.disabled = true;
  byId("verdict").className = "verdict";
  byId("verdict").textContent = `Running ${filter} on the GPU through the WASIX guest…`;
  try {
    const response = await fetch("api/filter", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({filter, param: Number(param.value), source})
    });
    const result = await response.json();
    if (!response.ok || result.ok !== true) throw new Error(result.error || `Server returned ${response.status}.`);
    showReport(result);
  } catch (error) {
    showError(error.message || "The run failed.");
  } finally {
    state.running = false;
    compare.classList.remove("running");
    for (const button of document.querySelectorAll(".preset")) button.disabled = false;
  }
}

for (const button of document.querySelectorAll(".preset")) {
  button.addEventListener("click", () => run(button.dataset.filter));
}

byId("file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  const hint = byId("source-hint");
  hint.classList.remove("warn");
  if (!file) return;
  try {
    const bitmap = await createImageBitmap(file);
    const scale = Math.min(1, MAX_SIDE / bitmap.width, MAX_SIDE / bitmap.height);
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d", {willReadFrequently: true});
    context.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();
    const data = context.getImageData(0, 0, width, height).data;
    state.file = {width, height, rgba_b64: encodePixels(new Uint8Array(data.buffer))};
    byId("file-name").textContent = file.name;
    document.querySelector("input[name=source][value=file]").checked = true;
    hint.textContent = `${file.name}: ${width} × ${height} px, decoded locally. Pick a preset to run it.`;
  } catch (error) {
    state.file = null;
    hint.textContent = "That file could not be decoded as an image.";
    hint.classList.add("warn");
  }
});

selectFilter(state.filter);
