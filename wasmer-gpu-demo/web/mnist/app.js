"use strict";
const byId = (id) => document.getElementById(id);
const GALLERY = 24;
let next = 0, runId = null, pollTimer = null, digits = [], weights = null;
let losses = [], accs = [], events = [], startEvent = null;

// --- gallery ---------------------------------------------------------------
function drawDigit(canvas, pixels) {
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) { const v = 255 - pixels[i]; img.data.set([v, v, v, 255], i * 4); }
  ctx.putImageData(img, 0, 0);
}
async function loadGallery() {
  const list = byId("gallery");
  try {
    const response = await fetch("api/digits");
    const data = await response.json();
    if (!response.ok || data.ok !== true) throw new Error(data.error || "digits unavailable");
    digits = data.digits;
  } catch (error) {
    const item = document.createElement("li"); item.textContent = `Held-out digits unavailable: ${error.message}`; list.replaceChildren(item); return;
  }
  list.replaceChildren(...digits.map(digit => {
    const item = document.createElement("li"); item.dataset.index = digit.index;
    const canvas = document.createElement("canvas"); canvas.width = 28; canvas.height = 28; drawDigit(canvas, digit.pixels);
    const pred = document.createElement("div"); pred.className = "pred";
    const guess = document.createElement("b"); guess.textContent = "·"; guess.setAttribute("aria-label", "no prediction yet");
    const truth = document.createElement("span"); truth.textContent = `label ${digit.label}`;
    pred.append(guess, truth);
    const conf = document.createElement("div"); conf.className = "conf"; conf.append(document.createElement("i"));
    item.append(canvas, pred, conf); return item;
  }));
}
function applyPredictions(items) {
  for (const p of items) {
    const item = byId("gallery").querySelector(`li[data-index="${p.index}"]`);
    if (!item) continue;
    const confidence = Math.max(...p.probs);
    item.querySelector("b").textContent = String(p.predicted);
    item.querySelector("b").setAttribute("aria-label", `predicted ${p.predicted}, confidence ${(confidence * 100).toFixed(0)} percent`);
    item.querySelector(".conf i").style.width = `${(confidence * 100).toFixed(1)}%`;
    item.className = p.predicted === p.label ? "right" : "wrong";
  }
}
function clearPredictions() {
  for (const item of byId("gallery").querySelectorAll("li")) {
    item.className = ""; const b = item.querySelector("b"); if (b) b.textContent = "·";
    const bar = item.querySelector(".conf i"); if (bar) bar.style.width = "0";
  }
}

// --- loss trace ------------------------------------------------------------
function drawTrace() {
  const W = 640, H = 300, L = 44, R = 44, T = 30, B = 30;
  const maxStep = Math.max(1, ...losses.map(p => p.step), ...accs.map(p => p.step));
  const maxLoss = Math.max(0.5, ...losses.map(p => p.loss));
  const x = (step) => L + (step / maxStep) * (W - L - R);
  const yLoss = (loss) => T + (1 - loss / maxLoss) * (H - T - B);
  const yAcc = (acc) => T + (1 - acc) * (H - T - B);
  byId("loss-line").setAttribute("points", losses.map(p => `${x(p.step).toFixed(1)},${yLoss(p.loss).toFixed(1)}`).join(" "));
  const grid = [];
  for (let k = 0; k <= 4; k++) {
    const y = T + (k / 4) * (H - T - B);
    grid.push(`<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}"/><text x="${L - 6}" y="${y + 4}" text-anchor="end">${(maxLoss * (1 - k / 4)).toFixed(2)}</text><text x="${W - R + 6}" y="${y + 4}">${(100 * (1 - k / 4)).toFixed(0)}%</text>`);
  }
  grid.push(`<text x="${W - R}" y="${H - 8}" text-anchor="end">step ${maxStep}</text>`);
  byId("trace-grid").innerHTML = grid.join("");
  byId("acc-marks").innerHTML = accs.map(p => `<circle cx="${x(p.step).toFixed(1)}" cy="${yAcc(p.acc).toFixed(1)}" r="5"/><text x="${x(p.step).toFixed(1)}" y="${(yAcc(p.acc) - 10).toFixed(1)}" text-anchor="middle">${(p.acc * 100).toFixed(1)}%</text>`).join("");
  byId("trace-note").textContent = losses.length ? `${losses.length} reported steps · ${accs.length} evaluation(s)` : "No points yet.";
}

// --- events ----------------------------------------------------------------
const fmtMs = (ms) => typeof ms === "number" ? `${(ms / 1000).toFixed(1)} s` : "—";
function handle(event) {
  events.push(event);
  switch (event.event) {
    case "start":
      startEvent = event;
      byId("adapter-line").textContent = `${event.adapter} (${event.backend}) · ${event.model} · seed ${event.seed} · ${event.train_samples} train / ${event.test_samples} held-out`;
      byId("model-line").textContent = `Model in this run: ${event.model}.`;
      byId("m-lr").textContent = String(event.learning_rate);
      break;
    case "progress":
      losses.push({ step: event.step, loss: event.loss });
      byId("m-step").textContent = String(event.step + 1);
      byId("m-samples").textContent = event.samples_seen.toLocaleString();
      byId("m-loss").textContent = Number.isFinite(event.loss) ? event.loss.toFixed(3) : "not finite";
      byId("m-elapsed").textContent = fmtMs(event.elapsed_ms);
      drawTrace(); break;
    case "eval":
      accs.push({ step: event.step, acc: event.test_accuracy });
      byId("m-acc").textContent = `${(event.test_accuracy * 100).toFixed(2)}%`;
      byId("m-elapsed").textContent = fmtMs(event.elapsed_ms);
      drawTrace(); break;
    case "predictions":
      applyPredictions(event.items);
      byId("gallery-note").textContent = `GPU forward pass on the held-out set after step ${event.step + 1}.`; break;
    case "check":
      byId("verdict").className = event.passed ? "verdict success" : "verdict error";
      byId("verdict").textContent = event.passed
        ? (event.kind === "cpu_parity_first_update"
          ? `Independent CPU check passed: first update within ${event.first_update_max_abs_diff.toExponential(1)} of the f64 reference across ${event.parameters_compared.toLocaleString()} parameters.`
          : `Learning signals verified: ${event.parameters.toLocaleString()} parameters changed (max ${event.final_vs_initial_max_abs_diff.toFixed(3)}), all finite.`)
        : "Run checks FAILED. This run is not a valid training result."; break;
    case "weights":
      // Kept, but only usable once the server confirms a finished run with a
      // passing done receipt (see setState); a failed or stopped run never
      // unlocks browser-side inference.
      weights = event.values; weightsLayout = event.layout || ""; break;
    case "done": break;
    case "error":
      byId("verdict").className = "verdict error"; byId("verdict").textContent = event.message; break;
  }
  byId("raw").textContent = events.slice(-40).map(e => JSON.stringify(e.event === "weights" ? { event: "weights", values: `[${e.values.length} f32]` } : e)).join("\n");
}
function setState(state) {
  document.body.dataset.state = state;
  byId("train").disabled = state === "running";
  byId("stop").disabled = state !== "running";
  byId("reset").disabled = state === "idle";
  const done = events.find(e => e.event === "done");
  const verified = state === "finished" && !!done && done.checks_passed === true && !!weights;
  byId("classify").disabled = !verified;
  if (!verified) byId("pad-result").innerHTML = '<li class="pending">No sketch classified.</li>';
  const words = {
    idle: "Idle. No run captured.",
    running: "Training on the GPU. Points arrive from the trainer as it reports them.",
    finished: done ? `Finished: ${(done.final_test_accuracy * 100).toFixed(2)}% held-out accuracy after ${fmtMs(done.runtime_ms)} · exit ${done.exit_status}.` : "Finished.",
    failed: "Run failed. Nothing above is a verified result.",
    stopped: "Stopped by you. Partial points shown; no final result.",
  };
  byId("state-line").textContent = words[state] || state;
}
async function poll() {
  try {
    const response = await fetch(`api/events?since=${next}`);
    const data = await response.json();
    if (data.run_id !== runId) { resetView(); runId = data.run_id; }
    for (const event of data.events) handle(event);
    next = data.next;
    setState(data.state);
    if (data.state === "running") pollTimer = setTimeout(poll, 250);
  } catch (error) {
    byId("verdict").className = "verdict error"; byId("verdict").textContent = `Lost the local server: ${error.message}`;
    setState("failed");
  }
}
function resetView() {
  next = 0; losses = []; accs = []; events = []; weights = null; weightsLayout = ""; startEvent = null;
  byId("model-line").textContent = "Model: reported by the trainer when a run starts.";
  clearPredictions(); drawTrace();
  for (const id of ["m-step", "m-samples", "m-loss", "m-acc", "m-elapsed", "m-lr"]) byId(id).textContent = "—";
  byId("verdict").className = "verdict"; byId("verdict").textContent = "";
  byId("gallery-note").textContent = "Predictions appear after the first evaluation.";
  byId("adapter-line").textContent = "Adapter identified when training starts.";
  byId("raw").textContent = "No events."; byId("classify").disabled = true;
  byId("pad-result").innerHTML = '<li class="pending">No sketch classified.</li>';
}
async function post(path) {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
  const data = await response.json();
  if (!response.ok || data.ok !== true) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
byId("train").addEventListener("click", async () => {
  try { clearTimeout(pollTimer); const data = await post("api/train"); runId = data.run_id; resetView(); setState("running"); poll(); }
  catch (error) { byId("verdict").className = "verdict error"; byId("verdict").textContent = error.message; }
});
// Stop and Reset only change what the page shows after the server confirms
// the owned process state; a failed request is surfaced and polling resumes
// so the display stays truthful about a process that may still be running.
async function control(path, onSuccess) {
  clearTimeout(pollTimer);
  try {
    const data = await post(path);
    onSuccess(data);
  } catch (error) {
    byId("verdict").className = "verdict error";
    byId("verdict").textContent = `${path} failed: ${error.message}. Showing the server's actual state.`;
  }
  poll();
}
byId("stop").addEventListener("click", () => control("api/stop", () => {}));
byId("reset").addEventListener("click", () => control("api/reset", (data) => {
  if (data.state !== "idle") throw new Error(`server reports ${data.state}`);
  runId = data.run_id; resetView(); setState("idle");
}));

// --- drawing pad (browser-side inference with the GPU-trained weights) -----
const pad = byId("pad"), pctx = pad.getContext("2d");
function clearPad() { pctx.fillStyle = "#fff"; pctx.fillRect(0, 0, 28, 28); }
clearPad();
let drawing = false;
function dot(e) {
  const rect = pad.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width * 28, y = (e.clientY - rect.top) / rect.height * 28;
  pctx.fillStyle = "rgba(23,41,71,0.9)"; pctx.beginPath(); pctx.arc(x, y, 1.6, 0, Math.PI * 2); pctx.fill();
}
pad.addEventListener("pointerdown", (e) => { drawing = true; pad.setPointerCapture(e.pointerId); dot(e); });
pad.addEventListener("pointermove", (e) => { if (drawing) dot(e); });
pad.addEventListener("pointerup", () => { drawing = false; });
pad.addEventListener("pointercancel", () => { drawing = false; });
byId("clear").addEventListener("click", () => { clearPad(); byId("pad-result").innerHTML = '<li class="pending">No sketch classified.</li>'; });
let weightsLayout = "";
function softmaxLogits(w, x) {
  const z = new Array(10).fill(0);
  for (let j = 0; j < 10; j++) { let acc = w[7840 + j]; for (let i = 0; i < 784; i++) acc += w[j * 784 + i] * x[i]; z[j] = acc; }
  return z;
}
// conv 5x5 x8 + ReLU -> max-pool 2x2 -> fc 1152->10, same layout as the guest buffer:
// W2 (10x1152) | b2 (10) | W1 (8x25) | b1 (8)
function cnnLogits(w, x) {
  const OFF_B2 = 11520, OFF_W1 = 11530, OFF_B1 = 11730;
  const conv = new Float32Array(8 * 24 * 24);
  for (let f = 0; f < 8; f++) for (let y = 0; y < 24; y++) for (let xx = 0; xx < 24; xx++) {
    let acc = w[OFF_B1 + f];
    for (let ky = 0; ky < 5; ky++) for (let kx = 0; kx < 5; kx++) acc += w[OFF_W1 + f * 25 + ky * 5 + kx] * x[(y + ky) * 28 + xx + kx];
    conv[(f * 24 + y) * 24 + xx] = Math.max(acc, 0);
  }
  const pooled = new Float32Array(1152);
  for (let f = 0; f < 8; f++) for (let py = 0; py < 12; py++) for (let px = 0; px < 12; px++) {
    const b = (f * 24 + py * 2) * 24 + px * 2;
    pooled[(f * 12 + py) * 12 + px] = Math.max(conv[b], conv[b + 1], conv[b + 24], conv[b + 25]);
  }
  const z = new Array(10).fill(0);
  for (let j = 0; j < 10; j++) { let acc = w[OFF_B2 + j]; for (let i = 0; i < 1152; i++) acc += w[j * 1152 + i] * pooled[i]; z[j] = acc; }
  return z;
}
byId("classify").addEventListener("click", () => {
  if (!weights || document.body.dataset.state !== "finished") return;
  const img = pctx.getImageData(0, 0, 28, 28).data, x = new Float32Array(784);
  for (let i = 0; i < 784; i++) x[i] = (255 - img[i * 4]) / 255;
  const z = weightsLayout.startsWith("cnn") ? cnnLogits(weights, x) : softmaxLogits(weights, x);
  const m = Math.max(...z), exps = z.map(v => Math.exp(v - m)), sum = exps.reduce((a, b) => a + b, 0), probs = exps.map(v => v / sum);
  const top = probs.indexOf(Math.max(...probs));
  byId("pad-result").replaceChildren(...probs.map((p, j) => {
    const li = document.createElement("li"); if (j === top) li.className = "top";
    const b = document.createElement("b"); b.textContent = String(j);
    const bar = document.createElement("i"); bar.style.setProperty("--w", `${(p * 100).toFixed(1)}%`);
    const pct = document.createElement("span"); pct.textContent = `${(p * 100).toFixed(1)}%`;
    li.append(b, bar, pct); return li;
  }));
});

loadGallery(); drawTrace(); setState("idle"); poll();
