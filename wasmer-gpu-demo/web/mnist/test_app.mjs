// Focused UI check for the Stop/Reset control path in app.js: a failed or
// refused /api/reset must not clear the view or claim idle. Run with
// `node web/mnist/test_app.mjs`; exits non-zero on failure.
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import vm from "node:vm";

const elements = new Map();
function element(id) {
  const el = { id, textContent: "", className: "", disabled: false, value: "", style: {}, dataset: {}, children: [], handlers: {},
    addEventListener(type, fn) { this.handlers[type] = fn; }, replaceChildren(...nodes) { this.children = nodes; },
    querySelector() { return null; }, querySelectorAll() { return []; }, setAttribute() {}, append(...n) { this.children.push(...n); },
    getContext() { return { fillRect() {}, beginPath() {}, arc() {}, fill() {}, createImageData: () => ({ data: new Uint8ClampedArray(28 * 28 * 4) }), putImageData() {}, getImageData: () => ({ data: new Uint8ClampedArray(28 * 28 * 4) }) }; },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 28, height: 28 }), setPointerCapture() {}, innerHTML: "" };
  return el;
}
const document = { getElementById: (id) => { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); },
  createElement: (tag) => element(tag), body: { dataset: {}, classList: { add() {}, remove() {} } } };
let fetchQueue = [];
const fetch = async (url, options) => {
  if (String(url).startsWith("api/digits")) return { ok: true, status: 200, json: async () => ({ ok: true, digits: [] }) };
  const next = fetchQueue.shift() || { status: 200, body: { ok: true, state: "idle", run_id: 1, next: 0, events: [] } };
  return { ok: next.status < 400, status: next.status, json: async () => next.body };
};
const context = { document, fetch, setTimeout: () => 0, clearTimeout() {}, Option: function (t, v) { this.text = t; this.value = v; }, Math, Number, Array, Float32Array, Uint8ClampedArray, JSON, String, Object, Set, console };
vm.createContext(context);
vm.runInContext(readFileSync(new URL("./app.js", import.meta.url), "utf8"), context);
const byId = (id) => document.getElementById(id);
const flush = () => new Promise((r) => setImmediate(r));

// Simulate a run in progress: state running with one progress point shown.
fetchQueue = [{ status: 200, body: { ok: true, state: "running", run_id: 7, next: 1, events: [{ event: "progress", step: 3, loss: 1.5, samples_seen: 256, elapsed_ms: 40 }] } }];
await context.poll(); await flush();
assert.equal(document.body.dataset.state, "running");
assert.equal(byId("m-step").textContent, "4");

// 1. Reset request fails outright (HTTP 500): view must keep its data and show the failure, then re-poll the real state.
fetchQueue = [{ status: 500, body: { ok: false, error: "Reset refused: the trainer process did not stop" } },
              { status: 200, body: { ok: true, state: "running", run_id: 7, next: 1, events: [] } }];
await byId("reset").handlers.click(); await flush();
assert.equal(byId("m-step").textContent, "4", "results cleared despite failed reset");
assert.equal(document.body.dataset.state, "running", "claimed a state the server did not confirm");
assert.match(byId("verdict").textContent, /reset failed/i);

// 2. Reset "succeeds" but the server reports a non-idle state: still not idle.
fetchQueue = [{ status: 200, body: { ok: true, state: "running", run_id: 7, next: 0, events: [] } },
              { status: 200, body: { ok: true, state: "running", run_id: 7, next: 1, events: [] } }];
await byId("reset").handlers.click(); await flush();
assert.equal(document.body.dataset.state, "running");
assert.equal(byId("m-step").textContent, "4");

// 3. Confirmed reset: idle and cleared.
fetchQueue = [{ status: 200, body: { ok: true, state: "idle", run_id: 8, next: 0, events: [] } },
              { status: 200, body: { ok: true, state: "idle", run_id: 8, next: 0, events: [] } }];
await byId("reset").handlers.click(); await flush();
assert.equal(document.body.dataset.state, "idle");
assert.equal(byId("m-step").textContent, "—");

// 4. Stop failure is surfaced, and polling continues to show the server's state.
fetchQueue = [{ status: 200, body: { ok: true, state: "running", run_id: 9, next: 0, events: [] } }];
await context.poll(); await flush();
fetchQueue = [{ status: 500, body: { ok: false, error: "did not stop" } }, { status: 200, body: { ok: true, state: "running", run_id: 9, next: 0, events: [] } }];
await byId("stop").handlers.click(); await flush();
assert.match(byId("verdict").textContent, /stop failed/i);
assert.equal(document.body.dataset.state, "running");
// 5. Weights never unlock Classify on a failed or stopped run; only a finished run with a passing done does.
fetchQueue = [{ status: 200, body: { ok: true, state: "running", run_id: 11, next: 3, events: [
  { event: "check", passed: true, kind: "learning_signals", parameters: 10, final_vs_initial_max_abs_diff: 0.1 },
  { event: "weights", layout: "softmax", values: new Array(7850).fill(0) },
  { event: "done", checks_passed: false, exit_status: 0 }] } }];
await context.poll(); await flush();
assert.equal(byId("classify").disabled, true, "weights unlocked Classify before the server confirmed the run");
fetchQueue = [{ status: 200, body: { ok: true, state: "failed", run_id: 11, next: 4, events: [{ event: "error", message: "resources leaked" }] } }];
await context.poll(); await flush();
assert.equal(byId("classify").disabled, true, "Classify enabled on a failed run");
assert.equal(document.body.dataset.state, "failed");
fetchQueue = [{ status: 200, body: { ok: true, state: "stopped", run_id: 11, next: 4, events: [] } }];
await context.poll(); await flush();
assert.equal(byId("classify").disabled, true, "Classify enabled on a stopped run");
fetchQueue = [{ status: 200, body: { ok: true, state: "running", run_id: 12, next: 2, events: [
  { event: "weights", layout: "softmax", values: new Array(7850).fill(0) },
  { event: "done", checks_passed: true, exit_status: 0, final_test_accuracy: 0.9, runtime_ms: 10 }] } },
  { status: 200, body: { ok: true, state: "finished", run_id: 12, next: 2, events: [] } }];
await context.poll(); await flush(); await context.poll(); await flush();
assert.equal(document.body.dataset.state, "finished");
assert.equal(byId("classify").disabled, false, "Classify stayed locked on a confirmed finished run");
console.log("app.js control-path checks passed");
