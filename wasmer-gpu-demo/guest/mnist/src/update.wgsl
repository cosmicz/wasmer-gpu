// Softmax regression SGD step. One invocation per parameter (7840 weights
// followed by 10 biases): minibatch-mean gradient (p - onehot(y)) * x, then
// in-place update with the learning rate from the header.
@group(0) @binding(0)
var<storage, read_write> d: array<f32>;

const INPUTS: u32 = 784u;
const CLASSES: u32 = 10u;
const H_BATCH: u32 = 0u;
const H_LR: u32 = 1u;
const MAX_BATCH: u32 = 64u;
const OFF_L: u32 = 16u;
const OFF_P: u32 = OFF_L + MAX_BATCH;
const OFF_W: u32 = OFF_P + MAX_BATCH * CLASSES;
const OFF_B: u32 = OFF_W + INPUTS * CLASSES;
const OFF_X: u32 = OFF_B + CLASSES;
const OFF_Y: u32 = OFF_X + MAX_BATCH * INPUTS;
const PARAMS: u32 = INPUTS * CLASSES + CLASSES;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = id.x;
  if (k >= PARAMS) { return; }
  let n = u32(d[H_BATCH]);
  let lr = d[H_LR];
  let is_bias = k >= INPUTS * CLASSES;
  let j = select(k / INPUTS, k - INPUTS * CLASSES, is_bias);
  let i = select(k % INPUTS, 0u, is_bias);
  var g: f32 = 0.0;
  for (var s: u32 = 0u; s < n; s = s + 1u) {
    let p = d[OFF_P + s * CLASSES + j];
    let t = select(0.0, 1.0, u32(d[OFF_Y + s]) == j);
    let x = select(d[OFF_X + s * INPUTS + i], 1.0, is_bias);
    g = g + (p - t) * x;
  }
  let idx = select(OFF_W + k, OFF_B + j, is_bias);
  d[idx] = d[idx] - lr * g / f32(n);
}
