// Softmax regression forward pass. One invocation per minibatch sample.
// Buffer layout (f32 elements): see ../src/lib.rs `Layout`.
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

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let s = id.x;
  let n = u32(d[H_BATCH]);
  if (s >= n) { return; }
  var z: array<f32, 10>;
  var zmax: f32 = -3.0e38;
  for (var j: u32 = 0u; j < CLASSES; j = j + 1u) {
    var acc: f32 = d[OFF_B + j];
    let wrow = OFF_W + j * INPUTS;
    let xrow = OFF_X + s * INPUTS;
    for (var i: u32 = 0u; i < INPUTS; i = i + 1u) {
      acc = acc + d[wrow + i] * d[xrow + i];
    }
    z[j] = acc;
    zmax = max(zmax, acc);
  }
  var sum: f32 = 0.0;
  for (var j: u32 = 0u; j < CLASSES; j = j + 1u) {
    let e = exp(z[j] - zmax);
    z[j] = e;
    sum = sum + e;
  }
  let y = u32(d[OFF_Y + s]);
  for (var j: u32 = 0u; j < CLASSES; j = j + 1u) {
    d[OFF_P + s * CLASSES + j] = z[j] / sum;
  }
  d[OFF_L + s] = -log(max(z[y] / sum, 1.0e-30));
}
