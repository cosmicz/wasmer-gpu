// K5: SGD step for the fully connected layer. One invocation per parameter.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = id.x;
  let params = CLASSES * FEAT + CLASSES;
  if (k >= params) { return; }
  let n = u32(d[H_BATCH]);
  let lr = d[H_LR];
  let is_bias = k >= CLASSES * FEAT;
  let j = select(k / FEAT, k - CLASSES * FEAT, is_bias);
  let i = select(k % FEAT, 0u, is_bias);
  var g: f32 = 0.0;
  for (var s: u32 = 0u; s < n; s = s + 1u) {
    let delta = d[OFF_P + s * CLASSES + j] - select(0.0, 1.0, u32(d[OFF_Y + s]) == j);
    let a = select(d[OFF_POOLED + s * FEAT + i], 1.0, is_bias);
    g = g + delta * a;
  }
  let idx = select(OFF_W2 + k, OFF_B2 + j, is_bias);
  d[idx] = d[idx] - lr * g / f32(n);
}
