// K6: SGD step for the conv layer. One invocation per conv parameter
// (8 filters x 25 weights, then 8 biases); each sums dconv over the batch
// and every output position.
@compute @workgroup_size(8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = id.x;
  let params = FILTERS * K * K + FILTERS;
  if (k >= params) { return; }
  let n = u32(d[H_BATCH]);
  let lr = d[H_LR];
  let is_bias = k >= FILTERS * K * K;
  let f = select(k / (K * K), k - FILTERS * K * K, is_bias);
  let ky = select((k % (K * K)) / K, 0u, is_bias);
  let kx = select(k % K, 0u, is_bias);
  var g: f32 = 0.0;
  for (var s: u32 = 0u; s < n; s = s + 1u) {
    let dbase = OFF_DCONV + (s * FILTERS + f) * CONV * CONV;
    let xbase = OFF_X + s * IMG * IMG;
    for (var y: u32 = 0u; y < CONV; y = y + 1u) {
      for (var x: u32 = 0u; x < CONV; x = x + 1u) {
        let dc = d[dbase + y * CONV + x];
        let xv = select(d[xbase + (y + ky) * IMG + (x + kx)], 1.0, is_bias);
        g = g + dc * xv;
      }
    }
  }
  let idx = select(OFF_W1 + k, OFF_B1 + f, is_bias);
  d[idx] = d[idx] - lr * g / f32(n);
}
