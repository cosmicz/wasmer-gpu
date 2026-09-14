// K3: fully connected 1152 -> 10, softmax, cross-entropy. One invocation per sample.
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let s = id.x;
  let n = u32(d[H_BATCH]);
  if (s >= n) { return; }
  var z: array<f32, 10>;
  var zmax: f32 = -3.0e38;
  for (var j: u32 = 0u; j < CLASSES; j = j + 1u) {
    var acc: f32 = d[OFF_B2 + j];
    for (var i: u32 = 0u; i < FEAT; i = i + 1u) {
      acc = acc + d[OFF_W2 + j * FEAT + i] * d[OFF_POOLED + s * FEAT + i];
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
