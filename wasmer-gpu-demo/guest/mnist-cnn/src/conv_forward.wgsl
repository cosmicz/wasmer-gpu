// K1: conv 5x5 valid + ReLU. One invocation per (sample, filter, y, x).
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = u32(d[H_BATCH]);
  let total = n * FILTERS * CONV * CONV;
  let g = id.x;
  if (g >= total) { return; }
  let x = g % CONV;
  let y = (g / CONV) % CONV;
  let f = (g / (CONV * CONV)) % FILTERS;
  let s = g / (CONV * CONV * FILTERS);
  var acc: f32 = d[OFF_B1 + f];
  for (var ky: u32 = 0u; ky < K; ky = ky + 1u) {
    for (var kx: u32 = 0u; kx < K; kx = kx + 1u) {
      acc = acc + d[OFF_W1 + f * K * K + ky * K + kx] * d[OFF_X + s * IMG * IMG + (y + ky) * IMG + (x + kx)];
    }
  }
  d[OFF_CONV + g] = max(acc, 0.0);
}
