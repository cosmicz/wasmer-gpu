// K4: gradient w.r.t. pooled features using the pre-update W2, and dconv
// routed through the pool argmax and the ReLU mask. One invocation per
// pooled cell (sample, filter, py, px).
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = u32(d[H_BATCH]);
  let total = n * FEAT;
  let g = id.x;
  if (g >= total) { return; }
  let i = g % FEAT;
  let s = g / FEAT;
  let y = u32(d[OFF_Y + s]);
  var grad: f32 = 0.0;
  for (var j: u32 = 0u; j < CLASSES; j = j + 1u) {
    let delta = d[OFF_P + s * CLASSES + j] - select(0.0, 1.0, j == y);
    grad = grad + delta * d[OFF_W2 + j * FEAT + i];
  }
  d[OFF_DPOOLED + g] = grad;
  // route to the argmax position of this pool window; the other three get 0
  let px = i % POOL;
  let py = (i / POOL) % POOL;
  let f = i / (POOL * POOL);
  let base = OFF_DCONV + ((s * FILTERS + f) * CONV + py * 2u) * CONV + px * 2u;
  let arg = u32(d[OFF_ARGMAX + g]);
  let cbase = OFF_CONV + ((s * FILTERS + f) * CONV + py * 2u) * CONV + px * 2u;
  d[base] = select(0.0, grad, (arg == 0u) && (d[cbase] > 0.0));
  d[base + 1u] = select(0.0, grad, (arg == 1u) && (d[cbase + 1u] > 0.0));
  d[base + CONV] = select(0.0, grad, (arg == 2u) && (d[cbase + CONV] > 0.0));
  d[base + CONV + 1u] = select(0.0, grad, (arg == 3u) && (d[cbase + CONV + 1u] > 0.0));
}
