// K2: 2x2 max pool with argmax bookkeeping. One invocation per pooled cell.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = u32(d[H_BATCH]);
  let total = n * FEAT;
  let g = id.x;
  if (g >= total) { return; }
  let px = g % POOL;
  let py = (g / POOL) % POOL;
  let f = (g / (POOL * POOL)) % FILTERS;
  let s = g / FEAT;
  let base = OFF_CONV + ((s * FILTERS + f) * CONV + py * 2u) * CONV + px * 2u;
  var best: f32 = d[base];
  var arg: u32 = 0u;
  let c1 = d[base + 1u];
  if (c1 > best) { best = c1; arg = 1u; }
  let c2 = d[base + CONV];
  if (c2 > best) { best = c2; arg = 2u; }
  let c3 = d[base + CONV + 1u];
  if (c3 > best) { best = c3; arg = 3u; }
  d[OFF_POOLED + g] = best;
  d[OFF_ARGMAX + g] = f32(arg);
}
