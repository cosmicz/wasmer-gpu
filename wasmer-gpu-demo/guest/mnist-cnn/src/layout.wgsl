// Shared layout for the CNN kernels (f32 elements in one storage buffer).
// header(16) | loss(64) | probs(64x10) | W2(10x1152) | b2(10) | W1(8x25) | b1(8)
// | X(64x784) | y(64) | conv(64x8x24x24) | pooled(64x8x12x12) | argmax(same)
// | dpooled(same) | dconv(64x8x24x24)
@group(0) @binding(0)
var<storage, read_write> d: array<f32>;

const H_BATCH: u32 = 0u;
const H_LR: u32 = 1u;
const MAX_BATCH: u32 = 64u;
const CLASSES: u32 = 10u;
const IMG: u32 = 28u;
const K: u32 = 5u;
const FILTERS: u32 = 8u;
const CONV: u32 = 24u;
const POOL: u32 = 12u;
const FEAT: u32 = FILTERS * POOL * POOL; // 1152
const OFF_L: u32 = 16u;
const OFF_P: u32 = OFF_L + MAX_BATCH;
const OFF_W2: u32 = OFF_P + MAX_BATCH * CLASSES;
const OFF_B2: u32 = OFF_W2 + CLASSES * FEAT;
const OFF_W1: u32 = OFF_B2 + CLASSES;
const OFF_B1: u32 = OFF_W1 + FILTERS * K * K;
const OFF_X: u32 = OFF_B1 + FILTERS;
const OFF_Y: u32 = OFF_X + MAX_BATCH * IMG * IMG;
const OFF_CONV: u32 = OFF_Y + MAX_BATCH;
const OFF_POOLED: u32 = OFF_CONV + MAX_BATCH * FILTERS * CONV * CONV;
const OFF_ARGMAX: u32 = OFF_POOLED + MAX_BATCH * FEAT;
const OFF_DPOOLED: u32 = OFF_ARGMAX + MAX_BATCH * FEAT;
const OFF_DCONV: u32 = OFF_DPOOLED + MAX_BATCH * FEAT;
