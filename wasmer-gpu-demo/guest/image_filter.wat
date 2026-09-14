;; WASIX command guest for the image workbench. It carries every kernel it
;; runs as NUL-terminated WGSL text in its own data segments; the host holds no
;; shader. Memory layout: shader slots of 4096 bytes at 0, 4096, 8192, 12288
;; (blur, edges, emboss, identity), image region at 16384:
;;   [width u32][height u32][filter u32][param u32] | input RGBA8 | output RGBA8
;; The host writes the header and input before _start; the guest uploads that
;; whole region as one storage buffer, creates the selected kernel, dispatches
;; ceil(w/8) x ceil(h/8) workgroups of 8x8, reads the buffer back in place and
;; releases both handles. Filter 255 skips the GPU entirely (no-dispatch
;; control). Exit code: 0 on success, 10 + failing step otherwise, 7 for an
;; unknown filter id.
(module
  (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
  (import "wasmer_gpu_v0" "buffer_upload" (func $buffer_upload (param i32 i32) (result i32)))
  (import "wasmer_gpu_v0" "buffer_read" (func $buffer_read (param i32 i32 i32) (result i32)))
  (import "wasmer_gpu_v0" "buffer_release" (func $buffer_release (param i32) (result i32)))
  (import "wasmer_gpu_v0" "pipeline_create" (func $pipeline_create (param i32 i32) (result i32)))
  (import "wasmer_gpu_v0" "pipeline_dispatch" (func $pipeline_dispatch (param i32 i32 i32 i32 i32) (result i32)))
  (import "wasmer_gpu_v0" "pipeline_release" (func $pipeline_release (param i32) (result i32)))
  ;; 48 pages = 3 MiB: shader slots + 16 B header + two 512x512 RGBA8 images.
  (memory (export "memory") 48 48)

  ;; slot 0: blur (1688 bytes + NUL)
  (data (i32.const 0)
    "@group(0) @binding(0) var<storage, read_write> buf: array<u32>;\n"
    "// Layout in u32 words: [width, height, filter, param] | input RGBA8 | output RGBA8.\n"
    "fn px(x: i32, y: i32) -> vec3<i32> {\n"
    "  let w = i32(buf[0]);\n"
    "  let h = i32(buf[1]);\n"
    "  let cx = clamp(x, 0, w - 1);\n"
    "  let cy = clamp(y, 0, h - 1);\n"
    "  let p = buf[4u + u32(cy * w + cx)];\n"
    "  return vec3<i32>(i32(p & 255u), i32((p >> 8u) & 255u), i32((p >> 16u) & 255u));\n"
    "}\n"
    "fn lum(x: i32, y: i32) -> i32 {\n"
    "  let c = px(x, y);\n"
    "  return (c.r * 77 + c.g * 151 + c.b * 28) >> 8u;\n"
    "}\n"
    "// Filters act on RGB; the source pixel's alpha is carried through unchanged.\n"
    "fn put(x: u32, y: u32, c: vec3<i32>) {\n"
    "  let w = buf[0];\n"
    "  let h = buf[1];\n"
    "  let k = clamp(c, vec3<i32>(0), vec3<i32>(255));\n"
    "  let alpha = buf[4u + y * w + x] & 0xff000000u;\n"
    "  buf[4u + w * h + y * w + x] = u32(k.r) | (u32(k.g) << 8u) | (u32(k.b) << 16u) | alpha;\n"
    "}\n"
    "// Binomial blur of radius r = param (1..5): kernel 2r+1 per axis, weight\n"
    "// C(2r, r+d) at offset d, exact normalisation by 4^(2r).\n"
    "fn binomial(n: u32, k: u32) -> i32 {\n"
    "  var c: u32 = 1u;\n"
    "  for (var i = 0u; i < k; i++) {\n"
    "    c = c * (n - i) / (i + 1u);\n"
    "  }\n"
    "  return i32(c);\n"
    "}\n"
    "@compute @workgroup_size(8, 8)\n"
    "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
    "  if (id.x >= buf[0] || id.y >= buf[1]) { return; }\n"
    "  let x = i32(id.x);\n"
    "  let y = i32(id.y);\n"
    "  let r = i32(clamp(buf[3], 1u, 5u));\n"
    "  var acc = vec3<i32>(0);\n"
    "  for (var j = -r; j <= r; j++) {\n"
    "    let wj = binomial(u32(2 * r), u32(r + j));\n"
    "    for (var i = -r; i <= r; i++) {\n"
    "      acc += px(x + i, y + j) * (binomial(u32(2 * r), u32(r + i)) * wj);\n"
    "    }\n"
    "  }\n"
    "  let norm = i32(1u << u32(4 * r));\n"
    "  put(id.x, id.y, (acc + vec3<i32>(norm / 2)) / vec3<i32>(norm));\n"
    "}\n"
    "\00"
  )

  ;; slot 1: edges (1488 bytes + NUL)
  (data (i32.const 4096)
    "@group(0) @binding(0) var<storage, read_write> buf: array<u32>;\n"
    "// Layout in u32 words: [width, height, filter, param] | input RGBA8 | output RGBA8.\n"
    "fn px(x: i32, y: i32) -> vec3<i32> {\n"
    "  let w = i32(buf[0]);\n"
    "  let h = i32(buf[1]);\n"
    "  let cx = clamp(x, 0, w - 1);\n"
    "  let cy = clamp(y, 0, h - 1);\n"
    "  let p = buf[4u + u32(cy * w + cx)];\n"
    "  return vec3<i32>(i32(p & 255u), i32((p >> 8u) & 255u), i32((p >> 16u) & 255u));\n"
    "}\n"
    "fn lum(x: i32, y: i32) -> i32 {\n"
    "  let c = px(x, y);\n"
    "  return (c.r * 77 + c.g * 151 + c.b * 28) >> 8u;\n"
    "}\n"
    "// Filters act on RGB; the source pixel's alpha is carried through unchanged.\n"
    "fn put(x: u32, y: u32, c: vec3<i32>) {\n"
    "  let w = buf[0];\n"
    "  let h = buf[1];\n"
    "  let k = clamp(c, vec3<i32>(0), vec3<i32>(255));\n"
    "  let alpha = buf[4u + y * w + x] & 0xff000000u;\n"
    "  buf[4u + w * h + y * w + x] = u32(k.r) | (u32(k.g) << 8u) | (u32(k.b) << 16u) | alpha;\n"
    "}\n"
    "// Sobel gradient magnitude (L1) on luminance, scaled by param / 8.\n"
    "@compute @workgroup_size(8, 8)\n"
    "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
    "  if (id.x >= buf[0] || id.y >= buf[1]) { return; }\n"
    "  let x = i32(id.x);\n"
    "  let y = i32(id.y);\n"
    "  let gx = -lum(x - 1, y - 1) - 2 * lum(x - 1, y) - lum(x - 1, y + 1)\n"
    "         + lum(x + 1, y - 1) + 2 * lum(x + 1, y) + lum(x + 1, y + 1);\n"
    "  let gy = -lum(x - 1, y - 1) - 2 * lum(x, y - 1) - lum(x + 1, y - 1)\n"
    "         + lum(x - 1, y + 1) + 2 * lum(x, y + 1) + lum(x + 1, y + 1);\n"
    "  let m = min(255, ((abs(gx) + abs(gy)) * i32(buf[3])) / 8);\n"
    "  put(id.x, id.y, vec3<i32>(m));\n"
    "}\n"
    "\00"
  )

  ;; slot 2: emboss (1320 bytes + NUL)
  (data (i32.const 8192)
    "@group(0) @binding(0) var<storage, read_write> buf: array<u32>;\n"
    "// Layout in u32 words: [width, height, filter, param] | input RGBA8 | output RGBA8.\n"
    "fn px(x: i32, y: i32) -> vec3<i32> {\n"
    "  let w = i32(buf[0]);\n"
    "  let h = i32(buf[1]);\n"
    "  let cx = clamp(x, 0, w - 1);\n"
    "  let cy = clamp(y, 0, h - 1);\n"
    "  let p = buf[4u + u32(cy * w + cx)];\n"
    "  return vec3<i32>(i32(p & 255u), i32((p >> 8u) & 255u), i32((p >> 16u) & 255u));\n"
    "}\n"
    "fn lum(x: i32, y: i32) -> i32 {\n"
    "  let c = px(x, y);\n"
    "  return (c.r * 77 + c.g * 151 + c.b * 28) >> 8u;\n"
    "}\n"
    "// Filters act on RGB; the source pixel's alpha is carried through unchanged.\n"
    "fn put(x: u32, y: u32, c: vec3<i32>) {\n"
    "  let w = buf[0];\n"
    "  let h = buf[1];\n"
    "  let k = clamp(c, vec3<i32>(0), vec3<i32>(255));\n"
    "  let alpha = buf[4u + y * w + x] & 0xff000000u;\n"
    "  buf[4u + w * h + y * w + x] = u32(k.r) | (u32(k.g) << 8u) | (u32(k.b) << 16u) | alpha;\n"
    "}\n"
    "// Emboss: diagonal relief kernel on luminance around mid grey, scaled by param / 2.\n"
    "@compute @workgroup_size(8, 8)\n"
    "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
    "  if (id.x >= buf[0] || id.y >= buf[1]) { return; }\n"
    "  let x = i32(id.x);\n"
    "  let y = i32(id.y);\n"
    "  let e = -2 * lum(x - 1, y - 1) - lum(x, y - 1) - lum(x - 1, y)\n"
    "        + lum(x + 1, y) + lum(x, y + 1) + 2 * lum(x + 1, y + 1);\n"
    "  put(id.x, id.y, vec3<i32>(128 + (e * i32(buf[3])) / 2));\n"
    "}\n"
    "\00"
  )

  ;; slot 3: identity (1142 bytes + NUL)
  (data (i32.const 12288)
    "@group(0) @binding(0) var<storage, read_write> buf: array<u32>;\n"
    "// Layout in u32 words: [width, height, filter, param] | input RGBA8 | output RGBA8.\n"
    "fn px(x: i32, y: i32) -> vec3<i32> {\n"
    "  let w = i32(buf[0]);\n"
    "  let h = i32(buf[1]);\n"
    "  let cx = clamp(x, 0, w - 1);\n"
    "  let cy = clamp(y, 0, h - 1);\n"
    "  let p = buf[4u + u32(cy * w + cx)];\n"
    "  return vec3<i32>(i32(p & 255u), i32((p >> 8u) & 255u), i32((p >> 16u) & 255u));\n"
    "}\n"
    "fn lum(x: i32, y: i32) -> i32 {\n"
    "  let c = px(x, y);\n"
    "  return (c.r * 77 + c.g * 151 + c.b * 28) >> 8u;\n"
    "}\n"
    "// Filters act on RGB; the source pixel's alpha is carried through unchanged.\n"
    "fn put(x: u32, y: u32, c: vec3<i32>) {\n"
    "  let w = buf[0];\n"
    "  let h = buf[1];\n"
    "  let k = clamp(c, vec3<i32>(0), vec3<i32>(255));\n"
    "  let alpha = buf[4u + y * w + x] & 0xff000000u;\n"
    "  buf[4u + w * h + y * w + x] = u32(k.r) | (u32(k.g) << 8u) | (u32(k.b) << 16u) | alpha;\n"
    "}\n"
    "// Identity copy: the wrong-shader control for every other filter.\n"
    "@compute @workgroup_size(8, 8)\n"
    "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
    "  if (id.x >= buf[0] || id.y >= buf[1]) { return; }\n"
    "  let x = i32(id.x);\n"
    "  let y = i32(id.y);\n"
    "  put(id.x, id.y, px(x, y));\n"
    "}\n"
    "\00"
  )

  (func $fail (param $step i32)
    (call $proc_exit (i32.add (i32.const 10) (local.get $step)))
  )

  (func $strlen (param $p i32) (result i32)
    (local $n i32)
    (block $done
      (loop $scan
        (br_if $done (i32.eqz (i32.load8_u (i32.add (local.get $p) (local.get $n)))))
        (local.set $n (i32.add (local.get $n) (i32.const 1)))
        (br $scan)
      )
    )
    (local.get $n)
  )

  (func $_start (export "_start")
    (local $w i32)
    (local $h i32)
    (local $filter i32)
    (local $bytes i32)
    (local $shader i32)
    (local $buffer i32)
    (local $pipeline i32)
    (local.set $w (i32.load (i32.const 16384)))
    (local.set $h (i32.load (i32.const 16388)))
    (local.set $filter (i32.load (i32.const 16392)))
    ;; No-dispatch control: leave the output region untouched.
    (if (i32.eq (local.get $filter) (i32.const 255))
      (then (call $proc_exit (i32.const 0)))
    )
    (if (i32.gt_u (local.get $filter) (i32.const 3))
      (then (call $proc_exit (i32.const 7)))
    )
    ;; bytes = 16 + 8 * w * h  (header, input, output)
    (local.set $bytes
      (i32.add (i32.const 16)
        (i32.mul (i32.const 8) (i32.mul (local.get $w) (local.get $h)))))
    (local.set $shader (i32.mul (local.get $filter) (i32.const 4096)))
    (local.set $buffer (call $buffer_upload (i32.const 16384) (local.get $bytes)))
    (if (i32.lt_s (local.get $buffer) (i32.const 0))
      (then (call $fail (i32.const 1)))
    )
    (local.set $pipeline
      (call $pipeline_create (local.get $shader) (call $strlen (local.get $shader))))
    (if (i32.lt_s (local.get $pipeline) (i32.const 0))
      (then (call $fail (i32.const 2)))
    )
    (if (i32.lt_s
          (call $pipeline_dispatch
            (local.get $pipeline) (local.get $buffer)
            (i32.div_u (i32.add (local.get $w) (i32.const 7)) (i32.const 8))
            (i32.div_u (i32.add (local.get $h) (i32.const 7)) (i32.const 8))
            (i32.const 1))
          (i32.const 0))
      (then (call $fail (i32.const 3)))
    )
    (if (i32.lt_s (call $buffer_read (local.get $buffer) (i32.const 16384) (local.get $bytes)) (i32.const 0))
      (then (call $fail (i32.const 4)))
    )
    (if (i32.lt_s (call $pipeline_release (local.get $pipeline)) (i32.const 0))
      (then (call $fail (i32.const 5)))
    )
    (if (i32.lt_s (call $buffer_release (local.get $buffer)) (i32.const 0))
      (then (call $fail (i32.const 6)))
    )
    (call $proc_exit (i32.const 0))
  )
)
