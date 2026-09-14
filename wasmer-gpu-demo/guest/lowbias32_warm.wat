;; Kernel guest: lowbias32 integer hash applied 8 times in place, so the
;; host records one cold and seven warm dispatch samples on one pipeline.
;; Input: 8 u32 at offset 0. Output: 8 u32 at offset 32.
(module
  (import "env" "memory" (memory 1 1))
  (import "wasmer_gpu_v0" "buffer_upload" (func $buffer_upload (param i32 i32) (result i32)))
  (import "wasmer_gpu_v0" "buffer_read" (func $buffer_read (param i32 i32 i32) (result i32)))
  (import "wasmer_gpu_v0" "buffer_release" (func $buffer_release (param i32) (result i32)))
  (import "wasmer_gpu_v0" "pipeline_create" (func $pipeline_create (param i32 i32) (result i32)))
  (import "wasmer_gpu_v0" "pipeline_dispatch" (func $pipeline_dispatch (param i32 i32 i32 i32 i32) (result i32)))
  (import "wasmer_gpu_v0" "pipeline_release" (func $pipeline_release (param i32) (result i32)))

  (data (i32.const 8192)
    "@group(0) @binding(0)\n"
    "var<storage, read_write> values: array<u32>;\n"
    "\n"
    "fn lowbias32(x_in: u32) -> u32 {\n"
    "  var x: u32 = x_in;\n"
    "  x = x ^ (x >> 16u);\n"
    "  x = x * 0x7feb352du;\n"
    "  x = x ^ (x >> 15u);\n"
    "  x = x * 0x846ca68bu;\n"
    "  x = x ^ (x >> 16u);\n"
    "  return x;\n"
    "}\n"
    "\n"
    "@compute @workgroup_size(1)\n"
    "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
    "  let i = id.x;\n"
    "  if (i >= arrayLength(&values)) { return; }\n"
    "  values[i] = lowbias32(values[i]);\n"
    "}\n"
  )

  (func $_start (export "_start") (result i32)
    (local $buffer i32)
    (local $pipeline i32)
    (local $status i32)
    (local $round i32)
    (local.set $buffer (call $buffer_upload (i32.const 0) (i32.const 32)))
    (if (i32.lt_s (local.get $buffer) (i32.const 0))
      (then (return (local.get $buffer)))
    )
    (local.set $pipeline (call $pipeline_create (i32.const 8192) (i32.const 432)))
    (if (i32.lt_s (local.get $pipeline) (i32.const 0))
      (then (return (local.get $pipeline)))
    )
    (loop $rounds
      (local.set $status (call $pipeline_dispatch (local.get $pipeline) (local.get $buffer) (i32.const 8) (i32.const 1) (i32.const 1)))
      (if (i32.lt_s (local.get $status) (i32.const 0))
        (then (return (local.get $status)))
      )
      (local.set $round (i32.add (local.get $round) (i32.const 1)))
      (br_if $rounds (i32.lt_u (local.get $round) (i32.const 8)))
    )
    (local.set $status (call $buffer_read (local.get $buffer) (i32.const 32) (i32.const 32)))
    (if (i32.lt_s (local.get $status) (i32.const 0))
      (then (return (local.get $status)))
    )
    (local.set $status (call $pipeline_release (local.get $pipeline)))
    (if (i32.lt_s (local.get $status) (i32.const 0))
      (then (return (local.get $status)))
    )
    (call $buffer_release (local.get $buffer))
  )
)
