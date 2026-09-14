;; Control: rejected shader text returns ERR_SHADER (-5) and leaves the host
;; alive; a valid kernel still runs afterwards in the same instance.
;; Returns 0 on success, otherwise the number of the failed check.
(module
  (import "env" "memory" (memory 1 1))
  (import "wasmer_gpu_v0" "buffer_upload" (func $buffer_upload (param i32 i32) (result i32)))
  (import "wasmer_gpu_v0" "buffer_read" (func $buffer_read (param i32 i32 i32) (result i32)))
  (import "wasmer_gpu_v0" "buffer_release" (func $buffer_release (param i32) (result i32)))
  (import "wasmer_gpu_v0" "pipeline_create" (func $pipeline_create (param i32 i32) (result i32)))
  (import "wasmer_gpu_v0" "pipeline_dispatch" (func $pipeline_dispatch (param i32 i32 i32 i32 i32) (result i32)))
  (import "wasmer_gpu_v0" "pipeline_release" (func $pipeline_release (param i32) (result i32)))

  (data (i32.const 4096)
    "@group(0) @binding(0)\n"
    "var<storage, read_write> values: array<u32>;\n"
    "\n"
    "fn collatz_iterations(n_base: u32) -> u32 {\n"
    "  var n: u32 = n_base;\n"
    "  var i: u32 = 0u;\n"
    "  loop {\n"
    "    if (n <= 1u) { break; }\n"
    "    if (n % 2u == 0u) {\n"
    "      n = n / 2u;\n"
    "    } else {\n"
    "      if (n >= 1431655765u) { return 4294967295u; }\n"
    "      n = 3u * n + 1u;\n"
    "    }\n"
    "    i = i + 1u;\n"
    "  }\n"
    "  return i;\n"
    "}\n"
    "\n"
    "@compute @workgroup_size(1)\n"
    "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
    "  let i = id.x;\n"
    "  if (i >= arrayLength(&values)) { return; }\n"
    "  values[i] = collatz_iterations(values[i]);\n"
    "}\n"
  )
  (data (i32.const 12288)
    "@compute @workgroup_size(1) fn main( { this is not wgsl\n"
  )
  (data (i32.const 13312)
    "@group(0) @binding(0)\n"
    "var<storage, read_write> values: array<u32>;\n"
    "\n"
    "@compute @workgroup_size(1)\n"
    "fn other(@builtin(global_invocation_id) id: vec3<u32>) {\n"
    "  values[id.x] = 0u;\n"
    "}\n"
  )
  (data (i32.const 14336)
    "@group(0) @binding(0)\n"
    "var<uniform> scale: u32;\n"
    "\n"
    "@compute @workgroup_size(1)\n"
    "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
    "  let unused = scale;\n"
    "}\n"
  )

  (func $expect (param $actual i32) (param $expected i32) (param $check i32) (result i32)
    (if (i32.ne (local.get $actual) (local.get $expected))
      (then (return (local.get $check)))
    )
    (i32.const 0)
  )

  (func $_start (export "_start") (result i32)
    (local $buffer i32)
    (local $pipeline i32)
    (local $r i32)
    (local.set $r (call $expect (call $pipeline_create (i32.const 12288) (i32.const 56)) (i32.const -5) (i32.const 1)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $r (call $expect (call $pipeline_create (i32.const 13312) (i32.const 176)) (i32.const -5) (i32.const 2)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $r (call $expect (call $pipeline_create (i32.const 14336) (i32.const 156)) (i32.const -5) (i32.const 3)))
    (if (local.get $r) (then (return (local.get $r))))
    ;; binary input bytes are not a shader either
    (local.set $r (call $expect (call $pipeline_create (i32.const 0) (i32.const 16)) (i32.const -5) (i32.const 4)))
    (if (local.get $r) (then (return (local.get $r))))
    ;; the valid kernel still works in this instance
    (local.set $buffer (call $buffer_upload (i32.const 0) (i32.const 16)))
    (if (i32.lt_s (local.get $buffer) (i32.const 0))
      (then (return (i32.const 5)))
    )
    (local.set $pipeline (call $pipeline_create (i32.const 4096) (i32.const 554)))
    (if (i32.lt_s (local.get $pipeline) (i32.const 0))
      (then (return (i32.const 6)))
    )
    (local.set $r (call $expect (call $pipeline_dispatch (local.get $pipeline) (local.get $buffer) (i32.const 4) (i32.const 1) (i32.const 1)) (i32.const 0) (i32.const 7)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $r (call $expect (call $buffer_read (local.get $buffer) (i32.const 16) (i32.const 16)) (i32.const 0) (i32.const 8)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $r (call $expect (call $pipeline_release (local.get $pipeline)) (i32.const 0) (i32.const 9)))
    (if (local.get $r) (then (return (local.get $r))))
    (call $expect (call $buffer_release (local.get $buffer)) (i32.const 0) (i32.const 10))
  )
)
