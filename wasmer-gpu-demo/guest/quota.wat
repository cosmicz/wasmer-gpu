;; Control: per-instance buffer and pipeline quotas return ERR_QUOTA (-4).
;; Nothing is released, so teardown reclamation is exercised as well.
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

  (func $_start (export "_start") (result i32)
    (local $attempt i32)
    (local $handle i32)
    (local $buffer_quota_hit i32)
    (block $uploads_done
      (loop $upload
        (local.set $handle (call $buffer_upload (i32.const 0) (i32.const 16)))
        (if (i32.eq (local.get $handle) (i32.const -4))
          (then
            (local.set $buffer_quota_hit (i32.const 1))
            (br $uploads_done)
          )
        )
        (if (i32.lt_s (local.get $handle) (i32.const 0))
          (then (return (i32.const 1)))
        )
        (local.set $attempt (i32.add (local.get $attempt) (i32.const 1)))
        (br_if $upload (i32.lt_u (local.get $attempt) (i32.const 16)))
      )
    )
    (if (i32.eqz (local.get $buffer_quota_hit))
      (then (return (i32.const 2)))
    )
    (local.set $attempt (i32.const 0))
    (loop $create
      (local.set $handle (call $pipeline_create (i32.const 4096) (i32.const 554)))
      (if (i32.eq (local.get $handle) (i32.const -4))
        (then (return (i32.const 0)))
      )
      (if (i32.lt_s (local.get $handle) (i32.const 0))
        (then (return (i32.const 3)))
      )
      (local.set $attempt (i32.add (local.get $attempt) (i32.const 1)))
      (br_if $create (i32.lt_u (local.get $attempt) (i32.const 8)))
    )
    (i32.const 4)
  )
)
