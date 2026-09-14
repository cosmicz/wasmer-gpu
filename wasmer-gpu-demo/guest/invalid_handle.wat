;; Control: unknown, foreign-instance, released and wrong-kind handles must
;; return ERR_HANDLE (-2).
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

  (func $expect (param $actual i32) (param $expected i32) (param $check i32) (result i32)
    (if (i32.ne (local.get $actual) (local.get $expected))
      (then (return (local.get $check)))
    )
    (i32.const 0)
  )

  (func $_start (export "_start") (result i32)
    (local $buffer i32)
    (local $buffer2 i32)
    (local $pipeline i32)
    (local $r i32)
    ;; never-issued handles
    (local.set $r (call $expect (call $pipeline_dispatch (i32.const 9999) (i32.const 9998) (i32.const 1) (i32.const 1) (i32.const 1)) (i32.const -2) (i32.const 1)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $r (call $expect (call $buffer_read (i32.const 9999) (i32.const 16) (i32.const 16)) (i32.const -2) (i32.const 2)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $r (call $expect (call $buffer_release (i32.const 9999)) (i32.const -2) (i32.const 3)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $r (call $expect (call $pipeline_release (i32.const 9999)) (i32.const -2) (i32.const 4)))
    (if (local.get $r) (then (return (local.get $r))))
    ;; cross-instance reuse: handles 1 and 2 were issued to an earlier guest
    ;; instance in this process (the bridge counter never restarts) and must
    ;; be foreign to this session, whether that earlier instance released them
    ;; or not
    (local.set $r (call $expect (call $pipeline_dispatch (i32.const 2) (i32.const 1) (i32.const 1) (i32.const 1) (i32.const 1)) (i32.const -2) (i32.const 20)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $r (call $expect (call $buffer_read (i32.const 1) (i32.const 16) (i32.const 16)) (i32.const -2) (i32.const 21)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $r (call $expect (call $buffer_release (i32.const 1)) (i32.const -2) (i32.const 22)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $r (call $expect (call $pipeline_release (i32.const 2)) (i32.const -2) (i32.const 23)))
    (if (local.get $r) (then (return (local.get $r))))
    ;; real handles of each kind
    (local.set $buffer (call $buffer_upload (i32.const 0) (i32.const 16)))
    (if (i32.lt_s (local.get $buffer) (i32.const 0))
      (then (return (i32.const 5)))
    )
    (local.set $pipeline (call $pipeline_create (i32.const 4096) (i32.const 554)))
    (if (i32.lt_s (local.get $pipeline) (i32.const 0))
      (then (return (i32.const 6)))
    )
    ;; kind confusion: buffer used as pipeline and vice versa
    (local.set $r (call $expect (call $pipeline_dispatch (local.get $buffer) (local.get $pipeline) (i32.const 1) (i32.const 1) (i32.const 1)) (i32.const -2) (i32.const 7)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $r (call $expect (call $buffer_release (local.get $pipeline)) (i32.const -2) (i32.const 8)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $r (call $expect (call $pipeline_release (local.get $buffer)) (i32.const -2) (i32.const 9)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $r (call $expect (call $buffer_read (local.get $pipeline) (i32.const 16) (i32.const 16)) (i32.const -2) (i32.const 10)))
    (if (local.get $r) (then (return (local.get $r))))
    ;; use after release, double release
    (local.set $r (call $expect (call $buffer_release (local.get $buffer)) (i32.const 0) (i32.const 11)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $r (call $expect (call $buffer_release (local.get $buffer)) (i32.const -2) (i32.const 12)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $r (call $expect (call $pipeline_dispatch (local.get $pipeline) (local.get $buffer) (i32.const 1) (i32.const 1) (i32.const 1)) (i32.const -2) (i32.const 13)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $r (call $expect (call $buffer_read (local.get $buffer) (i32.const 16) (i32.const 16)) (i32.const -2) (i32.const 14)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $r (call $expect (call $pipeline_release (local.get $pipeline)) (i32.const 0) (i32.const 15)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $r (call $expect (call $pipeline_release (local.get $pipeline)) (i32.const -2) (i32.const 16)))
    (if (local.get $r) (then (return (local.get $r))))
    (local.set $buffer2 (call $buffer_upload (i32.const 0) (i32.const 16)))
    (if (i32.lt_s (local.get $buffer2) (i32.const 0))
      (then (return (i32.const 17)))
    )
    (local.set $r (call $expect (call $pipeline_dispatch (local.get $pipeline) (local.get $buffer2) (i32.const 1) (i32.const 1) (i32.const 1)) (i32.const -2) (i32.const 18)))
    (if (local.get $r) (then (return (local.get $r))))
    (call $expect (call $buffer_release (local.get $buffer2)) (i32.const 0) (i32.const 19))
  )
)
