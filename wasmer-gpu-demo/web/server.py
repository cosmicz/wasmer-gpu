"""Loopback-only browser control for a prebuilt, fixed Wasmer GPU smoke."""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import urlsplit
import uuid

WEB_ROOT = Path(__file__).resolve().parent
# Report schema of gpu-smoke at 75e386d: every row carries its guest runtime.
KERNELS = ("collatz_valid.wat", "lowbias32_warm.wat", "wasix_collatz.wat")
CONTROLS = (
    "invalid_range.wat", "invalid_handle.wat", "invalid_shader.wat",
    "unaligned_read.wat", "quota.wat", "leak.wat", "early_exit.wat", "trap_after_create.wat",
    "wasix_leak.wat",
)
TRAPPING_CONTROLS = ("trap_after_create.wat",)


def expected_runtime(guest):
    return "wasix" if guest.startswith("wasix_") else "core"


# Host wall-clock fields the page displays. Each must be a bounded, non-negative
# integer: JSON floats, NaN/Infinity (which json.loads accepts), booleans and
# negatives are all refused rather than shown as "Not reported" on a verified run.
KERNEL_TIMING_FIELDS = ("cpu_reference_ns", "upload_us", "pipeline_create_us", "readback_us")
MAX_TIMING = 10 ** 12


def timing_ok(value):
    return type(value) is int and 0 <= value <= MAX_TIMING


class ProbeError(Exception):
    def __init__(self, message, status=502):
        super().__init__(message)
        self.status = status


def validate_report(report):
    if not isinstance(report, dict):
        raise ProbeError("The executable did not return a report object.")
    for key in ("all_kernels_match", "all_controls_pass", "resources_reclaimed"):
        if report.get(key) is not True:
            raise ProbeError(f"Smoke check is failed or missing: {key}")
    for key in ("adapter", "backend"):
        if not isinstance(report.get(key), str) or not report[key].strip():
            raise ProbeError(f"Report is missing its {key}.")
    if not timing_ok(report.get("setup_us")):
        raise ProbeError("Report timing is missing or invalid: setup_us")
    kernels = checked_rows(report, "kernels", set(KERNELS))
    for kernel in kernels:
        if kernel.get("matches") is not True or type(kernel.get("status")) is not int or kernel["status"] != 0:
            raise ProbeError("A Wasmer kernel did not succeed.")
        for key in ("gpu_result", "cpu_result"):
            values = kernel.get(key)
            if not isinstance(values, list) or not 1 <= len(values) <= 2048 or any(type(x) is not int or not 0 <= x <= 0xffffffff for x in values):
                raise ProbeError(f"Kernel report has an invalid {key}.")
        if kernel["gpu_result"] != kernel["cpu_result"] or type(kernel.get("elements")) is not int or kernel["elements"] != len(kernel["gpu_result"]):
            raise ProbeError("GPU output differs from the CPU reference or declared size.")
        rounds, samples = kernel.get("rounds"), kernel.get("dispatch_us")
        if type(rounds) is not int or rounds < 1 or not isinstance(samples, list) or len(samples) != rounds or not all(timing_ok(x) for x in samples):
            raise ProbeError("Kernel dispatch samples are incomplete.")
        for key in KERNEL_TIMING_FIELDS:
            if not timing_ok(kernel.get(key)):
                raise ProbeError(f"Kernel timing is missing or invalid: {key}")
    for control in checked_rows(report, "controls", set(CONTROLS)):
        if control.get("passed") is not True:
            raise ProbeError("A guest control failed.")
        if control["guest"] in TRAPPING_CONTROLS:
            if control.get("trapped") is not True or control.get("status") is not None:
                raise ProbeError("Expected guest trap was not observed.")
        elif type(control.get("status")) is not int or control["status"] != 0 or control.get("trapped") is not False:
            raise ProbeError("A guest control did not exit successfully.")


def checked_rows(report, field, expected):
    rows = report.get(field)
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) or not isinstance(row.get("guest"), str) for row in rows):
        raise ProbeError(f"Missing {field} evidence.")
    names = [row["guest"] for row in rows]
    if set(names) != expected or len(names) != len(expected):
        raise ProbeError(f"Missing, unexpected or duplicate {field} evidence.")
    for row in rows:
        for key in ("live_buffers_after", "live_pipelines_after"):
            if type(row.get(key)) is not int or row[key] != 0:
                raise ProbeError(f"Guest resources are not reclaimed: {key}")
        if row.get("runtime") != expected_runtime(row["guest"]):
            raise ProbeError(f"Guest {row['guest']} did not run on its expected runtime.")
    return rows


def run_probe(binary):
    started = time.monotonic()
    try:
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        result = subprocess.run([str(binary), "--json"], capture_output=True,
                                text=True, timeout=30, check=False)
        if result.returncode:
            raise ProbeError(f"GPU smoke exited {result.returncode}. " + result.stderr[-2000:])
        try:
            report = json.loads(result.stdout)
        except (ValueError, TypeError) as error:
            raise ProbeError("GPU smoke returned invalid JSON.") from error
        validate_report(report)
        if hashlib.sha256(binary.read_bytes()).hexdigest() != digest:
            raise ProbeError("Executable changed during the run; run again after the build finishes.", 409)
    except subprocess.TimeoutExpired as error:
        raise ProbeError("GPU smoke exceeded 30 seconds; its process was stopped. Check the terminal before retrying.", 504) from error
    except OSError as error:
        raise ProbeError(f"Cannot run the local executable: {error}", 503) from error
    return {"ok": True, "run_id": uuid.uuid4().hex[:12], "binary_sha256": digest,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2), "report": report}


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, binary):
        self.binary = binary
        self.run_lock = threading.Lock()
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "WasmerGpuLocal/1"

    @property
    def authority(self):
        return f"127.0.0.1:{self.server.server_port}"

    def respond(self, status, body, mime="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(body)

    def valid_host(self):
        if self.headers.get("Host") != self.authority:
            self.respond(403, {"ok": False, "error": "Use the printed 127.0.0.1 URL."})
            return False
        return True

    def do_GET(self):
        if not self.valid_host():
            return
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}
        entry = files.get(urlsplit(self.path).path)
        if not entry:
            self.respond(404, {"ok": False, "error": "Unknown route."})
            return
        self.respond(200, (WEB_ROOT / entry[0]).read_bytes(), entry[1])

    def do_POST(self):
        if not self.valid_host():
            return
        if self.path != "/api/run":
            self.respond(404, {"ok": False, "error": "Unknown route."})
            return
        if self.headers.get("Origin") != f"http://{self.authority}" or self.headers.get("Content-Type") != "application/json":
            self.respond(403, {"ok": False, "error": "Run from this local page with a same-origin JSON request."})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if not 0 <= length <= 16 or self.headers.get("Transfer-Encoding"):
                raise ValueError("invalid body length")
            if json.loads(self.rfile.read(length)) != {}:
                raise ValueError("unexpected input")
        except (ValueError, UnicodeError):
            self.respond(400, {"ok": False, "error": "This demo accepts only an empty JSON object, not commands or shader input."})
            return
        if not self.server.run_lock.acquire(blocking=False):
            self.respond(409, {"ok": False, "error": "A GPU run is already in progress. Wait for it to finish."})
            return
        try:
            self.respond(200, run_probe(self.server.binary))
        except ProbeError as error:
            self.respond(error.status, {"ok": False, "error": str(error)})
        finally:
            self.server.run_lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--binary", type=Path, default=WEB_ROOT.parent / "target/release/gpu-smoke")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    if not binary.is_file():
        parser.error("--binary must name the built gpu-smoke executable")
    app = DemoServer(("127.0.0.1", args.port), binary)
    print(f"Open http://127.0.0.1:{app.server_port}   executable: {binary}", flush=True)
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.server_close()


if __name__ == "__main__":
    main()
