"""Loopback-only browser control for the Wasmer GPU image workbench.

Browser -> 127.0.0.1 -> gpu-image --json -> WASIX guest -> wgpu -> native GPU.
The request chooses a preset, a bounded integer parameter and a pixel source:
the runner's generated scene, or bounded raw RGBA8 pixels decoded by the
browser from a local file. Shader text never comes from HTTP; the guest module
owns every kernel.
"""

import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlsplit
import uuid

WEB_ROOT = Path(__file__).resolve().parent
FILTERS = ("blur", "edges", "emboss")
MAX_SIDE = 512
MAX_PARAM = 16
# 16 bytes of framing, a 512x512 RGBA8 image as base64, and JSON slack.
MAX_BODY = 4 * MAX_SIDE * MAX_SIDE * 4 // 3 + 4096
RUN_TIMEOUT_S = 30


class ProbeError(Exception):
    def __init__(self, message, status=502):
        super().__init__(message)
        self.status = status


def parse_request(body):
    """Returns (filter, param, pixels or None, width, height)."""
    if not isinstance(body, dict) or set(body) - {"filter", "param", "source"}:
        raise ProbeError("Request must carry only filter, param and source.", 400)
    name = body.get("filter")
    if name not in FILTERS:
        raise ProbeError("Unknown preset.", 400)
    param = body.get("param", 1)
    if type(param) is not int or not 1 <= param <= MAX_PARAM:
        raise ProbeError(f"param must be an integer from 1 to {MAX_PARAM}.", 400)
    source = body.get("source", "generated")
    if source == "generated":
        return name, param, None, None, None
    if not isinstance(source, dict) or set(source) != {"width", "height", "rgba_b64"}:
        raise ProbeError("source must be \"generated\" or {width, height, rgba_b64}.", 400)
    width, height, encoded = source["width"], source["height"], source["rgba_b64"]
    if type(width) is not int or type(height) is not int or not (1 <= width <= MAX_SIDE and 1 <= height <= MAX_SIDE):
        raise ProbeError(f"Image sides must be 1 to {MAX_SIDE} pixels.", 400)
    if not isinstance(encoded, str) or len(encoded) > MAX_BODY:
        raise ProbeError("Pixel payload is not a string or is too large.", 400)
    try:
        pixels = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise ProbeError("Pixel payload is not valid base64.", 400) from error
    if len(pixels) != width * height * 4:
        raise ProbeError("Pixel payload does not match width x height x RGBA8.", 400)
    return name, param, pixels, width, height


def validate_report(report, name, param, width, height):
    if not isinstance(report, dict):
        raise ProbeError("The executable did not return a report object.")
    if report.get("matches") is not True or report.get("resources_reclaimed") is not True:
        raise ProbeError("GPU output did not match the CPU reference or resources leaked.")
    if report.get("control") != "none" or report.get("runtime") != "wasix" or report.get("exit_code") != 0:
        raise ProbeError("Report is not a clean WASIX guest run.")
    if report.get("filter") != name or report.get("param") != param:
        raise ProbeError("Report is for a different preset.")
    for key in ("adapter", "backend"):
        if not isinstance(report.get(key), str) or not report[key].strip():
            raise ProbeError(f"Report is missing its {key}.")
    if not re.fullmatch(r"[0-9a-f]{16}", str(report.get("guest_fnv1a64"))) or type(report.get("guest_bytes")) is not int or report["guest_bytes"] <= 0:
        raise ProbeError("Report does not identify its embedded guest module.")
    w, h = report.get("width"), report.get("height")
    if type(w) is not int or type(h) is not int or not (1 <= w <= MAX_SIDE and 1 <= h <= MAX_SIDE):
        raise ProbeError("Report has invalid dimensions.")
    if width is not None and (w, h) != (width, height):
        raise ProbeError("Report dimensions differ from the request.")
    for key in ("input_rgba_b64", "output_rgba_b64"):
        value = report.get(key)
        if not isinstance(value, str) or len(value) != -(-w * h * 4 // 3) * 4:
            raise ProbeError(f"Report has an invalid {key}.")
    for key in ("setup_us", "upload_us", "pipeline_create_us", "readback_us", "cpu_reference_us", "changed_pixels", "mismatched_pixels"):
        if type(report.get(key)) is not int or report[key] < 0:
            raise ProbeError(f"Report has an invalid {key}.")
    samples = report.get("dispatch_us")
    if not isinstance(samples, list) or len(samples) != 1 or type(samples[0]) is not int or samples[0] < 0:
        raise ProbeError("Report lacks its single dispatch sample.")
    if report["mismatched_pixels"] != 0 or report["changed_pixels"] > w * h:
        raise ProbeError("Report pixel counts are inconsistent.")


def run_filter(binary, name, param, pixels, width, height):
    started = time.monotonic()
    args = [str(binary), "--json", "--filter", name, "--param", str(param)]
    try:
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(prefix="wasmer-gpu-image-") as folder:
            if pixels is not None:
                path = Path(folder) / "input.rgba"
                path.write_bytes(pixels)
                args += ["--input", str(path), "--width", str(width), "--height", str(height)]
            result = subprocess.run(args, capture_output=True, text=True, timeout=RUN_TIMEOUT_S, check=False)
        if result.returncode:
            raise ProbeError(f"gpu-image exited {result.returncode}. " + result.stderr[-2000:])
        try:
            report = json.loads(result.stdout)
        except (ValueError, TypeError) as error:
            raise ProbeError("gpu-image returned invalid JSON.") from error
        validate_report(report, name, param, width, height)
        if hashlib.sha256(binary.read_bytes()).hexdigest() != digest:
            raise ProbeError("Executable changed during the run; run again after the build finishes.", 409)
    except subprocess.TimeoutExpired as error:
        raise ProbeError(f"gpu-image exceeded {RUN_TIMEOUT_S} seconds; its process was stopped.", 504) from error
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
    server_version = "WasmerGpuImageLocal/1"

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
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
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
        if self.path != "/api/filter":
            self.respond(404, {"ok": False, "error": "Unknown route."})
            return
        if self.headers.get("Origin") != f"http://{self.authority}" or self.headers.get("Content-Type") != "application/json":
            self.respond(403, {"ok": False, "error": "Run from this local page with a same-origin JSON request."})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if not 0 <= length <= MAX_BODY or self.headers.get("Transfer-Encoding"):
                raise ProbeError("Request body is missing or larger than one bounded image.", 413)
            request = parse_request(json.loads(self.rfile.read(length)))
        except (ValueError, UnicodeError):
            self.respond(400, {"ok": False, "error": "Request is not a JSON object."})
            return
        except ProbeError as error:
            self.respond(error.status, {"ok": False, "error": str(error)})
            return
        if not self.server.run_lock.acquire(blocking=False):
            self.respond(409, {"ok": False, "error": "A GPU run is already in progress. Wait for it to finish."})
            return
        try:
            self.respond(200, run_filter(self.server.binary, *request))
        except ProbeError as error:
            self.respond(error.status, {"ok": False, "error": str(error)})
        finally:
            self.server.run_lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--binary", type=Path, default=WEB_ROOT.parent.parent / "target/release/gpu-image")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    if not binary.is_file():
        parser.error("--binary must name the built gpu-image executable")
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
