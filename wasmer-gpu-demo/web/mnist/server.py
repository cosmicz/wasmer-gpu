"""Loopback-only control for one owned gpu-mnist training process.

Routes (127.0.0.1 only, Host and Origin checked, empty JSON bodies):
  GET  /               page        GET /app.js, /style.css  assets
  GET  /api/digits     first 24 held-out test digits (pixels + labels) from the local IDX file
  GET  /api/events?since=N  events the trainer has emitted after index N, plus job state
  POST /api/train      start the fixed executable if no job is running (409 otherwise)
  POST /api/stop       terminate the owned process
  POST /api/reset      stop and clear all captured events
Every event shown by the page comes from the trainer's own stdout.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlsplit

WEB_ROOT = Path(__file__).resolve().parent
DEMO_ROOT = WEB_ROOT.parent.parent
GALLERY = 24
EVENT_KINDS = {"start", "progress", "eval", "predictions", "check", "weights", "done", "error"}
MAX_EVENTS = 20_000
TRAIN_ARGS = ("--steps", "600", "--eval-every", "50", "--progress-every", "5")


class StartError(Exception):
    """The trainer executable could not be started."""


class Job:
    """One owned trainer process and the events it has produced."""

    def __init__(self, binary, args=TRAIN_ARGS):
        self.binary = binary
        self.args = tuple(args)
        self.lock = threading.Lock()
        # Serialises start/stop/reset end to end, so a reset cannot observe a
        # half-finished stop and clear state while the process is still alive.
        self.control = threading.RLock()
        self.events = []
        self.process = None
        self.state = "idle"        # idle | running | stopping | finished | failed | stopped
        self.exit_code = None
        self.error_seen = False    # any accepted error event makes the run terminal-failing
        self.started_at = None
        self.run_id = 0

    def snapshot(self, since):
        with self.lock:
            since = max(0, min(since, len(self.events)))
            return {"ok": True, "state": self.state, "run_id": self.run_id, "exit_code": self.exit_code,
                    "next": len(self.events), "events": self.events[since:]}

    def start(self):
        with self.control, self.lock:
            if self.state in ("running", "stopping") or self._alive():
                return False
            self.events = []
            self.exit_code = None
            self.error_seen = False
            self.run_id += 1
            try:
                self.process = subprocess.Popen([str(self.binary), *self.args], stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE, text=True)
            except OSError as error:
                self.process = None
                self.state = "failed"
                self.events.append({"event": "error", "message": f"Cannot start the trainer executable: {error}"})
                raise StartError(str(error)) from error
            self.state = "running"
            self.started_at = time.monotonic()
            process, run_id = self.process, self.run_id
            threading.Thread(target=self._pump, args=(process, run_id), daemon=True).start()
        return True

    def _alive(self):
        """Caller holds self.lock."""
        return self.process is not None and self.process.poll() is None

    def _pump(self, process, run_id):
        for line in process.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or event.get("event") not in EVENT_KINDS:
                continue
            with self.lock:
                if self.run_id != run_id:
                    return
                if len(self.events) < MAX_EVENTS:
                    self.events.append(event)
                if event["event"] == "error":
                    self.error_seen = True
        code = process.wait()
        stderr = process.stderr.read()[-2000:]
        with self.lock:
            if self.run_id != run_id:
                return
            self.exit_code = code
            if self.state in ("stopping", "stopped"):
                return
            done = next((e for e in reversed(self.events) if e.get("event") == "done"), None)
            if code == 0 and done and done.get("checks_passed") is True and not self.error_seen:
                self.state = "finished"
            else:
                self.state = "failed"
                reason = "reported an error event" if self.error_seen else f"exited {code} without a passing done event"
                self.events.append({"event": "error", "message": f"Trainer {reason}. {stderr}".strip()})

    def stop(self):
        """Terminates the owned process. Returns True only once it has exited.

        Holds the control lock for the whole wait, so a concurrent stop or
        reset blocks until this one has settled instead of seeing a
        transitional state.
        """
        with self.control:
            with self.lock:
                process = self.process
                if not self._alive():
                    return True
                self.state = "stopping"
            process.terminate()
            exited = self._wait(process, 5) or (process.kill() or self._wait(process, 5))
            with self.lock:
                if self.process is process:
                    self.state = "stopped" if exited else "running"
            return exited

    @staticmethod
    def _wait(process, timeout):
        try:
            process.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    def reset(self):
        """Clears state only after the owned process is really gone."""
        with self.control:
            if not self.stop():
                return False
            with self.lock:
                if self._alive():
                    return False
                self.events = []
                self.state = "idle"
                self.exit_code = None
                self.error_seen = False
                self.run_id += 1
                self.process = None
            return True


def load_digits(path, count=GALLERY):
    images = path / "t10k-images-idx3-ubyte"
    labels = path / "t10k-labels-idx1-ubyte"
    with open(images, "rb") as f:
        header = f.read(16)
        if int.from_bytes(header[:4], "big") != 2051:
            raise ValueError("unexpected IDX magic")
        pixels = f.read(count * 784)
    with open(labels, "rb") as f:
        f.read(8)
        digit_labels = f.read(count)
    return [{"index": i, "label": digit_labels[i], "pixels": list(pixels[i * 784:(i + 1) * 784])}
            for i in range(count)]


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, binary, data_dir):
        self.job = Job(binary)
        self.data_dir = data_dir
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "WasmerGpuMnist/1"

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
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass

    def valid_host(self):
        if self.headers.get("Host") != self.authority:
            self.respond(403, {"ok": False, "error": "Use the printed 127.0.0.1 URL."})
            return False
        return True

    def do_GET(self):
        if not self.valid_host():
            return
        url = urlsplit(self.path)
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}
        if url.path in files:
            name, mime = files[url.path]
            self.respond(200, (WEB_ROOT / name).read_bytes(), mime)
        elif url.path == "/api/events":
            since = parse_qs(url.query).get("since", ["0"])[0]
            self.respond(200, self.server.job.snapshot(int(since) if since.isdigit() else 0))
        elif url.path == "/api/digits":
            try:
                self.respond(200, {"ok": True, "digits": load_digits(self.server.data_dir)})
            except (OSError, ValueError) as error:
                self.respond(503, {"ok": False, "error": f"MNIST test file unavailable: {error}"})
        else:
            self.respond(404, {"ok": False, "error": "Unknown route."})

    def do_POST(self):
        if not self.valid_host():
            return
        if self.path not in ("/api/train", "/api/stop", "/api/reset"):
            self.respond(404, {"ok": False, "error": "Unknown route."})
            return
        if self.headers.get("Origin") != f"http://{self.authority}" or self.headers.get("Content-Type") != "application/json":
            self.respond(403, {"ok": False, "error": "Run from this local page with a same-origin JSON request."})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if not 0 <= length <= 16 or self.headers.get("Transfer-Encoding"):
                raise ValueError("invalid body length")
            if json.loads(self.rfile.read(length) or b"{}") != {}:
                raise ValueError("unexpected input")
        except (ValueError, UnicodeError):
            self.respond(400, {"ok": False, "error": "This demo accepts only an empty JSON object; training parameters are fixed."})
            return
        job = self.server.job
        try:
            if self.path == "/api/train":
                if not job.start():
                    self.respond(409, {"ok": False, "error": "A training job is already running."})
                    return
            elif self.path == "/api/stop":
                if not job.stop():
                    self.respond(500, {"ok": False, "error": "The trainer process did not stop; it is still running."})
                    return
            elif not job.reset():
                self.respond(500, {"ok": False, "error": "Reset refused: the trainer process did not stop and its state is kept."})
                return
        except StartError as error:
            self.respond(503, {"ok": False, "error": str(error)})
            return
        self.respond(200, job.snapshot(0) | {"events": []})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--binary", type=Path, default=DEMO_ROOT / "target/release/gpu-mnist")
    parser.add_argument("--data-dir", type=Path, default=DEMO_ROOT / "data/mnist")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    if not binary.is_file():
        parser.error("--binary must name the built gpu-mnist executable")
    app = DemoServer(("127.0.0.1", args.port), binary, args.data_dir.resolve())
    print(f"Open http://127.0.0.1:{app.server_port}   executable: {binary}", flush=True)
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.job.stop()
        app.server_close()


if __name__ == "__main__":
    main()
