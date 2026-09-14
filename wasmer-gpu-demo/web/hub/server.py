"""One loopback application for Compute, Image Lab and MNIST.

The existing handlers own validation and execution. This entry point supplies
their per-app context and a path prefix; no proxy or extra servers are needed.
"""
import argparse
from http.server import ThreadingHTTPServer
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
WEB = ROOT.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compute = load("compute_app", WEB / "server.py")
image = load("image_app", WEB / "image/server.py")
mnist = load("mnist_app", WEB / "mnist/server.py")
APPS = {"compute": compute, "image": image, "mnist": mnist}
ASSETS = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8")}


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, bin_dir, data_dir):
        lock = threading.Lock()
        self.contexts = {
            "compute": SimpleNamespace(binary=bin_dir / "gpu-smoke", run_lock=lock),
            "image": SimpleNamespace(binary=bin_dir / "gpu-image", run_lock=lock),
            "mnist": SimpleNamespace(job=mnist.Job(bin_dir / "gpu-mnist",
                (*mnist.TRAIN_ARGS, "--data-dir", str(data_dir))), data_dir=data_dir),
        }
        super().__init__(address, Handler)
        for context in self.contexts.values():
            context.server_port = self.server_port


class Handler(compute.Handler):
    server_version = "WasmerGpuApp/1"

    def log_message(self, *_):
        pass

    def send_header(self, name, value):
        if name.lower() == "content-security-policy":
            # Only our own origin may embed these local applications.
            value = ("default-src 'self'; script-src 'self'; style-src 'self'; "
                     "img-src 'self' data: blob:; connect-src 'self'; "
                     "frame-src 'self'; frame-ancestors 'self'; "
                     "base-uri 'none'; form-action 'none'")
        super().send_header(name, value)

    def mount(self):
        name, _, rest = self.path.lstrip("/").partition("/")
        return (name, "/" + rest) if name in APPS else (None, None)

    def call_app(self, method, name, path):
        # Each HTTP request has its own Handler. Never change the shared server
        # or module globals while another tab is using a different app.
        original_server, original_path = self.server, self.path
        self.server, self.path = self.server.contexts[name], path
        try:
            getattr(APPS[name].Handler, method)(self)
        finally:
            self.server, self.path = original_server, original_path

    def do_GET(self):
        if not self.valid_host():
            return
        path = urlsplit(self.path).path
        if path in {f"/{name}" for name in APPS}:
            self.send_response(302)
            self.send_header("Location", path + "/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/embedded.css":
            self.respond(200, (ROOT / "embedded.css").read_bytes(), "text/css; charset=utf-8")
            return
        name, child_path = self.mount()
        entry = ASSETS.get(urlsplit(child_path).path if name else path)
        if entry:
            filename, mime = entry
            body = ((APPS[name].WEB_ROOT if name else ROOT) / filename).read_bytes()
            if name and filename == "index.html":
                body = body.replace(b"</head>", b'<link rel="stylesheet" href="../embedded.css"></head>')
            self.respond(200, body, mime)
        elif name == "mnist" and urlsplit(child_path).path in ("/api/digits", "/api/events"):
            self.call_app("do_GET", name, child_path)
        else:
            self.respond(404, {"ok": False, "error": "Unknown route."})

    def do_POST(self):
        if not self.valid_host():
            return
        name, path = self.mount()
        if name:
            self.call_app("do_POST", name, path)
        else:
            self.respond(404, {"ok": False, "error": "Unknown route."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8764)
    parser.add_argument("--bin-dir", type=Path, default=WEB.parent / "target/release")
    parser.add_argument("--data-dir", type=Path, default=WEB.parent / "data/mnist")
    args = parser.parse_args()
    binary_dir = args.bin_dir.resolve()
    for name in ("gpu-smoke", "gpu-image", "gpu-mnist"):
        if not (binary_dir / name).is_file():
            parser.error(f"Build {name} first: cargo +stable build --release")
    app = DemoServer(("127.0.0.1", args.port), binary_dir, args.data_dir.resolve())
    print(f"Open http://127.0.0.1:{app.server_port} — Compute / Image Lab / MNIST", flush=True)
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.contexts["mnist"].job.stop()
        app.server_close()


if __name__ == "__main__":
    main()
