"""Mounted apps exercise the existing handlers, without launching GPU jobs."""
import http.client
import importlib.util
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("hub", Path(__file__).with_name("server.py"))
hub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hub)


class HubTests(unittest.TestCase):
    def setUp(self):
        self.app = hub.DemoServer(("127.0.0.1", 0), Path("/owned/bin"), Path("/owned/data"))
        self.thread = threading.Thread(target=self.app.serve_forever, daemon=True)
        self.thread.start()
        self.origin = f"http://127.0.0.1:{self.app.server_port}"

    def tearDown(self):
        self.app.shutdown()
        self.app.server_close()
        self.thread.join()

    def request(self, path, payload=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.app.server_port)
        opts = {"Origin": self.origin, "Content-Type": "application/json"}
        opts.update(headers or {})
        conn.request("GET" if payload is None else "POST", path,
                     body=None if payload is None else json.dumps(payload), headers=opts)
        response = conn.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        conn.close()
        return result

    def test_shell_and_three_self_contained_mounts(self):
        code, _, body = self.request("/")
        self.assertEqual(code, 200)
        self.assertEqual(body.count(b'role="tab"'), 3)
        for name in ("compute", "image", "mnist"):
            code, headers, body = self.request(f"/{name}/")
            self.assertEqual(code, 200)
            self.assertIn(b'embedded.css', body)
            self.assertIn("frame-ancestors 'self'", headers["Content-Security-Policy"])
            self.assertNotIn(b'src="/', body)
            self.assertEqual(self.request(f"/{name}/app.js")[0], 200)
            self.assertEqual(self.request(f"/{name}/style.css")[0], 200)

    def test_host_origin_and_paths_remain_bounded(self):
        self.assertEqual(self.request("/", headers={"Host": "evil.example"})[0], 403)
        for route in ("/compute/api/run", "/image/api/filter", "/mnist/api/train"):
            self.assertEqual(self.request(route, {}, {"Origin": "https://evil.example"})[0], 403)
        for path in ("/../Cargo.toml", "/image/../../Cargo.toml", "/mnist/server.py", "/api/run"):
            self.assertEqual(self.request(path)[0], 404)

    def test_compute_delegates_to_the_existing_probe(self):
        with patch.object(hub.compute, "run_probe", return_value={"ok": True, "source": "compute"}) as run:
            code, _, body = self.request("/compute/api/run", {})
            self.assertEqual(code, 200)
            self.assertEqual(json.loads(body)["source"], "compute")
            run.assert_called_once_with(Path("/owned/bin/gpu-smoke"))

    def test_image_delegates_to_the_existing_filter(self):
        with patch.object(hub.image, "run_filter", return_value={"ok": True, "source": "image"}) as run:
            code, _, body = self.request("/image/api/filter", {"filter": "edges", "param": 2})
            self.assertEqual(code, 200)
            self.assertEqual(json.loads(body)["source"], "image")
            run.assert_called_once_with(Path("/owned/bin/gpu-image"), "edges", 2, None, None, None)

    def test_training_events_reset_and_dataset_context(self):
        job = self.app.contexts["mnist"].job
        self.assertIn("/owned/data", job.args)
        job.events = [{"event": "progress", "step": 123}]
        code, _, body = self.request("/mnist/api/events?since=0")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["events"][0]["step"], 123)
        self.assertEqual(self.request("/mnist/api/reset", {})[0], 200)
        self.assertEqual(job.events, [])
        with patch.object(hub.mnist, "load_digits", return_value=[]) as read:
            self.assertEqual(self.request("/mnist/api/digits")[0], 200)
            read.assert_called_once_with(Path("/owned/data"))

    def test_namespaced_training_launch_failure_is_not_success(self):
        with patch.object(hub.mnist.subprocess, "Popen", side_effect=OSError("not executable")):
            code, _, body = self.request("/mnist/api/train", {})
            self.assertEqual(code, 503)
            self.assertFalse(json.loads(body)["ok"])
        _, _, body = self.request("/mnist/api/events")
        self.assertEqual(json.loads(body)["state"], "failed")

    def test_mount_redirect_keeps_relative_urls_under_mount(self):
        code, headers, _ = self.request("/mnist")
        self.assertEqual(code, 302)
        self.assertEqual(headers["Location"], "/mnist/")


if __name__ == "__main__":
    unittest.main()
