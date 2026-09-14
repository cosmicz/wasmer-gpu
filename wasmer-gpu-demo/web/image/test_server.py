import base64
import http.client
import json
from pathlib import Path
import subprocess
import threading
import unittest
from unittest.mock import patch

import server

W, H = 4, 3


def pixels(fill=7):
    return bytes([fill]) * (W * H * 4)


def b64(data):
    return base64.b64encode(data).decode()


def report(**edits):
    value = {"adapter": "Test adapter", "backend": "Metal", "matches": True, "resources_reclaimed": True,
             "control": "none", "runtime": "wasix", "exit_code": 0, "filter": "blur", "param": 1,
             "width": W, "height": H, "pixels": W * H, "source": "file", "guest": "image_filter.wat",
             "setup_us": 1, "upload_us": 2, "pipeline_create_us": 3, "dispatch_us": [4], "readback_us": 5,
             "cpu_reference_us": 6, "changed_pixels": 12, "mismatched_pixels": 0,
             "input_rgba_b64": b64(pixels(1)), "output_rgba_b64": b64(pixels(2)), "output_fnv1a64": "0" * 16,
             "guest_bytes": 12345, "guest_fnv1a64": "a" * 16}
    value.update(edits)
    return value


class RunTests(unittest.TestCase):
    def invoke(self, value=None, code=0, stdout=None, name="blur", param=1, pixel_bytes=None):
        result = subprocess.CompletedProcess([], code, stdout or json.dumps(value), "failure" if code else "")
        with patch.object(Path, "read_bytes", return_value=b"fixed executable"), \
             patch.object(server.subprocess, "run", return_value=result) as run:
            response = server.run_filter(Path("/owned/gpu-image"), name, param, pixel_bytes,
                                         W if pixel_bytes else None, H if pixel_bytes else None)
            args = run.call_args.args[0]
            self.assertEqual(args[:5], ["/owned/gpu-image", "--json", "--filter", name, "--param"])
            self.assertFalse(run.call_args.kwargs.get("shell", False))
            self.assertEqual(run.call_args.kwargs.get("timeout"), server.RUN_TIMEOUT_S)
            return response, args

    def test_generated_success_shape(self):
        response, args = self.invoke(report(source="generated"))
        self.assertTrue(response["ok"])
        self.assertEqual(response["report"]["output_rgba_b64"], b64(pixels(2)))
        self.assertNotIn("--input", args)
        self.assertEqual(len(response["binary_sha256"]), 64)

    def test_file_input_is_written_to_a_temporary_file_and_bounded(self):
        def capture(args, **kwargs):
            path = Path(args[args.index("--input") + 1])
            self.assertEqual(path.read_bytes(), pixels())
            self.assertEqual(args[args.index("--width") + 1], str(W))
            self.assertEqual(args[args.index("--height") + 1], str(H))
            return subprocess.CompletedProcess(args, 0, json.dumps(report()), "")
        original = Path.read_bytes
        with patch.object(server.subprocess, "run", side_effect=capture), \
             patch.object(Path, "read_bytes", autospec=True,
                          side_effect=lambda self: b"exe" if self.name == "gpu-image" else original(self)):
            response = server.run_filter(Path("/owned/gpu-image"), "blur", 1, pixels(), W, H)
        self.assertTrue(response["ok"])

    def test_nonzero_exit_and_mismatch_refuse(self):
        with self.assertRaises(server.ProbeError):
            self.invoke(report(), code=1)
        for edit in ({"matches": False}, {"resources_reclaimed": False}, {"control": "wrong-shader"},
                     {"runtime": "core"}, {"exit_code": 7}, {"filter": "edges"}, {"param": 3},
                     {"adapter": ""}, {"width": 513}, {"dispatch_us": []}, {"dispatch_us": [1, 2]},
                     {"mismatched_pixels": 1}, {"changed_pixels": W * H + 1},
                     {"output_rgba_b64": b64(pixels()[:-4])}, {"input_rgba_b64": 5},
                     {"guest_fnv1a64": "xyz"}, {"guest_fnv1a64": None}, {"guest_bytes": 0}):
            with self.subTest(edit=edit), self.assertRaises(server.ProbeError):
                self.invoke(report(**edit))

    def test_request_dimensions_must_match_report(self):
        with self.assertRaises(server.ProbeError):
            self.invoke(report(width=W + 1, input_rgba_b64=b64(bytes((W + 1) * H * 4)),
                               output_rgba_b64=b64(bytes((W + 1) * H * 4))), pixel_bytes=pixels())

    def test_invalid_json_refuses(self):
        with self.assertRaises(server.ProbeError):
            self.invoke(stdout="not json")


class ParseTests(unittest.TestCase):
    def test_generated_request(self):
        self.assertEqual(server.parse_request({"filter": "edges", "param": 3}), ("edges", 3, None, None, None))
        self.assertEqual(server.parse_request({"filter": "blur", "source": "generated"})[:2], ("blur", 1))

    def test_file_request(self):
        name, param, data, width, height = server.parse_request(
            {"filter": "emboss", "param": 2, "source": {"width": W, "height": H, "rgba_b64": b64(pixels())}})
        self.assertEqual((name, param, width, height), ("emboss", 2, W, H))
        self.assertEqual(data, pixels())

    def test_rejections(self):
        good = {"width": W, "height": H, "rgba_b64": b64(pixels())}
        for body in ([], {"filter": "identity"}, {"filter": "blur", "param": 0}, {"filter": "blur", "param": 17},
                     {"filter": "blur", "param": "2"}, {"filter": "blur", "param": True},
                     {"filter": "blur", "shader": "x"}, {"filter": "blur", "source": "file"},
                     {"filter": "blur", "source": good | {"width": 0}}, {"filter": "blur", "source": good | {"height": 513}},
                     {"filter": "blur", "source": good | {"rgba_b64": "!!!!"}},
                     {"filter": "blur", "source": good | {"rgba_b64": b64(pixels()[:-1])}},
                     {"filter": "blur", "source": good | {"extra": 1}}, {"filter": "blur", "source": {"width": W, "height": H}}):
            with self.subTest(body=body), self.assertRaises(server.ProbeError) as raised:
                server.parse_request(body)
            self.assertEqual(raised.exception.status, 400)


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = server.DemoServer(("127.0.0.1", 0), Path("/owned/gpu-image"))
        cls.thread = threading.Thread(target=cls.app.serve_forever, daemon=True)
        cls.thread.start()
        cls.authority = f"127.0.0.1:{cls.app.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.app.shutdown()
        cls.app.server_close()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.app.server_port, timeout=5)
        merged = {"Host": self.authority}
        if body is not None:
            merged.update({"Origin": f"http://{self.authority}", "Content-Type": "application/json"})
        merged.update(headers or {})
        connection.request(method, path, body=body, headers=merged)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response, data

    def test_get_serves_page_without_running_anything(self):
        with patch.object(server.subprocess, "run") as run:
            response, data = self.request("GET", "/")
            self.assertEqual(response.status, 200)
            self.assertIn(b"Filter", data)
            self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])
            self.assertEqual(self.request("GET", "/api/filter")[0].status, 404)
            self.assertEqual(self.request("GET", "/../server.py")[0].status, 404)
            run.assert_not_called()

    def test_host_and_origin_are_checked(self):
        self.assertEqual(self.request("GET", "/", headers={"Host": "example.com"})[0].status, 403)
        self.assertEqual(self.request("POST", "/api/filter", body=b"{}", headers={"Origin": "http://evil"})[0].status, 403)
        self.assertEqual(self.request("POST", "/api/filter", body=b"{}", headers={"Content-Type": "text/plain"})[0].status, 403)

    def test_bad_requests_never_reach_the_executable(self):
        with patch.object(server.subprocess, "run") as run:
            self.assertEqual(self.request("POST", "/api/filter", body=b"[1")[0].status, 400)
            self.assertEqual(self.request("POST", "/api/filter", body=json.dumps({"filter": "nope"}).encode())[0].status, 400)
            self.assertEqual(self.request("POST", "/api/filter", body=b"{}", headers={"Content-Length": str(server.MAX_BODY + 1)})[0].status, 413)
            self.assertEqual(self.request("POST", "/wrong", body=b"{}")[0].status, 404)
            run.assert_not_called()

    def test_valid_request_runs_and_returns_the_report(self):
        result = subprocess.CompletedProcess([], 0, json.dumps(report(source="generated")), "")
        with patch.object(Path, "read_bytes", return_value=b"exe"), patch.object(server.subprocess, "run", return_value=result):
            response, data = self.request("POST", "/api/filter", body=json.dumps({"filter": "blur", "param": 1}).encode())
        self.assertEqual(response.status, 200)
        self.assertTrue(json.loads(data)["ok"])


if __name__ == "__main__":
    unittest.main()
