import http.client
import json
from pathlib import Path
import subprocess
import threading
import unittest
from unittest.mock import patch

import server


def report():
    return {"adapter": "Test adapter", "backend": "Metal", "all_kernels_match": True,
            "all_controls_pass": True, "resources_reclaimed": True, "setup_us": 25000,
            "kernels": [{"guest": guest, "runtime": server.expected_runtime(guest), "status": 0,
                         "matches": True, "gpu_result": [0, 1, 7, 2], "cpu_result": [0, 1, 7, 2],
                         "elements": 4, "rounds": 1, "dispatch_us": [10], "cpu_reference_ns": 500,
                         "upload_us": 300, "pipeline_create_us": 4000, "readback_us": 1200,
                         "live_buffers_after": 0, "live_pipelines_after": 0}
                        for guest in server.KERNELS],
            "controls": [{"guest": guest, "runtime": server.expected_runtime(guest),
                          "status": None if guest in server.TRAPPING_CONTROLS else 0,
                          "trapped": guest in server.TRAPPING_CONTROLS, "passed": True,
                          "live_buffers_after": 0, "live_pipelines_after": 0}
                         for guest in server.CONTROLS]}


class ProbeTests(unittest.TestCase):
    def invoke(self, value=None, code=0, stdout=None):
        result = subprocess.CompletedProcess([], code, stdout or json.dumps(value), "failure" if code else "")
        with patch.object(Path, "read_bytes", return_value=b"fixed executable"), patch.object(server.subprocess, "run", return_value=result) as run:
            response = server.run_probe(Path("/owned/gpu-smoke"))
            self.assertEqual(run.call_args.args[0], ["/owned/gpu-smoke", "--json"])
            self.assertFalse(run.call_args.kwargs.get("shell", False))
            return response

    def test_real_success_shape(self):
        result = self.invoke(report())
        self.assertEqual(result["report"]["kernels"][0]["gpu_result"], [0, 1, 7, 2])
        self.assertEqual([row["guest"] for row in result["report"]["kernels"]], list(server.KERNELS))
        self.assertEqual(len(result["report"]["controls"]), 9)
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["binary_sha256"]), 64)

    def test_timings_must_be_present_bounded_and_integral(self):
        self.assertTrue(self.invoke(report())["ok"])
        with self.assertRaises(server.ProbeError):
            self.invoke({k: v for k, v in report().items() if k != "setup_us"})
        for bad in (-1, 1.5, True, None, "12", 10 ** 13, float("nan"), float("inf")):
            for key in server.KERNEL_TIMING_FIELDS:
                value = report()
                value["kernels"][2][key] = bad
                with self.subTest(key=key, bad=bad), self.assertRaises(server.ProbeError):
                    self.invoke(value)
            value = report()
            value["kernels"][0]["dispatch_us"] = [bad]
            with self.subTest(key="dispatch_us", bad=bad), self.assertRaises(server.ProbeError):
                self.invoke(value)
            value = report()
            value["setup_us"] = bad
            with self.subTest(key="setup_us", bad=bad), self.assertRaises(server.ProbeError):
                self.invoke(value)
        for key in server.KERNEL_TIMING_FIELDS:
            value = report()
            del value["kernels"][1][key]
            with self.subTest(missing=key), self.assertRaises(server.ProbeError):
                self.invoke(value)

    def test_wasix_rows_are_required_and_must_run_on_wasix(self):
        for edit in ({"kernels": report()["kernels"][:2]}, {"controls": report()["controls"][:8]}):
            with self.subTest(edit=edit), self.assertRaises(server.ProbeError):
                self.invoke(report() | edit)
        for field, index, runtime in (("kernels", 2, "core"), ("kernels", 0, "wasix"),
                                      ("controls", 8, "core"), ("controls", 0, None)):
            value = report()
            value[field][index]["runtime"] = runtime
            with self.subTest(field=field, index=index, runtime=runtime), self.assertRaises(server.ProbeError):
                self.invoke(value)

    def test_nonzero_cannot_export_success(self):
        with self.assertRaises(server.ProbeError):
            self.invoke(report(), code=1)

    def test_bad_or_missing_results_refuse(self):
        for edit in ({"gpu_result": [999]}, {"cpu_result": []},
                     {"matches": False}, {"live_buffers_after": 1},
                     {"status": False}, {"gpu_result": [True]}, {"elements": 3}):
            with self.subTest(edit=edit), self.assertRaises(server.ProbeError):
                value = report()
                value["kernels"][1].update(edit)
                self.invoke(value)
        with self.assertRaises(server.ProbeError):
            self.invoke(stdout="not JSON")

    def test_empty_missing_or_failed_controls_cannot_pass(self):
        for edit in ({"kernels": []}, {"controls": []}, {"all_kernels_match": False},
                     {"resources_reclaimed": None}, {"kernels": report()["kernels"][:1]}):
            with self.subTest(edit=edit), self.assertRaises(server.ProbeError):
                self.invoke(report() | edit)
        for edit in ({"passed": False}, {"status": 1}, {"live_pipelines_after": 1}):
            value = report()
            value["controls"][0].update(edit)
            with self.subTest(edit=edit), self.assertRaises(server.ProbeError):
                self.invoke(value)

    def test_timeout_is_a_failure(self):
        with patch.object(Path, "read_bytes", return_value=b"binary"), patch.object(server.subprocess, "run", side_effect=subprocess.TimeoutExpired("probe", 30)):
            with self.assertRaises(server.ProbeError) as raised:
                server.run_probe(Path("/owned/gpu-smoke"))
            self.assertEqual(raised.exception.status, 504)

    def test_binary_drift_refuses(self):
        result = subprocess.CompletedProcess([], 0, json.dumps(report()), "")
        with patch.object(Path, "read_bytes", side_effect=[b"before", b"after"]), patch.object(server.subprocess, "run", return_value=result):
            with self.assertRaises(server.ProbeError):
                server.run_probe(Path("/owned/gpu-smoke"))


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.app = server.DemoServer(("127.0.0.1", 0), Path("/owned/gpu-smoke"))
        self.thread = threading.Thread(target=self.app.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.app.server_port

    def tearDown(self):
        self.app.shutdown()
        self.app.server_close()
        self.thread.join()

    def request(self, path="/api/run", body="{}", headers=None, method="POST"):
        connection = http.client.HTTPConnection("127.0.0.1", self.port)
        default = {"Origin": f"http://127.0.0.1:{self.port}", "Content-Type": "application/json"}
        connection.request(method, path, body=body, headers=default | (headers or {}))
        response = connection.getresponse()
        status, data = response.status, response.read()
        connection.close()
        return status, data

    def test_endpoint_really_invokes_probe(self):
        with patch.object(server, "run_probe", return_value={"ok": True, "report": report()}) as run:
            self.assertEqual(self.request()[0], 200)
            run.assert_called_once_with(Path("/owned/gpu-smoke"))

    def test_cross_origin_and_unrecognized_hosts_refuse_without_run(self):
        with patch.object(server, "run_probe") as run:
            for headers in ({"Origin": "https://elsewhere.test"}, {"Host": "elsewhere.test"}, {"Content-Type": "text/plain"}):
                self.assertGreaterEqual(self.request(headers=headers)[0], 400)
            run.assert_not_called()

    def test_arbitrary_input_and_parallel_runs_refuse(self):
        with patch.object(server, "run_probe") as run:
            self.assertEqual(self.request(body='{"command":"ignored"}')[0], 400)
            self.app.run_lock.acquire()
            try:
                self.assertEqual(self.request()[0], 409)
            finally:
                self.app.run_lock.release()
            run.assert_not_called()

    def test_get_is_static_and_cannot_run_or_read_arbitrary_files(self):
        with patch.object(server, "run_probe") as run:
            status, body = self.request(path="/", body=None, method="GET")
            self.assertEqual(status, 200)
            self.assertIn(b"Run on my GPU", body)
            self.assertEqual(self.request(path="/../Cargo.toml", method="GET")[0], 404)
            self.assertEqual(self.request(path="/api/run", method="GET")[0], 404)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
