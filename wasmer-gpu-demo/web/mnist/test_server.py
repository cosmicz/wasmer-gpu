import http.client
import io
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import server


def events(ok=True):
    lines = [
        {"event": "start", "steps": 600, "adapter": "Test adapter"},
        {"event": "progress", "step": 0, "loss": 2.30, "samples_seen": 64, "elapsed_ms": 5},
        {"event": "eval", "step": 49, "test_accuracy": 0.8, "correct": 1600, "total": 2000},
        {"event": "predictions", "step": 49, "items": [{"index": 0, "label": 7, "predicted": 7, "probs": [0.0] * 10}]},
        {"event": "check", "passed": ok},
        {"event": "done", "checks_passed": ok, "final_test_accuracy": 0.8},
    ]
    return "\n".join(json.dumps(e) for e in lines) + "\nnot json\n" + json.dumps({"event": "bogus"}) + "\n"


class FakeProcess:
    """`block` keeps the process alive until terminate; `slow_exit` keeps it
    alive after terminate until the test calls `release()`."""

    def __init__(self, text, code=0, block=False, unkillable=False, slow_exit=False):
        self.stdout = io.StringIO(text)
        self.stderr = io.StringIO("stderr text")
        self.code = code
        self.exited = threading.Event()
        if not block and not slow_exit:
            self.exited.set()
        self.terminated = False
        self.unkillable = unkillable
        self.slow_exit = slow_exit

    def poll(self):
        return None if self.unkillable or not self.exited.is_set() else self.code

    def wait(self, timeout=None):
        if self.unkillable:
            if timeout is None:
                self.exited.wait()  # the pump thread parks forever; daemon
                return self.code
            raise subprocess.TimeoutExpired("trainer", timeout)
        if not self.exited.wait(timeout):
            raise subprocess.TimeoutExpired("trainer", timeout or 0)
        return self.code

    def terminate(self):
        self.terminated = True
        if not self.unkillable and not self.slow_exit:
            self.exited.set()

    def kill(self):
        self.terminate()

    def release(self):
        self.exited.set()


def wait_state(job, states, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job.snapshot(0)["state"] in states:
            return job.snapshot(0)
        time.sleep(0.01)
    raise AssertionError(f"job stuck in {job.snapshot(0)['state']}")


class JobTests(unittest.TestCase):
    def test_events_come_only_from_the_owned_process(self):
        job = server.Job(Path("/owned/gpu-mnist"))
        with patch.object(server.subprocess, "Popen", return_value=FakeProcess(events())) as popen:
            self.assertTrue(job.start())
            snapshot = wait_state(job, {"finished", "failed"})
            self.assertEqual(popen.call_args.args[0][0], "/owned/gpu-mnist")
            self.assertFalse(popen.call_args.kwargs.get("shell", False))
        self.assertEqual(snapshot["state"], "finished")
        kinds = [e["event"] for e in snapshot["events"]]
        self.assertEqual(kinds, ["start", "progress", "eval", "predictions", "check", "done"])
        self.assertEqual(job.snapshot(4)["events"][0]["event"], "check")

    def test_error_after_passing_done_is_a_failed_run(self):
        job = server.Job(Path("/owned/gpu-mnist"))
        text = events() + json.dumps({"event": "error", "message": "late failure"}) + "\n"
        with patch.object(server.subprocess, "Popen", return_value=FakeProcess(text)):
            job.start()
            snapshot = wait_state(job, {"finished", "failed"})
        self.assertEqual(snapshot["state"], "failed")
        self.assertEqual(snapshot["events"][-1]["event"], "error")
        self.assertIn("late failure", json.dumps(snapshot["events"]))

    def test_failed_check_or_nonzero_exit_is_not_success(self):
        for process in (FakeProcess(events(ok=False)), FakeProcess(events(), code=1), FakeProcess("", code=0)):
            job = server.Job(Path("/owned/gpu-mnist"))
            with patch.object(server.subprocess, "Popen", return_value=process):
                job.start()
                snapshot = wait_state(job, {"finished", "failed"})
            self.assertEqual(snapshot["state"], "failed")
            self.assertEqual(snapshot["events"][-1]["event"], "error")

    def test_single_job_stop_and_reset(self):
        job = server.Job(Path("/owned/gpu-mnist"))
        process = FakeProcess(events(), block=True)
        with patch.object(server.subprocess, "Popen", return_value=process):
            self.assertTrue(job.start())
            self.assertFalse(job.start())
            self.assertTrue(job.stop())
            self.assertTrue(process.terminated)
            self.assertEqual(wait_state(job, {"stopped"})["state"], "stopped")
            self.assertTrue(job.reset())
            self.assertEqual(job.snapshot(0), {"ok": True, "state": "idle", "run_id": 2, "exit_code": None, "next": 0, "events": []})


    def test_popen_failure_is_reported_not_running(self):
        job = server.Job(Path("/owned/gpu-mnist"))
        with patch.object(server.subprocess, "Popen", side_effect=OSError("no such executable")):
            with self.assertRaises(server.StartError):
                job.start()
        snapshot = job.snapshot(0)
        self.assertEqual(snapshot["state"], "failed")
        self.assertEqual(snapshot["events"][-1]["event"], "error")
        with patch.object(server.subprocess, "Popen", return_value=FakeProcess(events())):
            self.assertTrue(job.start())
            self.assertEqual(wait_state(job, {"finished"})["state"], "finished")

    def test_reset_refuses_while_the_process_survives(self):
        job = server.Job(Path("/owned/gpu-mnist"))
        with patch.object(server.subprocess, "Popen", return_value=FakeProcess(events(), block=True, unkillable=True)):
            job.start()
            with patch.object(server.subprocess, "TimeoutExpired", subprocess.TimeoutExpired):
                self.assertFalse(job.stop())
                self.assertFalse(job.reset())
        snapshot = job.snapshot(0)
        self.assertEqual(snapshot["state"], "running")
        self.assertEqual(snapshot["run_id"], 1)


    def test_concurrent_reset_waits_for_a_slow_stop(self):
        job = server.Job(Path("/owned/gpu-mnist"))
        process = FakeProcess(events(), block=True, slow_exit=True)
        with patch.object(server.subprocess, "Popen", return_value=process):
            job.start()
        stop_result, reset_result = [], []
        stopper = threading.Thread(target=lambda: stop_result.append(job.stop()))
        stopper.start()
        deadline = time.monotonic() + 2
        while job.snapshot(0)["state"] != "stopping" and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(job.snapshot(0)["state"], "stopping")
        resetter = threading.Thread(target=lambda: reset_result.append(job.reset()))
        resetter.start()
        time.sleep(0.1)
        # The process is still alive: reset must not have cleared anything.
        self.assertEqual(reset_result, [])
        self.assertEqual(job.snapshot(0)["state"], "stopping")
        self.assertEqual(job.snapshot(0)["run_id"], 1)
        process.release()
        stopper.join(2)
        resetter.join(2)
        self.assertEqual(stop_result, [True])
        self.assertEqual(reset_result, [True])
        self.assertEqual(job.snapshot(0)["state"], "idle")
        self.assertEqual(job.snapshot(0)["run_id"], 2)


class DigitTests(unittest.TestCase):
    def test_digits_are_read_from_idx_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "t10k-images-idx3-ubyte").write_bytes(struct.pack(">IIII", 2051, 30, 28, 28) + bytes(range(256)) * 92)
            (path / "t10k-labels-idx1-ubyte").write_bytes(struct.pack(">II", 2049, 30) + bytes(range(30)))
            digits = server.load_digits(path)
            self.assertEqual(len(digits), 24)
            self.assertEqual(digits[3]["label"], 3)
            self.assertEqual(len(digits[3]["pixels"]), 784)
            self.assertEqual(digits[1]["pixels"][0], 784 % 256)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.app = server.DemoServer(("127.0.0.1", 0), Path("/owned/gpu-mnist"), Path("/nonexistent"))
        self.thread = threading.Thread(target=self.app.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.app.server_port

    def tearDown(self):
        self.app.shutdown()
        self.app.server_close()
        self.thread.join()

    def request(self, path, body="{}", headers=None, method="POST"):
        connection = http.client.HTTPConnection("127.0.0.1", self.port)
        default = {"Origin": f"http://127.0.0.1:{self.port}", "Content-Type": "application/json"}
        connection.request(method, path, body=body, headers=default | (headers or {}))
        response = connection.getresponse()
        status, data = response.status, response.read()
        connection.close()
        return status, data

    def test_train_route_starts_owned_process_once(self):
        with patch.object(server.subprocess, "Popen", return_value=FakeProcess(events(), block=True)):
            self.assertEqual(self.request("/api/train")[0], 200)
            self.assertEqual(self.request("/api/train")[0], 409)
            status, data = self.request("/api/events?since=0", method="GET", body=None)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(data)["state"], "running")
            self.assertEqual(self.request("/api/stop")[0], 200)
            self.assertEqual(self.request("/api/reset")[0], 200)

    def test_cross_origin_bad_input_and_unknown_routes_refuse(self):
        with patch.object(server.subprocess, "Popen") as popen:
            self.assertEqual(self.request("/api/train", headers={"Origin": "https://elsewhere.test"})[0], 403)
            self.assertEqual(self.request("/api/train", headers={"Host": "elsewhere.test"})[0], 403)
            self.assertEqual(self.request("/api/train", body='{"steps": 1}')[0], 400)
            self.assertEqual(self.request("/api/other")[0], 404)
            self.assertEqual(self.request("/../Cargo.toml", method="GET", body=None)[0], 404)
            popen.assert_not_called()

    def test_start_and_reset_failures_are_http_errors(self):
        with patch.object(server.subprocess, "Popen", side_effect=OSError("missing")):
            status, data = self.request("/api/train")
            self.assertEqual(status, 503)
            self.assertFalse(json.loads(data)["ok"])
        with patch.object(server.subprocess, "Popen", return_value=FakeProcess(events(), block=True, unkillable=True)):
            self.assertEqual(self.request("/api/train")[0], 200)
            status, data = self.request("/api/reset")
            self.assertEqual(status, 500)
            self.assertFalse(json.loads(data)["ok"])
            state = json.loads(self.request("/api/events?since=0", method="GET", body=None)[1])
            self.assertEqual(state["state"], "running")

    def test_static_and_missing_digits(self):
        status, body = self.request("/", method="GET", body=None)
        self.assertEqual(status, 200)
        self.assertIn(b"Train on my GPU", body)
        self.assertEqual(self.request("/api/digits", method="GET", body=None)[0], 503)


if __name__ == "__main__":
    unittest.main()
