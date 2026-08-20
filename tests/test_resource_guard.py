from __future__ import annotations

import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from noema_lab.core.resource_guard import (
    GIB,
    MIB,
    ExecutionDeadlineExceeded,
    IsolatedJobSupervisor,
    IsolatedJobError,
    MemoryGuardConfig,
    ResourceExhausted,
    _read_json_if_present,
)
from noema_lab.core.plan_cache import ExecutionPlanCache
from noema_lab.core.recipes import recipe_from_dict
from noema_lab.core.storage import LocalStore
from noema_lab.ops import build_registry
from noema_lab.ui.server import RunJob
from noema_lab.core import resource_guard as resource_guard_module


class _CommandSupervisor(IsolatedJobSupervisor):
    def __init__(self, command, **kwargs):
        super().__init__(**kwargs)
        self.command = command

    def _worker_command(self):
        return list(self.command)


class ResourceGuardTests(unittest.TestCase):
    def _supervisor(
        self,
        root: Path,
        command,
        config: MemoryGuardConfig,
        **kwargs,
    ):
        return _CommandSupervisor(
            command,
            job_id="guard-test",
            request={"kind": "test"},
            workspace=root / ".noema",
            project_root=root,
            config=config,
            **kwargs,
        )

    def test_default_worker_uses_isolated_python_and_drops_pythonpath_from_systemd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supervisor = IsolatedJobSupervisor(
                job_id="isolated-worker-test",
                request={"kind": "test"},
                workspace=root / ".noema",
                project_root=root,
                config=MemoryGuardConfig(reserve_bytes=512 * MIB),
            )
            command = supervisor._worker_command()
            self.assertEqual(
                command[:7],
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-X",
                    "pycache_prefix=%s" % supervisor.bytecode_cache_path,
                    "-m",
                    "noema_lab.core.job_worker",
                ],
            )
            with mock.patch.dict(
                "noema_lab.core.resource_guard.os.environ",
                {"PATH": "/bin", "PYTHONPATH": "/untrusted"},
                clear=True,
            ):
                arguments = supervisor._systemd_environment_arguments()
            self.assertIn("--setenv=PATH=/bin", arguments)
            self.assertFalse(
                any(argument.startswith("--setenv=PYTHONPATH") for argument in arguments)
            )

    def test_fresh_pycache_prefix_ignores_valid_source_adjacent_bytecode(self):
        with tempfile.TemporaryDirectory(prefix="noema-poisoned-pyc-") as raw:
            root = Path(raw)
            source = root / "probe_module.py"
            source.write_text("VALUE = 'poison'\n", encoding="utf-8")
            stat = source.stat()
            adjacent_cache = root / "__pycache__"
            adjacent_cache.mkdir()
            pyc_path = adjacent_cache / (
                "probe_module.%s.pyc" % sys.implementation.cache_tag
            )
            py_compile.compile(
                str(source), cfile=str(pyc_path), doraise=True
            )
            source.write_text("VALUE = 'safeee'\n", encoding="utf-8")
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            # Force the timestamp-based cache header to match the replacement
            # source. This makes the adjacent pyc demonstrably loadable in the
            # unprotected control process.
            pyc = bytearray(pyc_path.read_bytes())
            current = source.stat()
            pyc[8:12] = int(current.st_mtime).to_bytes(4, "little")
            pyc[12:16] = int(current.st_size).to_bytes(4, "little")
            pyc_path.write_bytes(pyc)
            code = (
                "import sys; sys.path.insert(0, %r); "
                "import probe_module; print(probe_module.VALUE)"
            ) % str(root)
            unprotected = subprocess.run(
                [sys.executable, "-I", "-B", "-c", code],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            protected = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-X",
                    "pycache_prefix=%s" % (root / "fresh-bytecode-cache"),
                    "-c",
                    code,
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        self.assertEqual(unprotected.returncode, 0, unprotected.stderr)
        self.assertEqual(unprotected.stdout.strip(), "poison")
        self.assertEqual(protected.returncode, 0, protected.stderr)
        self.assertEqual(protected.stdout.strip(), "safeee")

    def test_worker_bytecode_cache_must_be_empty_before_launch(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            supervisor = IsolatedJobSupervisor(
                job_id="nonempty-bytecode-test",
                request={"kind": "test"},
                workspace=root / ".noema",
                project_root=root,
                config=MemoryGuardConfig(reserve_bytes=512 * MIB),
            )
            supervisor.control_dir.mkdir(parents=True)
            supervisor.bytecode_cache_path.mkdir()
            (supervisor.bytecode_cache_path / "poison.pyc").write_bytes(b"poison")
            with self.assertRaisesRegex(IsolatedJobError, "not empty"):
                supervisor._prepare_bytecode_cache_path()

    def test_exact_environment_controls_override_hostile_ambient(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supervisor = self._supervisor(
                root,
                ["/bin/true"],
                MemoryGuardConfig(reserve_bytes=512 * MIB),
                environment_set={
                    "CUDA_VISIBLE_DEVICES": "",
                    "OMP_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
                environment_unset=(
                    "LD_LIBRARY_PATH",
                    "LD_PRELOAD",
                    "CUDA_DEVICE_ORDER",
                ),
            )
            hostile = {
                "PATH": "/hostile/bin",
                "CUDA_VISIBLE_DEVICES": "0,1",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "OMP_NUM_THREADS": "32",
                "NUMEXPR_NUM_THREADS": "16",
                "LD_LIBRARY_PATH": "/hostile/lib",
                "LD_PRELOAD": "/hostile/inject.so",
                "PYTHONPATH": "/hostile/python",
            }
            with mock.patch.dict(
                "noema_lab.core.resource_guard.os.environ",
                hostile,
                clear=True,
            ):
                environment = supervisor._child_environment()
                arguments = supervisor._systemd_environment_arguments(
                    environment=environment
                )

        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(environment["OMP_NUM_THREADS"], "1")
        self.assertEqual(environment["NUMEXPR_NUM_THREADS"], "1")
        for name in (
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "CUDA_DEVICE_ORDER",
        ):
            self.assertNotIn(name, environment)
        self.assertIn("--setenv=CUDA_VISIBLE_DEVICES=", arguments)
        self.assertIn("--setenv=OMP_NUM_THREADS=1", arguments)
        self.assertIn("--setenv=NUMEXPR_NUM_THREADS=1", arguments)
        self.assertIn("--setenv=LD_LIBRARY_PATH=", arguments)
        self.assertIn("--setenv=LD_PRELOAD=", arguments)
        self.assertIn("--setenv=CUDA_DEVICE_ORDER=", arguments)
        self.assertNotIn("--setenv=CUDA_VISIBLE_DEVICES=0,1", arguments)

    @mock.patch(
        "noema_lab.core.resource_guard._systemd_user_scope_available",
        return_value=False,
    )
    @mock.patch(
        "noema_lab.core.resource_guard._available_memory_bytes",
        return_value=5 * GIB,
    )
    def test_exact_environment_controls_are_passed_to_popen(
        self,
        _available,
        _systemd,
    ):
        process = mock.Mock()
        process.pid = 12345
        process.poll.return_value = 0
        process.wait.return_value = 0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supervisor = self._supervisor(
                root,
                ["/bin/true"],
                MemoryGuardConfig(reserve_bytes=512 * MIB),
                environment_set={"OMP_NUM_THREADS": "1"},
                environment_unset=("LD_LIBRARY_PATH",),
            )
            with (
                mock.patch.dict(
                    "noema_lab.core.resource_guard.os.environ",
                    {
                        "OMP_NUM_THREADS": "64",
                        "LD_LIBRARY_PATH": "/hostile/lib",
                    },
                    clear=True,
                ),
                mock.patch(
                    "noema_lab.core.resource_guard.subprocess.Popen",
                    return_value=process,
                ) as popen,
                mock.patch(
                    "noema_lab.core.resource_guard._read_json_if_present",
                    return_value={"status": "completed"},
                ),
            ):
                result = supervisor._run_admitted()

        self.assertEqual(result.status, "completed")
        launched_environment = popen.call_args.kwargs["env"]
        self.assertEqual(launched_environment["OMP_NUM_THREADS"], "1")
        self.assertNotIn("LD_LIBRARY_PATH", launched_environment)

    def test_environment_controls_reject_ambiguous_or_invalid_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(
                IsolatedJobError,
                "both set and unset",
            ):
                self._supervisor(
                    root,
                    ["/bin/true"],
                    MemoryGuardConfig(reserve_bytes=512 * MIB),
                    environment_set={"OMP_NUM_THREADS": "1"},
                    environment_unset=("OMP_NUM_THREADS",),
                )
            with self.assertRaisesRegex(IsolatedJobError, "invalid name"):
                self._supervisor(
                    root,
                    ["/bin/true"],
                    MemoryGuardConfig(reserve_bytes=512 * MIB),
                    environment_set={"NOT-AN-ENV-NAME": "1"},
                )

    @mock.patch("noema_lab.core.resource_guard._systemd_user_scope_available", return_value=False)
    @mock.patch("noema_lab.core.resource_guard._available_memory_bytes", return_value=5 * GIB)
    def test_explicit_max_cannot_consume_the_protected_reserve(self, _available, _systemd):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self._supervisor(
                Path(tmp),
                [sys.executable, "-c", "import sys; sys.exit(75)"],
                MemoryGuardConfig(reserve_bytes=2 * GIB, explicit_max_bytes=10 * GIB),
            )
            with self.assertRaises(ResourceExhausted) as raised:
                supervisor.run()
        self.assertEqual(raised.exception.evidence["memory_max_bytes"], 3 * GIB)
        self.assertFalse(raised.exception.evidence["hard_limit_enforced"])

    @mock.patch("noema_lab.core.resource_guard._systemd_user_scope_available", return_value=False)
    def test_watchdog_terminates_process_group_at_host_reserve(self, _systemd):
        readings = iter([2 * GIB, 400 * MIB, 400 * MIB])
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "noema_lab.core.resource_guard._available_memory_bytes",
            side_effect=lambda: next(readings, 400 * MIB),
        ):
            supervisor = self._supervisor(
                Path(tmp),
                [sys.executable, "-c", "import time; time.sleep(30)"],
                MemoryGuardConfig(reserve_bytes=512 * MIB, poll_hz=50),
            )
            with self.assertRaises(ResourceExhausted) as raised:
                supervisor.run()
        self.assertEqual(raised.exception.evidence["termination_reason"], "system_memory_reserve")
        self.assertIsNotNone(supervisor.process)
        self.assertIsNotNone(supervisor.process.poll())

    @mock.patch("noema_lab.core.resource_guard._systemd_user_scope_available", return_value=False)
    @mock.patch("noema_lab.core.resource_guard._available_memory_bytes", return_value=5 * GIB)
    def test_deadline_terminates_worker_process_group(self, _available, _systemd):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self._supervisor(
                Path(tmp),
                [sys.executable, "-c", "import time; time.sleep(30)"],
                MemoryGuardConfig(
                    reserve_bytes=512 * MIB,
                    poll_hz=50,
                    timeout_seconds=0.05,
                ),
            )
            with self.assertRaises(ExecutionDeadlineExceeded) as raised:
                supervisor.run()
        self.assertEqual(
            raised.exception.evidence["termination_reason"],
            "execution_deadline",
        )
        self.assertIsNotNone(supervisor.process)
        self.assertIsNotNone(supervisor.process.poll())

    @mock.patch("noema_lab.core.resource_guard._available_memory_bytes", return_value=5 * GIB)
    def test_deadline_also_bounds_waiting_for_execution_admission(self, _available):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self._supervisor(
                Path(tmp),
                [sys.executable, "-c", "pass"],
                MemoryGuardConfig(
                    reserve_bytes=512 * MIB,
                    timeout_seconds=0.05,
                ),
            )
            resource_guard_module._EXECUTION_ADMISSION_LOCK.acquire()
            try:
                with self.assertRaises(ExecutionDeadlineExceeded) as raised:
                    supervisor.run()
            finally:
                resource_guard_module._EXECUTION_ADMISSION_LOCK.release()
        self.assertEqual(
            raised.exception.evidence["termination_reason"],
            "execution_deadline",
        )
        self.assertIsNone(supervisor.process)

    def test_systemd_command_requests_cgroup_wide_oom_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self._supervisor(
                Path(tmp),
                ["/bin/true"],
                MemoryGuardConfig(reserve_bytes=512 * MIB),
            )
            supervisor.evidence = {}
            command = supervisor._systemd_scope_command(["/bin/true"], GIB, 2 * GIB)
        rendered = " ".join(command)
        self.assertIn("MemoryHigh=%d" % GIB, rendered)
        self.assertIn("MemoryMax=%d" % (2 * GIB), rendered)
        self.assertIn("OOMPolicy=kill", rendered)
        self.assertIn("KillMode=control-group", rendered)
        self.assertNotIn("MemoryOOMGroup", rendered)

    def test_systemd_inspection_and_cleanup_failures_are_explicit_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self._supervisor(
                Path(tmp),
                ["/bin/true"],
                MemoryGuardConfig(reserve_bytes=512 * MIB),
            )
            supervisor.evidence = {"cgroup_unit": "noema-job-test.service"}
            with mock.patch(
                "noema_lab.core.resource_guard.subprocess.run",
                side_effect=OSError("systemctl unavailable"),
            ):
                evidence = supervisor._systemd_unit_evidence()
            self.assertEqual(evidence["systemd_inspection_status"], "failed")
            self.assertIn("systemctl unavailable", evidence["systemd_inspection_error"])

            show = mock.Mock(returncode=0, stdout="Result=exit-code\n", stderr="")
            reset = mock.Mock(returncode=1, stdout="", stderr="failed")
            with mock.patch(
                "noema_lab.core.resource_guard.subprocess.run",
                side_effect=[show, reset],
            ) as run_command:
                evidence = supervisor._systemd_unit_evidence()
            self.assertEqual(evidence["systemd_inspection_status"], "ok")
            self.assertEqual(evidence["systemd_cleanup_status"], "failed")
            self.assertEqual(run_command.call_count, 2)

    def test_successful_systemd_unit_needs_no_failed_state_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self._supervisor(
                Path(tmp),
                ["/bin/true"],
                MemoryGuardConfig(reserve_bytes=512 * MIB),
            )
            supervisor.evidence = {"cgroup_unit": "noema-job-test.service"}
            show = mock.Mock(
                returncode=0,
                stdout="Result=success\nMemoryPeak=123\nControlGroup=\n",
                stderr="",
            )
            with mock.patch(
                "noema_lab.core.resource_guard.subprocess.run",
                return_value=show,
            ) as run_command:
                evidence = supervisor._systemd_unit_evidence()
        self.assertEqual(evidence["systemd_result"], "success")
        self.assertEqual(evidence["systemd_cleanup_status"], "ok")
        self.assertEqual(run_command.call_count, 1)

    def test_live_cgroup_peak_and_events_survive_post_exit_empty_properties(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self._supervisor(
                Path(tmp),
                ["/bin/true"],
                MemoryGuardConfig(reserve_bytes=512 * MIB),
            )
            supervisor.evidence = {"cgroup_unit": "noema-job-test.service"}
            supervisor._cgroup_path = Path("/sys/fs/cgroup/live-test")
            supervisor._cgroup_memory_sample_count = 4
            supervisor._cgroup_kernel_peak_bytes = 123456
            supervisor._cgroup_sampled_current_peak_bytes = 120000
            supervisor._cgroup_memory_events = {
                "oom": 0,
                "oom_kill": 0,
                "oom_group_kill": 0,
            }
            show = mock.Mock(
                returncode=0,
                stdout="Result=success\nMemoryPeak=\nControlGroup=\n",
                stderr="",
            )
            reset = mock.Mock(returncode=0, stdout="", stderr="")
            with mock.patch(
                "noema_lab.core.resource_guard.subprocess.run",
                side_effect=[show, reset],
            ):
                evidence = supervisor._systemd_unit_evidence()
        self.assertEqual(evidence["peak_memory_bytes"], 123456)
        self.assertEqual(
            evidence["peak_memory_source"],
            "cgroup_v2_memory.peak_live",
        )
        self.assertEqual(evidence["cgroup_memory_sample_count"], 4)
        self.assertTrue(evidence["cgroup_control_path_observed_live"])
        self.assertEqual(evidence["memory_events"]["oom_kill"], 0)

    def test_live_cgroup_sampler_retains_maxima_and_event_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            cgroup = Path(tmp) / "cgroup"
            cgroup.mkdir()
            (cgroup / "memory.peak").write_text("1000\n", encoding="ascii")
            (cgroup / "memory.current").write_text("900\n", encoding="ascii")
            (cgroup / "memory.events").write_text(
                "oom 0\noom_kill 0\noom_group_kill 0\n",
                encoding="ascii",
            )
            supervisor = self._supervisor(
                Path(tmp),
                ["/bin/true"],
                MemoryGuardConfig(reserve_bytes=512 * MIB),
            )
            with mock.patch.object(
                supervisor,
                "_discover_systemd_cgroup_path",
                return_value=cgroup,
            ):
                supervisor._sample_systemd_cgroup_memory()
                (cgroup / "memory.peak").write_text("2000\n", encoding="ascii")
                (cgroup / "memory.current").write_text("1500\n", encoding="ascii")
                (cgroup / "memory.events").write_text(
                    "oom 1\noom_kill 1\noom_group_kill 0\n",
                    encoding="ascii",
                )
                supervisor._sample_systemd_cgroup_memory()
        self.assertEqual(supervisor._cgroup_memory_sample_count, 2)
        self.assertEqual(supervisor._cgroup_kernel_peak_bytes, 2000)
        self.assertEqual(supervisor._cgroup_sampled_current_peak_bytes, 1500)
        self.assertEqual(supervisor._cgroup_memory_events["oom"], 1)
        self.assertEqual(supervisor._cgroup_memory_events["oom_kill"], 1)

    @unittest.skipUnless(
        Path("/sys/fs/cgroup/user.slice").is_dir(),
        "requires a Linux cgroup-v2 user.slice fixture path",
    )
    def test_live_cgroup_discovery_retries_caches_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self._supervisor(
                Path(tmp),
                ["/bin/true"],
                MemoryGuardConfig(reserve_bytes=512 * MIB),
            )
            supervisor.evidence = {"cgroup_unit": "noema-job-test.service"}
            missing = mock.Mock(returncode=1, stdout="", stderr="not ready")
            found = mock.Mock(
                returncode=0,
                stdout="/user.slice\n",
                stderr="",
            )
            with mock.patch(
                "noema_lab.core.resource_guard.subprocess.run",
                side_effect=[missing, found],
            ) as run:
                self.assertIsNone(supervisor._discover_systemd_cgroup_path())
                discovered = supervisor._discover_systemd_cgroup_path()
                cached = supervisor._discover_systemd_cgroup_path()
            self.assertEqual(run.call_count, 2)
            self.assertEqual(discovered, Path("/sys/fs/cgroup/user.slice"))
            self.assertEqual(cached, discovered)

            escaped = self._supervisor(
                Path(tmp),
                ["/bin/true"],
                MemoryGuardConfig(reserve_bytes=512 * MIB),
            )
            escaped.evidence = {"cgroup_unit": "noema-job-test.service"}
            response = mock.Mock(
                returncode=0,
                stdout="/../../tmp\n",
                stderr="",
            )
            with mock.patch(
                "noema_lab.core.resource_guard.subprocess.run",
                return_value=response,
            ):
                self.assertIsNone(escaped._discover_systemd_cgroup_path())

    def test_worker_status_distinguishes_missing_from_malformed(self):
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "status.json"
            self.assertEqual(_read_json_if_present(status_path), {})
            status_path.write_text(
                '{"status":"failed","status":"completed"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                IsolatedJobError,
                "Duplicate JSON object key",
            ):
                _read_json_if_present(status_path)

    def test_worker_event_stream_fails_on_ambiguous_json_and_sink_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supervisor = self._supervisor(
                root,
                ["/bin/true"],
                MemoryGuardConfig(reserve_bytes=512 * MIB),
            )
            supervisor.control_dir.mkdir(parents=True)
            supervisor.events_path.write_text(
                '{"kind":"first","kind":"second"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                IsolatedJobError,
                "Duplicate JSON object key",
            ):
                supervisor._drain_events()

            supervisor._event_offset = 0
            supervisor.events_path.write_text('{"kind":"valid"}\n', encoding="utf-8")
            supervisor.event_sink = lambda _event: (_ for _ in ()).throw(
                RuntimeError("sink failed")
            )
            with self.assertRaisesRegex(RuntimeError, "sink failed"):
                supervisor._drain_events()

    def test_run_job_recovers_running_bundle_after_resource_termination(self):
        evidence = {
            "kind": "noema.execution_resource_guard",
            "termination_reason": "system_memory_reserve",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalStore(root / ".noema")
            recipe = recipe_from_dict(
                {
                    "schema_version": 1,
                    "name": "guarded_recipe",
                    "steps": [
                        {
                            "id": "data",
                            "op": "source.random_bits",
                            "inputs": {},
                            "params": {"bit_count": 8},
                        }
                    ],
                }
            )

            class FakeSupervisor:
                def __init__(self, *, event_sink, **_kwargs):
                    self.event_sink = event_sink

                def cancel(self):
                    return None

                def run(self):
                    run_dir = store.create_run_dir("guarded_recipe")
                    store.write_json(
                        run_dir / "summary.json",
                        {"schema_version": 1, "run_id": run_dir.name, "status": "running", "steps": [], "metrics": {}},
                    )
                    store.write_json(
                        run_dir / "manifest.json",
                        {"schema_version": 1, "run_id": run_dir.name, "status": "running"},
                    )
                    self.event_sink({"kind": "run_created", "message": "created", "run_id": run_dir.name})
                    self.event_sink({"kind": "step_started", "message": "started", "step_id": "data", "op": "source.random_bits"})
                    raise ResourceExhausted("protected reserve reached", evidence)

            job = RunJob(
                recipe,
                build_registry(),
                store,
                ExecutionPlanCache(),
                {"parallel_workers": 1, "use_plan_cache": True},
                root,
            )
            with mock.patch("noema_lab.ui.server.IsolatedJobSupervisor", FakeSupervisor):
                job._run()

            summary = store.get_run(str(job.run_id))
            manifest = store.get_manifest(str(job.run_id))

        self.assertEqual(job.status, "resource_exhausted")
        self.assertEqual(job.progress["data"]["phase"], "resource_exhausted")
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["failure"]["kind"], "resource_exhausted")
        self.assertEqual(manifest["status"], "failed")


if __name__ == "__main__":
    unittest.main()
