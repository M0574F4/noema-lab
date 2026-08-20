from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from noema_lab.core import job_worker
from noema_lab.core.runtime_policy import (
    apply_deterministic_cpu_runtime_policy,
    deterministic_cpu_runtime_policy_sha256_v1,
    deterministic_cpu_runtime_policy_v1,
    observe_deterministic_cpu_runtime_policy,
    runtime_policy_environment,
    runtime_policy_lifecycle_evidence,
    validate_deterministic_cpu_runtime_policy_evidence,
    validate_runtime_policy_lifecycle_evidence,
)


class _FakeDtype:
    def __str__(self) -> str:
        return "torch.float32"


def _fake_torch() -> types.ModuleType:
    module = types.ModuleType("torch")
    module.__version__ = "test-torch"
    module.float32 = _FakeDtype()
    state = {
        "intra": 9,
        "interop": 7,
        "deterministic": False,
        "dtype": module.float32,
        "device": "cuda:0",
    }

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def device_count() -> int:
            return 0

    module.cuda = _Cuda()
    module.set_num_threads = lambda value: state.__setitem__("intra", value)
    module.get_num_threads = lambda: state["intra"]
    module.set_num_interop_threads = lambda value: state.__setitem__(
        "interop", value
    )
    module.get_num_interop_threads = lambda: state["interop"]
    module.use_deterministic_algorithms = lambda value: state.__setitem__(
        "deterministic", value
    )
    module.are_deterministic_algorithms_enabled = lambda: state[
        "deterministic"
    ]
    module.set_default_dtype = lambda value: state.__setitem__("dtype", value)
    module.get_default_dtype = lambda: state["dtype"]
    module.set_default_device = lambda value: state.__setitem__("device", value)
    module.get_default_device = lambda: state["device"]
    return module


def _fake_sionna() -> tuple[types.ModuleType, types.ModuleType]:
    sionna = types.ModuleType("sionna")
    sionna.__version__ = "test-sionna"
    phy = types.ModuleType("sionna.phy")
    phy.config = types.SimpleNamespace(device="cuda:0", precision="double")
    sionna.phy = phy
    return sionna, phy


class DeterministicCpuRuntimePolicyTests(unittest.TestCase):
    def test_policy_is_closed_canonical_and_omits_ignored_python_hash_seed(self):
        first = deterministic_cpu_runtime_policy_v1()
        second = deterministic_cpu_runtime_policy_v1()
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertRegex(
            deterministic_cpu_runtime_policy_sha256_v1(),
            r"^[0-9a-f]{64}$",
        )
        process = first["process_environment"]
        self.assertNotIn("PYTHONHASHSEED", process["set"])
        self.assertEqual(process["set"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(process["set"]["OMP_NUM_THREADS"], "1")
        self.assertIn("LD_LIBRARY_PATH", process["unset"])
        self.assertIn("LD_PRELOAD", process["unset"])
        self.assertIn("NVIDIA_VISIBLE_DEVICES", process["unset"])
        self.assertIn("CUBLAS_WORKSPACE_CONFIG", process["unset"])
        self.assertIn("XLA_FLAGS", process["unset"])
        self.assertIn("TORCH_HOME", process["unset"])
        self.assertIn("TORCHINDUCTOR_CACHE_DIR", process["unset"])

        changed = copy.deepcopy(first)
        changed["torch"]["inter_op_threads"] = 2
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(ValueError, "not the exact v1 policy"):
                runtime_policy_environment(changed, workspace=Path(raw))

    def test_policy_bootstrap_sets_and_verifies_torch_and_sionna(self):
        policy = deterministic_cpu_runtime_policy_v1()
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            environment_set, _environment_unset = runtime_policy_environment(
                policy, workspace=workspace
            )
            fake_torch = _fake_torch()
            fake_sionna, fake_phy = _fake_sionna()
            with (
                mock.patch.dict(os.environ, environment_set, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "torch": fake_torch,
                        "sionna": fake_sionna,
                        "sionna.phy": fake_phy,
                    },
                ),
            ):
                evidence = apply_deterministic_cpu_runtime_policy(
                    policy, workspace=workspace
                )

        self.assertEqual(evidence["status"], "passed")
        self.assertEqual(
            evidence["policy_sha256"],
            deterministic_cpu_runtime_policy_sha256_v1(),
        )
        self.assertRegex(evidence["evidence_sha256"], r"^[0-9a-f]{64}$")
        observed = evidence["observed"]
        self.assertTrue(
            all(
                value is None
                for value in observed["process_environment"]["unset"].values()
            )
        )
        self.assertEqual(observed["torch"]["device"], "cpu")
        self.assertEqual(observed["torch"]["intra_op_threads"], 1)
        self.assertEqual(observed["torch"]["inter_op_threads"], 1)
        self.assertTrue(observed["torch"]["deterministic_algorithms"])
        self.assertEqual(observed["torch"]["default_dtype"], "float32")
        self.assertEqual(observed["sionna"]["device"], "cpu")
        self.assertEqual(observed["sionna"]["precision"], "single")
        self.assertEqual(
            validate_deterministic_cpu_runtime_policy_evidence(evidence),
            evidence,
        )
        changed = copy.deepcopy(evidence)
        changed["observed"]["torch"]["inter_op_threads"] = 2
        with self.assertRaisesRegex(ValueError, "commitment changed"):
            validate_deterministic_cpu_runtime_policy_evidence(changed)

    def test_policy_bootstrap_rejects_hostile_ambient_before_backend_import(self):
        policy = deterministic_cpu_runtime_policy_v1()
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            environment_set, _environment_unset = runtime_policy_environment(
                policy, workspace=workspace
            )
            hostile = dict(environment_set)
            hostile["OMP_NUM_THREADS"] = "64"
            with mock.patch.dict(os.environ, hostile, clear=True):
                with self.assertRaisesRegex(RuntimeError, "OMP_NUM_THREADS"):
                    apply_deterministic_cpu_runtime_policy(
                        policy, workspace=workspace
                    )

    def test_policy_free_resource_accounting_probe_remains_supported(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            workspace = root / "workspace"
            workspace.mkdir()
            request_path = root / "request.json"
            status_path = root / "status.json"
            events_path = root / "events.jsonl"
            request_path.write_text(
                json.dumps(
                    {
                        "kind": "resource_accounting_probe",
                        "workspace": str(workspace),
                        "project_root": str(root),
                        "adapter_paths": [],
                        "allocation_bytes": 1024 * 1024,
                        "hold_seconds": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                job_worker,
                "build_registry",
                side_effect=AssertionError("policy-free probe built registry"),
            ):
                return_code = job_worker.run_worker(
                    request_path,
                    status_path,
                    events_path,
                )
            status = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(return_code, 0)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["kind"], "resource_accounting_probe")
        self.assertNotIn("runtime_policy_evidence", status)
        self.assertNotIn("source_closure_evidence", status)

    def test_lifecycle_binds_initial_and_final_and_rejects_tampering(self):
        policy = deterministic_cpu_runtime_policy_v1()
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            environment_set, _ = runtime_policy_environment(
                policy, workspace=workspace
            )
            fake_torch = _fake_torch()
            fake_sionna, fake_phy = _fake_sionna()
            with (
                mock.patch.dict(os.environ, environment_set, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "torch": fake_torch,
                        "sionna": fake_sionna,
                        "sionna.phy": fake_phy,
                    },
                ),
            ):
                initial = apply_deterministic_cpu_runtime_policy(
                    policy, workspace=workspace
                )
                final = observe_deterministic_cpu_runtime_policy(
                    policy,
                    workspace=workspace,
                    observation_phase="worker_final",
                    prior_evidence_sha256=initial["evidence_sha256"],
                )
                lifecycle = runtime_policy_lifecycle_evidence(initial, final)
        self.assertEqual(
            validate_runtime_policy_lifecycle_evidence(lifecycle), lifecycle
        )
        changed = copy.deepcopy(lifecycle)
        changed["final"]["prior_evidence_sha256"] = "0" * 64
        changed["final"]["evidence_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_runtime_policy_lifecycle_evidence(changed)

    def test_real_stack_isolated_subprocess_normalizes_bootstrap_and_validates_final(self):
        policy = deterministic_cpu_runtime_policy_v1()
        with tempfile.TemporaryDirectory(prefix="noema-policy-real-stack-") as raw:
            workspace = Path(raw) / "workspace"
            workspace.mkdir()
            pycache = Path(raw) / "pycache"
            environment_set, environment_unset = runtime_policy_environment(
                policy, workspace=workspace
            )
            environment = os.environ.copy()
            for name in environment_unset:
                environment.pop(name, None)
            environment.update(environment_set)
            code = (
                "import json; from pathlib import Path; "
                "from noema_lab.core.runtime_policy import "
                "deterministic_cpu_runtime_policy_v1 as p, "
                "apply_deterministic_cpu_runtime_policy as a, "
                "observe_deterministic_cpu_runtime_policy as o, "
                "runtime_policy_lifecycle_evidence as l, "
                "validate_runtime_policy_lifecycle_evidence as v; "
                "w=Path(%r); i=a(p(), workspace=w); "
                "f=o(p(), workspace=w, observation_phase='worker_final', "
                "prior_evidence_sha256=i['evidence_sha256']); "
                "e=l(i,f); v(e); print(json.dumps({'status':'passed', "
                "'unset':f['observed']['process_environment']['unset']}))"
            ) % str(workspace)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-X",
                    "pycache_prefix=%s" % pycache,
                    "-c",
                    code,
                ],
                cwd=str(Path(__file__).resolve().parents[1]),
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        value = json.loads(completed.stdout)
        self.assertEqual(value["status"], "passed")
        self.assertTrue(all(raw is None for raw in value["unset"].values()))


if __name__ == "__main__":
    unittest.main()
