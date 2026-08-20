from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from noema_lab.ops.models.deepjscc_checkpoint import (
    CHECKPOINT_ARCHITECTURE as DEEPJSCC_CHECKPOINT_ARCHITECTURE,
    CHECKPOINT_FORMAT as DEEPJSCC_CHECKPOINT_FORMAT,
    CHECKPOINT_KIND as DEEPJSCC_CHECKPOINT_KIND,
)
from noema_lab.ui.server import start_ui_server_in_thread


class TrainedArtifactImportApiTests(unittest.TestCase):
    def test_raw_npz_upload_returns_discoverable_direct_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "trainer-output.npz"
            expected_sha = _write_test_deepset_checkpoint(checkpoint)
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                root / ".workspace",
                root,
            )
            try:
                host, port = server.server_address
                query = urllib.parse.urlencode(
                    {
                        "operation": "model.symbol_power_allocator",
                        "filename": "external allocator.npz",
                        "label": "External allocator",
                    }
                )
                request = urllib.request.Request(
                    "http://%s:%d/api/trained-artifacts/import?%s" % (host, port, query),
                    data=checkpoint.read_bytes(),
                    method="POST",
                    headers={"Content-Type": "application/octet-stream"},
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    self.assertEqual(response.status, 201)
                    payload = json.loads(response.read().decode("utf-8"))

                self.assertEqual(payload["status"], "imported")
                artifact = payload["artifact"]
                self.assertTrue(artifact["ready"])
                self.assertEqual(artifact["label"], "External allocator")
                self.assertEqual(artifact["artifact"]["sha256"], expected_sha)
                binding = artifact["compatible_operations"][0]
                self.assertEqual(binding["operation"], "model.symbol_power_allocator")
                self.assertEqual(binding["params"]["policy"], "learned_checkpoint")
                self.assertEqual(binding["params"]["checkpoint_sha256"], expected_sha)

                list_url = (
                    "http://%s:%d/api/trained-artifacts?operation=model.symbol_power_allocator"
                    % (host, port)
                )
                with urllib.request.urlopen(list_url, timeout=5) as response:
                    discovered = json.loads(response.read().decode("utf-8"))
                self.assertEqual(
                    [item["id"] for item in discovered["artifacts"]],
                    [artifact["id"]],
                )
                self.assertFalse(
                    (root / ".noema" / "trained_artifacts" / ".staging").exists()
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_safe_deepjscc_upload_returns_one_paired_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "deepjscc-output.npz"
            expected_sha = _write_test_deepjscc_checkpoint(checkpoint)
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                root / ".workspace",
                root,
            )
            try:
                host, port = server.server_address
                query = urllib.parse.urlencode(
                    {
                        "operation": "model.deepjscc_external_decode",
                        "filename": "external deepjscc.npz",
                        "label": "External DeepJSCC",
                    }
                )
                request = urllib.request.Request(
                    "http://%s:%d/api/trained-artifacts/import?%s" % (host, port, query),
                    data=checkpoint.read_bytes(),
                    method="POST",
                    headers={"Content-Type": "application/octet-stream"},
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    self.assertEqual(response.status, 201)
                    payload = json.loads(response.read().decode("utf-8"))

                artifact = payload["artifact"]
                self.assertTrue(artifact["ready"], artifact["issues"])
                self.assertEqual(artifact["artifact"]["sha256"], expected_sha)
                self.assertEqual(artifact["application"], {"mode": "all_group_bindings"})
                bindings = artifact["compatible_operations"]
                self.assertEqual(
                    [binding["operation"] for binding in bindings],
                    ["model.deepjscc_external_encode", "model.deepjscc_external_decode"],
                )
                self.assertEqual(
                    [binding["preferred_step_id"] for binding in bindings],
                    ["sender", "receiver"],
                )
                self.assertTrue(all(binding["params"]["symbol_channels"] == 2 for binding in bindings))
                list_url = (
                    "http://%s:%d/api/trained-artifacts?operation=model.deepjscc_external_encode"
                    % (host, port)
                )
                with urllib.request.urlopen(list_url, timeout=5) as response:
                    discovered = json.loads(response.read().decode("utf-8"))
                self.assertEqual(
                    [item["id"] for item in discovered["artifacts"]],
                    [artifact["id"]],
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


def _write_test_deepset_checkpoint(path: Path) -> str:
    metadata = {
        "schema_version": 1,
        "kind": "noema.csi_power_allocator_checkpoint",
        "format": "noema_csi_power_deepset_npz_v1",
        "input_contract": "log(max(gain*average_power/noise_variance,eps))",
        "output_contract": "euclidean_simplex_projection_times_fixed_sum_power",
        "activation": "relu",
        "hidden_dim": 2,
        "training": {
            "objective": "maximize_parallel_channel_shannon_spectral_efficiency",
            "supervised_labels_used": False,
            "water_filling_used_during_training": False,
        },
    }
    np.savez_compressed(
        path,
        feature_mean=np.asarray([0.0], dtype=np.float32),
        feature_scale=np.asarray([1.0], dtype=np.float32),
        phi_weight_0=np.asarray([[1.0], [-1.0]], dtype=np.float32),
        phi_bias_0=np.zeros(2, dtype=np.float32),
        phi_weight_1=np.eye(2, dtype=np.float32),
        phi_bias_1=np.zeros(2, dtype=np.float32),
        rho_weight_0=np.asarray(
            [[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, -1.0]],
            dtype=np.float32,
        ),
        rho_bias_0=np.zeros(2, dtype=np.float32),
        rho_weight_out=np.asarray([[1.0, -1.0]], dtype=np.float32),
        rho_bias_out=np.zeros(1, dtype=np.float32),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_test_deepjscc_checkpoint(path: Path, symbol_channels: int = 2) -> str:
    np.savez_compressed(
        path,
        encoder_0_weight=np.zeros((32, 3, 3, 3), dtype=np.float32),
        encoder_0_bias=np.zeros((32,), dtype=np.float32),
        encoder_2_weight=np.zeros((2 * symbol_channels, 32, 3, 3), dtype=np.float32),
        encoder_2_bias=np.zeros((2 * symbol_channels,), dtype=np.float32),
        decoder_0_weight=np.zeros((2 * symbol_channels, 32, 4, 4), dtype=np.float32),
        decoder_0_bias=np.zeros((32,), dtype=np.float32),
        decoder_2_weight=np.zeros((32, 3, 4, 4), dtype=np.float32),
        decoder_2_bias=np.zeros((3,), dtype=np.float32),
        metadata_json=json.dumps(
            {
                "schema_version": 1,
                "kind": DEEPJSCC_CHECKPOINT_KIND,
                "format": DEEPJSCC_CHECKPOINT_FORMAT,
                "architecture": DEEPJSCC_CHECKPOINT_ARCHITECTURE,
                "input_channels": 3,
                "output_channels": 3,
                "symbol_channels": symbol_channels,
            },
            sort_keys=True,
        ),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
