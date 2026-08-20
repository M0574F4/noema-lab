import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.cli.main import main
from noema_lab.core.benchmarks import load_benchmark_pack, validate_benchmark_pack
from noema_lab.core.executor import LocalExecutor
from noema_lab.core.lint import lint_recipe_invariants
from noema_lab.core.recipes import load_recipe, recipe_from_dict
from noema_lab.core.storage import LocalStore
from noema_lab.ops import build_registry


def _protected_jpeg_recipe(image_path: Path, snr_db: float):
    return recipe_from_dict(
        {
            "schema_version": 1,
            "name": "jpeg_crc_repetition_awgn_smoke",
            "metadata": {
                "seed": 17,
                "rate_count_fixed_point": "tx_bit_boundary.channel.fixed.tx.bit_count",
            },
            "steps": [
                {
                    "id": "data",
                    "op": "source.local_npz_images",
                    "params": {"path": str(image_path), "array": "images"},
                },
                {
                    "id": "sender",
                    "op": "model.jpeg_encode",
                    "inputs": {"images": "data.images"},
                    "params": {"quality": 75, "subsampling": "420", "optimize": False, "progressive": False},
                },
                {
                    "id": "payload_bit_boundary",
                    "op": "channel.bit_boundary",
                    "inputs": {"bits": "sender.bits"},
                    "params": {"label": "payload", "role": "payload"},
                },
                {
                    "id": "packetizer",
                    "op": "channel.packetize_crc32",
                    "inputs": {"bits": "payload_bit_boundary.bits"},
                    "params": {"packet_payload_bits": 512},
                },
                {
                    "id": "channel_encoder",
                    "op": "channel.repetition_encoder",
                    "inputs": {"bits": "packetizer.bits"},
                    "params": {"factor": 3},
                },
                {
                    "id": "tx_bit_boundary",
                    "op": "channel.bit_boundary",
                    "inputs": {"bits": "channel_encoder.coded_bits"},
                    "params": {"label": "tx", "role": "protected_link_input"},
                },
                {
                    "id": "modulator",
                    "op": "modulation.digital_modulate",
                    "inputs": {"bits": "tx_bit_boundary.bits"},
                    "params": {"modulation": "qpsk"},
                },
                {
                    "id": "wireless_channel",
                    "op": "wireless.channel",
                    "inputs": {"symbols": "modulator.symbols"},
                    "params": {"channel": "awgn", "snr_db": snr_db, "wireless_backend": "numpy", "seed": 123},
                },
                {
                    "id": "demodulator",
                    "op": "demodulation.digital_demodulate",
                    "inputs": {"rx_symbols": "wireless_channel.rx_symbols"},
                    "params": {"modulation": "auto"},
                },
                {
                    "id": "rx_bit_boundary",
                    "op": "channel.bit_boundary",
                    "inputs": {"bits": "demodulator.bits"},
                    "params": {"label": "rx", "role": "protected_link_output"},
                },
                {
                    "id": "coded_ber",
                    "op": "metrics.bit_error_rate",
                    "inputs": {"reference": "tx_bit_boundary.bits", "candidate": "rx_bit_boundary.bits"},
                    "params": {"label": "coded"},
                },
                {
                    "id": "channel_bit_count_match",
                    "op": "channel.bit_count_match",
                    "inputs": {"reference": "tx_bit_boundary.bits", "candidate": "rx_bit_boundary.bits"},
                    "params": {"label": "protected_link_io"},
                },
                {
                    "id": "channel_decoder",
                    "op": "channel.repetition_decoder",
                    "inputs": {"coded_bits": "rx_bit_boundary.bits"},
                    "params": {"factor": 3},
                },
                {
                    "id": "crc_check",
                    "op": "channel.crc32_check",
                    "inputs": {"bits": "channel_decoder.bits"},
                    "params": {"on_decode_failure": "gray_image"},
                },
                {
                    "id": "payload_ber",
                    "op": "metrics.bit_error_rate",
                    "inputs": {"reference": "payload_bit_boundary.bits", "candidate": "crc_check.bits"},
                    "params": {"label": "payload"},
                },
                {
                    "id": "receiver",
                    "op": "model.jpeg_decode",
                    "inputs": {"bits": "crc_check.bits"},
                    "params": {"on_error": "gray_image"},
                },
                {
                    "id": "evaluation",
                    "op": "metrics.image_reconstruction",
                    "inputs": {"reference": "data.images", "reconstruction": "receiver.images"},
                },
            ],
        }
    )


def _step(summary, step_id):
    return next(step for step in summary["steps"] if step["id"] == step_id)


class ProtectedDigitalTests(unittest.TestCase):
    def test_crc_repetition_awgn_link_reports_snr_dependent_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = np.full((1, 32, 32, 3), 90, dtype=np.uint8)
            images[:, 8:24, 8:24, :] = [20, 190, 230]
            image_path = root / "images.npz"
            np.savez_compressed(image_path, images=images)
            registry = build_registry()

            def run_at(snr_db):
                recipe = _protected_jpeg_recipe(image_path, snr_db)
                store = LocalStore(root / ("workspace_%s" % str(snr_db).replace("-", "m")))
                run_dir = LocalExecutor(registry, store).run(recipe)
                return store.get_run(run_dir.name)

            high = run_at(30.0)
            low = run_at(-8.0)
            high_crc = _step(high, "crc_check")["metrics"]
            low_crc = _step(low, "crc_check")["metrics"]
            high_ber = _step(high, "coded_ber")["metrics"]["channel.coded.ber"]
            low_ber = _step(low, "coded_ber")["metrics"]["channel.coded.ber"]

            self.assertEqual(high_crc["channel.packet_success_rate"], 1.0)
            self.assertEqual(high_crc["channel.outage_rate"], 0.0)
            self.assertLess(low_crc["channel.packet_success_rate"], high_crc["channel.packet_success_rate"])
            self.assertGreater(low_crc["channel.outage_rate"], 0.0)
            self.assertGreater(low_ber, high_ber)
            self.assertIn("channel.payload.ber", _step(low, "payload_ber")["metrics"])

    def test_real_protected_baseline_recipes_lint(self):
        registry = build_registry()
        for recipe_name in [
            "jpeg_q75_kodak_repetition_awgn.yaml",
            "compressai_q3_kodak_repetition_awgn.yaml",
        ]:
            recipe = load_recipe(ROOT / "recipes" / recipe_name)
            lint = lint_recipe_invariants(recipe, registry, strict=True)
            self.assertEqual(lint["status"], "passed", lint)
            ops = {step.op for step in recipe.steps}
            self.assertIn("channel.packetize_crc32", ops)
            self.assertIn("channel.crc32_check", ops)
            self.assertIn("channel.repetition_encoder", ops)
            self.assertIn("channel.repetition_decoder", ops)
            self.assertIn("wireless.channel", ops)

    def test_publication_candidate_baselines_use_nr_ldpc_and_multi_rate_points(self):
        registry = build_registry()
        for recipe_name in [
            "jpeg_q75_kodak_nr_ldpc_awgn.yaml",
            "compressai_q3_kodak_nr_ldpc_awgn.yaml",
        ]:
            recipe = load_recipe(ROOT / "recipes" / recipe_name)
            lint = lint_recipe_invariants(recipe, registry, strict=True)
            self.assertEqual(lint["status"], "passed", lint)
            by_id = {step.id: step for step in recipe.steps}
            self.assertEqual(by_id["channel_encoder"].op, "channel.nr_ldpc_encoder")
            self.assertEqual(by_id["channel_decoder"].op, "channel.nr_ldpc_decoder")
            self.assertEqual(
                by_id["channel_decoder"].inputs["llr"], "demodulator.llr"
            )
            self.assertEqual(
                by_id["resource_accounting"].op,
                "channel.communication_resource_accounting",
            )
            self.assertEqual(by_id["tx_power"].op, "channel.symbol_power_normalize")
            self.assertEqual(
                by_id["tx_power"].params["normalization_scope"], "source_item"
            )

        pack = load_benchmark_pack(
            ROOT
            / "benchmarks"
            / "semantic_comm_v1"
            / "protected_digital_vs_deepjscc_awgn.yaml"
        )
        recipe_ids = {entry.id for entry in pack.recipes}
        for quality in (50, 75, 90):
            for snr in ("m4", "0", "4", "8", "12"):
                self.assertIn(f"jpeg_q{quality}_nr_ldpc_snr_{snr}", recipe_ids)
        for quality in (1, 3, 5):
            for snr in ("m4", "0", "4", "8", "12"):
                self.assertIn(
                    f"compressai_q{quality}_nr_ldpc_snr_{snr}", recipe_ids
                )

    def test_tutorial_uses_an_explicit_capacity_oracle_digital_reference(self):
        tutorial = (ROOT / "docs" / "tutorials" / "digital_vs_deepjscc_sionna.md").read_text(encoding="utf-8")
        self.assertNotIn("python3 - <<", tutorial)
        self.assertNotIn("uv run python - <<", tutorial)
        self.assertIn("build_benchmark.py", tutorial)
        self.assertIn("CLI training summary", tutorial)
        self.assertIn("0.5", tutorial)
        recipe_paths = sorted(set(re.findall(r"recipes/[A-Za-z0-9_./-]+\.yaml", tutorial)))
        capacity_recipes = [
            path
            for path in recipe_paths
            if path.endswith("jpeg_capacity_matched_kodak_awgn.yaml")
        ]
        self.assertEqual(len(capacity_recipes), 1)
        recipe = load_recipe(ROOT / capacity_recipes[0])
        by_id = {step.id: step for step in recipe.steps}
        reference = by_id["wireless_channel"]
        self.assertEqual(reference.op, "channel.jpeg_capacity_oracle")
        self.assertEqual(reference.inputs["images"], "data.images")
        self.assertEqual(reference.params["channel_uses_per_pixel"], 0.5)

    def test_semantic_comm_v1_benchmark_packs_validate(self):
        registry = build_registry()
        for name in [
            "protected_digital_vs_deepjscc_awgn.yaml",
            "adaptive_digital_vs_semantic_awgn.yaml",
        ]:
            pack = load_benchmark_pack(ROOT / "benchmarks" / "semantic_comm_v1" / name)
            self.assertEqual(pack.suite.get("id"), "semantic_comm")
            validation = validate_benchmark_pack(pack, registry, ROOT)
            self.assertEqual(validation["suite"]["id"], "semantic_comm")
            self.assertEqual(validation["benchmark_tier"], "experimental")
            self.assertGreaterEqual(validation["recipe_count"], 1)
            self.assertEqual(validation["catalog_validation"]["status"], "valid")

    def test_benchmark_run_emits_protected_digital_metrics_and_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            images = np.full((1, 32, 32, 3), 70, dtype=np.uint8)
            images[:, 6:26, 8:24, :] = [25, 180, 230]
            image_path = root / "images.npz"
            np.savez_compressed(image_path, images=images)

            recipe_path = root / "protected_jpeg_recipe.json"
            recipe_path.write_text(json.dumps(_protected_jpeg_recipe(image_path, 30.0).to_dict()), encoding="utf-8")
            benchmark_path = root / "protected_benchmark.json"
            benchmark_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "id": "protected_digital_smoke",
                        "version": "1",
                        "name": "Protected Digital Smoke",
                        "dataset": {"id": "images", "modality": "image"},
                        "task": {"id": "image_reconstruction", "kind": "reconstruction", "modality": "image"},
                        "metrics": [
                            {"id": "channel.snr_db"},
                            {"id": "quality.psnr_db"},
                            {"id": "channel.packet_success_rate"},
                            {"id": "channel.outage_rate"},
                            {"id": "channel.transmitted_bit_count"},
                            {
                                "id": "channel.channel_use_count",
                                "source_step": "wireless_channel",
                                "source_operation": "wireless.channel",
                            },
                            {"id": "channel.code_rate"},
                            {"id": "channel.bits_per_symbol"},
                        ],
                        "recipes": [
                            {
                                "id": "jpeg_repetition_snr_m8",
                                "label": "JPEG + protected digital link",
                                "role": "baseline",
                                "path": str(recipe_path),
                                "params": {
                                    "sweep_values": {"channel.snr_db": -8},
                                    "step_params": {"wireless_channel": {"snr_db": -8}},
                                },
                            },
                            {
                                "id": "jpeg_repetition_snr_30",
                                "label": "JPEG + protected digital link",
                                "role": "baseline",
                                "path": str(recipe_path),
                                "params": {
                                    "sweep_values": {"channel.snr_db": 30},
                                    "step_params": {"wireless_channel": {"snr_db": 30}},
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_output = io.StringIO()
            with contextlib.redirect_stdout(run_output):
                code = main(["--workspace", str(workspace), "benchmark", "run", str(benchmark_path)])
            self.assertEqual(code, 0, run_output.getvalue())
            result_line = [line for line in run_output.getvalue().splitlines() if line.startswith("result: ")][0]
            result_path = Path(result_line.split("result: ", 1)[1])
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["recipes"]), 2)
            for row in result["recipes"]:
                metrics = row["metrics"]
                for metric in [
                    "channel.snr_db",
                    "quality.psnr_db",
                    "channel.packet_success_rate",
                    "channel.outage_rate",
                    "channel.transmitted_bit_count",
                    "channel.channel_use_count",
                    "channel.code_rate",
                    "channel.bits_per_symbol",
                ]:
                    self.assertIn(metric, metrics)
            self.assertLess(
                result["recipes"][0]["metrics"]["channel.packet_success_rate"],
                result["recipes"][1]["metrics"]["channel.packet_success_rate"],
            )

            for plot_type, filename in [
                ("graceful-degradation", "graceful_degradation.png"),
                ("packet-success", "packet_success.png"),
            ]:
                plot_output = io.StringIO()
                with contextlib.redirect_stdout(plot_output):
                    code = main(
                        [
                            "--workspace",
                            str(workspace),
                            "benchmark",
                            "plot",
                            result_path.parent.name,
                            "--plot",
                            plot_type,
                            "--out",
                            "figures/%s" % filename,
                            "--json",
                        ]
                    )
                self.assertEqual(code, 0, plot_output.getvalue())
                payload = json.loads(plot_output.getvalue())
                self.assertTrue(Path(payload["plot"]["path"]).is_file())
                self.assertTrue(Path(payload["plot"]["data_csv_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
