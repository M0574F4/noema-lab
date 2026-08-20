from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from noema_lab.core.artifacts import Artifact
from noema_lab.core.operations import OperationContext
from noema_lab.core.recipes import load_recipe
from noema_lab.ops import build_registry
from noema_lab.ops.channel.digital import WirelessChannelOperation
from noema_lab.ops.models.learned_codecs import JpegCapacityOracleOperation
from noema_lab.training.exporter import export_differentiable_scenario


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "deepjscc_kodak_slow_rayleigh_train.yaml"
PLAN = (
    ROOT
    / "demo_trainings"
    / "deepjscc_image_reconstruction"
    / "training_plan.yaml"
)


def _context(
    root: Path,
    step_id: str,
    params: dict,
    inputs: dict[str, Artifact],
) -> OperationContext:
    step_dir = root / step_id
    step_dir.mkdir()
    return OperationContext(
        recipe_name="slow_rayleigh_pairing_test",
        step_id=step_id,
        params=params,
        inputs=inputs,
        run_dir=root,
        step_dir=step_dir,
    )


class DeepJsccSlowFadingDemoTests(unittest.TestCase):
    def test_export_selects_blind_fading_and_nested_bandwidth(self):
        recipe = load_recipe(RECIPE)
        plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "bundle"
            result = export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=plan["selected_steps"],
                loss=plan["objective"],
                framework=plan["framework"],
                out_dir=out,
                exporter="deepjscc-image",
                include_starter=True,
                source_path=RECIPE,
                project_root=ROOT,
            )
            self.assertEqual(result["starter_exporter"], "deepjscc-image")
            starter = out / "reference_training"
            config = yaml.safe_load(
                (starter / "train_config.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(config["channel"]["type"], "flat_rayleigh")
            self.assertEqual(
                config["channel"]["receiver_processing"], "none"
            )
            self.assertEqual(
                config["channel"]["fading_scope"], "source_item"
            )
            self.assertEqual(
                config["model"]["symbol_channel_options"], [8, 16, 32]
            )
            self.assertTrue(
                (starter / "build_slow_fading_benchmark.py").is_file()
            )
            post_training = result["project_manifest"]["external_training"][
                "optional_demo_scaffold"
            ]["post_training"]
            self.assertEqual(
                post_training["command"],
                "python build_slow_fading_benchmark.py",
            )

    def test_digital_and_learned_paths_share_source_item_fades(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_ids = ["first", "second"]
            images = np.stack(
                [
                    np.full((8, 8, 3), 40, dtype=np.uint8),
                    np.full((8, 8, 3), 180, dtype=np.uint8),
                ]
            )
            image_metadata = {
                "shape": list(images.shape),
                "original_shapes": [[1, 8, 8, 3], [1, 8, 8, 3]],
                "source_item_ids": image_ids,
            }
            image_path = root / "images.npz"
            np.savez_compressed(
                image_path,
                images=images,
                metadata_json=json.dumps(image_metadata),
            )
            image_artifact = Artifact(
                "image.batch.numpy", image_path, image_metadata
            )

            counts = [32, 32]
            symbols = np.ones(sum(counts), dtype=np.complex64)
            symbol_metadata = {
                "source_item_symbol_counts": counts,
                "source_item_ids": image_ids,
                "source_item_pixel_counts": [64, 64],
            }
            symbol_path = root / "symbols.npz"
            np.savez_compressed(
                symbol_path,
                symbols=symbols,
                metadata_json=json.dumps(symbol_metadata),
            )
            symbol_artifact = Artifact(
                "channel.symbols.complex_numpy",
                symbol_path,
                symbol_metadata,
            )
            seed = 81001
            learned = WirelessChannelOperation().run(
                _context(
                    root,
                    "learned_channel",
                    {
                        "channel": "flat_rayleigh",
                        "snr_db": 10.0,
                        "wireless_backend": "numpy",
                        "receiver_processing": "none",
                        "fading_scope": "source_item",
                        "seed": seed,
                    },
                    {"symbols": symbol_artifact},
                )
            )
            digital = JpegCapacityOracleOperation().run(
                _context(
                    root,
                    "digital_channel",
                    {
                        "channel_model": "slow_rayleigh",
                        "snr_db": 10.0,
                        "channel_uses_per_pixel": 0.5,
                        "on_outage": "image_channel_mean",
                        "seed": seed,
                    },
                    {"images": image_artifact},
                )
            )
            learned_metadata = learned.outputs["rx_symbols"].metadata
            digital_metadata = digital.outputs["images"].metadata
            np.testing.assert_allclose(
                learned_metadata["source_item_channel_gain_real"],
                digital_metadata["source_item_channel_gain_real"],
            )
            np.testing.assert_allclose(
                learned_metadata["source_item_channel_gain_imag"],
                digital_metadata["source_item_channel_gain_imag"],
            )
            self.assertEqual(
                digital_metadata["channel_model"], "slow_rayleigh"
            )
            self.assertEqual(learned_metadata["receiver_processing"], "none")
            self.assertEqual(learned_metadata["fading_scope"], "source_item")


if __name__ == "__main__":
    unittest.main()
