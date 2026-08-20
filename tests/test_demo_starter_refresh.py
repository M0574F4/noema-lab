from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from noema_lab.training.starter_refresh import prepare_demo_starter_directory


class DemoStarterRefreshTests(unittest.TestCase):
    def test_force_keeps_unmanaged_evidence_in_recognized_starter(self):
        with tempfile.TemporaryDirectory() as temporary:
            starter = Path(temporary) / "reference_training"
            starter.mkdir()
            (starter / "project_manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "kind": "noema.standalone_training_project",
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            history = starter / "training_history.json"
            metrics = starter / "evaluation_metrics.json"
            artifact = starter / "returned_artifacts" / "researcher_checkpoint.bin"
            artifact.parent.mkdir()
            history.write_bytes(b"history")
            metrics.write_bytes(b"metrics")
            artifact.write_bytes(b"checkpoint")

            prepared = prepare_demo_starter_directory(starter, force=True)

            self.assertEqual(prepared, starter)
            self.assertEqual(history.read_bytes(), b"history")
            self.assertEqual(metrics.read_bytes(), b"metrics")
            self.assertEqual(artifact.read_bytes(), b"checkpoint")

    def test_force_refuses_unrecognized_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            starter = Path(temporary) / "researcher_project"
            starter.mkdir()
            researcher_file = starter / "train.py"
            researcher_file.write_text("# mine\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "not a recognized Noema demonstration starter",
            ):
                prepare_demo_starter_directory(starter, force=True)

            self.assertEqual(
                researcher_file.read_text(encoding="utf-8"),
                "# mine\n",
            )

    def test_legacy_generated_marker_set_can_be_refreshed_in_place(self):
        with tempfile.TemporaryDirectory() as temporary:
            starter = Path(temporary) / "legacy_reference_training"
            starter.mkdir()
            for filename in (
                "training_template.yaml",
                "train_config.yaml",
                "noema_recipe.yaml",
            ):
                (starter / filename).write_text("generated: true\n", encoding="utf-8")
            evidence = starter / "evaluation_metrics.json"
            evidence.write_bytes(b"legacy evidence")

            prepare_demo_starter_directory(starter, force=True)

            self.assertEqual(evidence.read_bytes(), b"legacy evidence")


if __name__ == "__main__":
    unittest.main()
