from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from noema_lab.core.operations import OperationError
from noema_lab.core.recipes import load_recipe
from noema_lab.ops.models.catalog import diffusers_load_kwargs, load_model_catalog
from noema_lab.ops.models.learned_codecs import (
    DiffusersAutoencoderKlDecodeOperation,
    DiffusersAutoencoderKlEncodeOperation,
    DiffusersVqModelDecodeOperation,
    DiffusersVqModelEncodeOperation,
)
from noema_lab.ops.models.text_codec import (
    DEFAULT_BART_MODEL_REVISION,
    TextBartJsccDecodeOperation,
    TextBartJsccEncodeOperation,
    _resolved_seq2seq_model_revision,
)


ROOT = Path(__file__).resolve().parents[1]
FULL_REVISION = "1" * 40


class RemoteModelProvenanceTests(unittest.TestCase):
    def test_default_diffusers_models_are_commit_pinned(self) -> None:
        catalog = load_model_catalog()
        autoencoder = diffusers_load_kwargs(
            catalog,
            "autoencoderkl",
            "stabilityai/sd-vae-ft-mse",
            {},
        )
        vqmodel = diffusers_load_kwargs(
            catalog,
            "vqmodel",
            "CompVis/ldm-celebahq-256",
            {},
        )
        self.assertRegex(autoencoder["revision"], r"^[0-9a-f]{40}$")
        self.assertRegex(vqmodel["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(vqmodel["subfolder"], "vqvae")

    def test_custom_remote_diffusers_model_requires_full_commit(self) -> None:
        catalog = load_model_catalog()
        with self.assertRaisesRegex(ValueError, "full immutable"):
            diffusers_load_kwargs(catalog, "autoencoderkl", "owner/model", {})
        with self.assertRaisesRegex(ValueError, "full immutable"):
            diffusers_load_kwargs(
                catalog,
                "autoencoderkl",
                "owner/model",
                {"revision": "main"},
            )
        self.assertEqual(
            diffusers_load_kwargs(
                catalog,
                "autoencoderkl",
                "owner/model",
                {"revision": FULL_REVISION},
            )["revision"],
            FULL_REVISION,
        )

    def test_local_diffusers_directory_does_not_claim_remote_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            kwargs = diffusers_load_kwargs(
                load_model_catalog(),
                "autoencoderkl",
                temporary,
                {"revision": FULL_REVISION},
            )
        self.assertNotIn("revision", kwargs)

    def test_diffusers_operation_schemas_expose_revision(self) -> None:
        for operation in (
            DiffusersAutoencoderKlEncodeOperation(),
            DiffusersAutoencoderKlDecodeOperation(),
            DiffusersVqModelEncodeOperation(),
            DiffusersVqModelDecodeOperation(),
        ):
            self.assertIn("revision", operation.params_schema["properties"])

    def test_bart_default_and_recipe_bind_the_same_full_commit(self) -> None:
        self.assertRegex(DEFAULT_BART_MODEL_REVISION, r"^[0-9a-f]{40}$")
        self.assertEqual(
            _resolved_seq2seq_model_revision("facebook/bart-base", ""),
            DEFAULT_BART_MODEL_REVISION,
        )
        for operation in (TextBartJsccEncodeOperation(), TextBartJsccDecodeOperation()):
            self.assertEqual(
                operation.params_schema["properties"]["model_revision"]["default"],
                DEFAULT_BART_MODEL_REVISION,
            )
        recipe = load_recipe(ROOT / "recipes" / "text_bart_jscc_clean.yaml")
        bart_steps = [
            step
            for step in recipe.steps
            if step.op in {
                "model.text_bart_jscc_encode",
                "model.text_bart_jscc_decode",
            }
        ]
        self.assertEqual(len(bart_steps), 2)
        self.assertTrue(
            all(
                step.params.get("model_revision") == DEFAULT_BART_MODEL_REVISION
                for step in bart_steps
            )
        )

    def test_custom_remote_bart_model_rejects_branch_names(self) -> None:
        with self.assertRaisesRegex(OperationError, "full immutable"):
            _resolved_seq2seq_model_revision("owner/model", "main")
        self.assertEqual(
            _resolved_seq2seq_model_revision("owner/model", FULL_REVISION),
            FULL_REVISION,
        )
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(
                _resolved_seq2seq_model_revision(temporary, ""),
                "local_path",
            )


if __name__ == "__main__":
    unittest.main()
