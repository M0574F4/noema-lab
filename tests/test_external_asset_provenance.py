from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

from noema_lab.core.operations import OperationError
from noema_lab.ops import foundation
from noema_lab.ops import vision_yolo
from noema_lab.ops import vqa_goal
from noema_lab.ops.source import coco_yolo
from noema_lab.ops.source import retrieval_flickr8k
from noema_lab.ops.source import vqa_small


class ExternalDatasetProvenanceTests(unittest.TestCase):
    def test_hugging_face_dataset_sources_are_commit_and_digest_pinned(self):
        self.assertRegex(vqa_small.HF_REVISION, r"^[0-9a-f]{40}$")
        self.assertRegex(retrieval_flickr8k.HF_REVISION, r"^[0-9a-f]{40}$")
        self.assertNotIn("/main", vqa_small.HF_BASE_URL)
        self.assertNotIn("/main", retrieval_flickr8k.HF_BASE_URL)
        for digest in list(vqa_small.SPLIT_SHA256.values()) + list(
            retrieval_flickr8k.FILE_SHA256.values()
        ):
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_vqa_cache_hit_is_rejected_when_bytes_do_not_match(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / vqa_small.SPLIT_FILES["validation"]
            target.parent.mkdir(parents=True)
            target.write_bytes(b"altered parquet")
            with self.assertRaisesRegex(OperationError, "SHA-256 mismatch"):
                vqa_small._ensure_split_file(root, "validation", False)

    def test_flickr_cache_hit_is_rejected_when_bytes_do_not_match(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            relative = retrieval_flickr8k.SPLIT_FILES["test"][0]
            target = root / relative
            target.parent.mkdir(parents=True)
            target.write_bytes(b"altered parquet")
            with self.assertRaisesRegex(OperationError, "SHA-256 mismatch"):
                retrieval_flickr8k._ensure_split_files(root, "test", False)

    def test_coco_extraction_rejects_traversal_symlinks_and_zip_bombs(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            traversal = root / "traversal.zip"
            with zipfile.ZipFile(traversal, "w") as archive:
                archive.writestr("../escape.txt", b"escape")
            with self.assertRaisesRegex(OperationError, "Unsafe path"):
                coco_yolo._safe_extract_zip(traversal, root / "traversal")

            symlink = root / "symlink.zip"
            link = zipfile.ZipInfo("dataset/link")
            link.create_system = 3
            link.external_attr = (0o120777 << 16)
            with zipfile.ZipFile(symlink, "w") as archive:
                archive.writestr(link, "target")
            with self.assertRaisesRegex(OperationError, "Symbolic links"):
                coco_yolo._safe_extract_zip(symlink, root / "symlink")

            bomb = root / "bomb.zip"
            with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("dataset/zeros.bin", b"\0" * (1024 * 1024))
            with self.assertRaisesRegex(OperationError, "compression ratio"):
                coco_yolo._safe_extract_zip(bomb, root / "bomb")

    def test_coco_cache_is_verified_against_pinned_archive_contents(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive_path = root / "sample.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("sample/", b"")
                archive.writestr("sample/data.txt", b"trusted bytes")
            digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            extracted = coco_yolo._ensure_dataset(
                root,
                "sample",
                "https://example.invalid/sample.zip",
                digest,
                False,
            )
            self.assertEqual((extracted / "data.txt").read_bytes(), b"trusted bytes")
            (extracted / "data.txt").write_bytes(b"altered bytes")
            with self.assertRaisesRegex(OperationError, "content mismatch"):
                coco_yolo._ensure_dataset(
                    root,
                    "sample",
                    "https://example.invalid/sample.zip",
                    digest,
                    False,
                )


class ExternalModelProvenanceTests(unittest.TestCase):
    def test_yolo_rejects_unpinned_remote_and_implicit_downloads(self):
        with self.assertRaisesRegex(OperationError, "expected_model_sha256"):
            vision_yolo._resolve_yolo_model_ref(
                "https://example.invalid/model.pt"
            )
        with self.assertRaisesRegex(OperationError, "is not a local file"):
            vision_yolo._resolve_yolo_model_ref("unreviewed-model.pt")

    def test_yolo_verifies_local_model_when_digest_is_supplied(self):
        with tempfile.TemporaryDirectory() as raw:
            model = Path(raw) / "model.pt"
            model.write_bytes(b"trusted model")
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            self.assertEqual(
                vision_yolo._resolve_yolo_model_ref(str(model), digest),
                str(model),
            )
            with self.assertRaisesRegex(OperationError, "SHA-256 mismatch"):
                vision_yolo._resolve_yolo_model_ref(str(model), "0" * 64)

    def test_hugging_face_model_revisions_fail_closed_for_custom_remotes(self):
        with self.assertRaisesRegex(OperationError, "full immutable"):
            foundation._resolved_remote_model_revision(
                "example/custom-model",
                "main",
                default_model_id=foundation.DEFAULT_CLIP_MODEL_ID,
                default_revision=foundation.DEFAULT_CLIP_MODEL_REVISION,
                label="CLIP",
            )
        with self.assertRaisesRegex(OperationError, "full immutable"):
            vqa_goal._resolved_vqa_model_revision("example/custom-vqa", "")

    def test_builtin_hugging_face_models_resolve_to_pinned_commits(self):
        self.assertEqual(
            foundation._resolved_remote_model_revision(
                foundation.DEFAULT_CLIP_MODEL_ID,
                "",
                default_model_id=foundation.DEFAULT_CLIP_MODEL_ID,
                default_revision=foundation.DEFAULT_CLIP_MODEL_REVISION,
                label="CLIP",
            ),
            foundation.DEFAULT_CLIP_MODEL_REVISION,
        )
        self.assertEqual(
            foundation._resolved_remote_model_revision(
                foundation.DEFAULT_DIFFUSION_MODEL_ID,
                "",
                default_model_id=foundation.DEFAULT_DIFFUSION_MODEL_ID,
                default_revision=foundation.DEFAULT_DIFFUSION_MODEL_REVISION,
                label="Diffusion",
            ),
            foundation.DEFAULT_DIFFUSION_MODEL_REVISION,
        )
        self.assertEqual(
            foundation._resolved_remote_model_revision(
                foundation.DEFAULT_MASKED_LM_MODEL_ID,
                "",
                default_model_id=foundation.DEFAULT_MASKED_LM_MODEL_ID,
                default_revision=foundation.DEFAULT_MASKED_LM_MODEL_REVISION,
                label="Masked language model",
            ),
            foundation.DEFAULT_MASKED_LM_MODEL_REVISION,
        )
        self.assertEqual(
            vqa_goal._resolved_vqa_model_revision(
                vqa_goal.DEFAULT_VQA_MODEL_ID, ""
            ),
            vqa_goal.DEFAULT_VQA_MODEL_REVISION,
        )

    def test_local_hugging_face_model_paths_do_not_require_remote_revisions(self):
        with tempfile.TemporaryDirectory() as raw:
            model = Path(raw) / "local-model"
            model.mkdir()
            self.assertEqual(
                foundation._resolved_remote_model_revision(
                    str(model),
                    "",
                    default_model_id=foundation.DEFAULT_CLIP_MODEL_ID,
                    default_revision=foundation.DEFAULT_CLIP_MODEL_REVISION,
                    label="CLIP",
                ),
                "local_path",
            )
            self.assertEqual(
                vqa_goal._resolved_vqa_model_revision(str(model), ""),
                "local_path",
            )

    def test_masked_lm_loader_passes_pinned_revision_to_both_components(self):
        calls = []

        class _Loader:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                calls.append((cls.__name__, model_id, kwargs))
                return object()

        fake = types.ModuleType("transformers")
        fake.AutoModelForMaskedLM = type("AutoModelForMaskedLM", (_Loader,), {})
        fake.AutoTokenizer = type("AutoTokenizer", (_Loader,), {})
        fake.pipeline = lambda *_args, **_kwargs: "pipeline"
        with mock.patch.dict(sys.modules, {"transformers": fake}):
            result = foundation._load_fill_mask_pipeline(
                foundation.DEFAULT_MASKED_LM_MODEL_ID,
                foundation.DEFAULT_MASKED_LM_MODEL_REVISION,
                "cpu",
                "/tmp/noema-test-hf-cache",
            )
        self.assertEqual(result, "pipeline")
        self.assertEqual(len(calls), 2)
        self.assertTrue(
            all(
                call[2]["revision"] == foundation.DEFAULT_MASKED_LM_MODEL_REVISION
                for call in calls
            )
        )


if __name__ == "__main__":
    unittest.main()
