from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from noema_lab.core.artifacts import Artifact
from noema_lab.core.benchmark_plots import (
    _aggregate_plot_rows,
    _mean_std_ci95,
    _plot_rows,
)
from noema_lab.core.benchmarks import BenchmarkError
from noema_lab.core.demo_export import (
    DemoExportError,
    _normalize_plot_specs,
    _plot_rows as _demo_plot_rows,
    _render_plot_svg,
    _series_summaries,
    _validate_plot_metrics,
)
from noema_lab.core.operations import OperationContext
from noema_lab.ops.metrics.image import (
    ImageReconstructionMetricsOperation,
    _try_ms_ssim,
)
from noema_lab.ops.metrics.task import (
    CaptioningMetricsOperation,
    ClassificationMetricsOperation,
    ClipRetrievalRankOperation,
    DetectionMetricsOperation,
    EmbeddingSimilarityMetricsOperation,
    RetrievalMetricsOperation,
    SegmentationMetricsOperation,
    VqaMetricsOperation,
)
from noema_lab.ops.metrics.text import TextSemanticSimilarityMetricsOperation
from noema_lab.ops.foundation import (
    ClipImageEmbeddingOperation,
    ClipTextEmbeddingOperation,
    SemanticStateFaithfulnessMetricsOperation,
)


class PublicationMetricContractTests(unittest.TestCase):
    def _context(
        self,
        root: Path,
        inputs: dict[str, Artifact],
        params: dict | None = None,
    ) -> OperationContext:
        return OperationContext(
            recipe_name="publication_metric_contract",
            step_id="evaluation",
            params=dict(params or {}),
            inputs=inputs,
            run_dir=root,
            step_dir=root / "evaluation",
        )

    def _image_artifact(
        self,
        root: Path,
        name: str,
        images: np.ndarray,
        metadata: dict | None = None,
    ) -> Artifact:
        path = root / (name + ".npz")
        payload_metadata = dict(metadata or {})
        np.savez_compressed(
            path,
            images=np.asarray(images),
            metadata_json=json.dumps(payload_metadata),
        )
        return Artifact("image.batch.numpy", path, payload_metadata)

    def _json_artifact(
        self,
        root: Path,
        name: str,
        kind: str,
        examples: list[dict],
        payload_metadata: dict | None = None,
    ) -> Artifact:
        path = root / (name + ".json")
        payload = {"schema_version": 1, "kind": kind, "examples": examples}
        payload.update(payload_metadata or {})
        path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return Artifact(kind + ".json", path, {})

    def _embedding_artifact(
        self,
        root: Path,
        name: str,
        values: np.ndarray,
        ids: list[str] | None = None,
        *,
        modality: str = "generic",
        space_overrides: dict | None = None,
    ) -> Artifact:
        path = root / (name + ".npz")
        dimensions = int(np.asarray(values).shape[1])
        space = {
            "schema_version": 1,
            "space_family": "test_space",
            "backend": "test_backend",
            "model_id": "test/model",
            "model_revision": "test-revision",
            "preprocessing_contract": "test_preprocessing_v1",
            "dimensions": dimensions,
        }
        space.update(space_overrides or {})
        metadata = {
            "embedding_modality": modality,
            "embedding_space": space,
        }
        if ids is not None:
            metadata["sample_ids"] = ids
        np.savez_compressed(
            path,
            embeddings=np.asarray(values, dtype=np.float32),
            metadata_json=json.dumps(metadata),
        )
        return Artifact("foundation.embedding.numpy", path, metadata)

    def _mask_artifact(
        self,
        root: Path,
        name: str,
        values: np.ndarray,
        ids: list[str] | None = None,
    ) -> Artifact:
        path = root / (name + ".npz")
        metadata = {"sample_ids": ids} if ids is not None else {}
        np.savez_compressed(
            path,
            masks=np.asarray(values),
            metadata_json=json.dumps(metadata),
        )
        return Artifact("vision.segmentation_mask.numpy", path, metadata)

    def _semantic_state_artifact(
        self, root: Path, name: str, states: list[dict]
    ) -> Artifact:
        path = root / (name + ".json")
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "semantic.state",
                    "id": name,
                    "modality": "text",
                    "states": states,
                }
            ),
            encoding="utf-8",
        )
        return Artifact("semantic.state.json", path, {})

    def _knowledge_base_artifact(self, root: Path, facts: list[dict]) -> Artifact:
        path = root / "kb.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "knowledge_base",
                    "id": "test_kb",
                    "facts": facts,
                }
            ),
            encoding="utf-8",
        )
        return Artifact("foundation.kb.json", path, {})

    def test_image_metrics_reject_count_id_shape_and_range_mismatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = np.zeros((2, 4, 4, 3), dtype=np.uint8)
            one_image = np.zeros((1, 4, 4, 3), dtype=np.uint8)
            with self.assertRaisesRegex(RuntimeError, "count mismatch"):
                ImageReconstructionMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "reference": self._image_artifact(root, "ref_count", reference),
                            "reconstruction": self._image_artifact(root, "rec_count", one_image),
                        },
                    )
                )

            with self.assertRaisesRegex(RuntimeError, "ID/order mismatch"):
                ImageReconstructionMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "reference": self._image_artifact(
                                root, "ref_ids", reference, {"sample_ids": ["a", "b"]}
                            ),
                            "reconstruction": self._image_artifact(
                                root, "rec_ids", reference, {"sample_ids": ["b", "a"]}
                            ),
                        },
                    )
                )

            with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
                ImageReconstructionMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "reference": self._image_artifact(
                                root,
                                "ref_shapes",
                                reference,
                                {"original_shapes": [[1, 4, 4, 3], [1, 3, 4, 3]]},
                            ),
                            "reconstruction": self._image_artifact(
                                root,
                                "rec_shapes",
                                reference,
                                {"original_shapes": [[1, 4, 4, 3], [1, 4, 4, 3]]},
                            ),
                        },
                    )
                )

            normalized_reference = np.zeros(reference.shape, dtype=np.float32)
            out_of_range = normalized_reference.copy()
            out_of_range[0, 0, 0, 0] = 1.1
            with self.assertRaisesRegex(RuntimeError, "outside declared"):
                ImageReconstructionMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "reference": self._image_artifact(
                                root, "ref_range", normalized_reference
                            ),
                            "reconstruction": self._image_artifact(
                                root, "rec_range", out_of_range
                            ),
                        },
                    )
                )

    def test_image_metrics_retain_per_example_values_and_macro_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = np.zeros((2, 2, 2, 3), dtype=np.uint8)
            reconstruction = reference.copy()
            reconstruction[1] = 1
            result = ImageReconstructionMetricsOperation().run(
                self._context(
                    root,
                    {
                        "reference": self._image_artifact(root, "ref", reference),
                        "reconstruction": self._image_artifact(root, "rec", reconstruction),
                    },
                )
            )
            report = json.loads(result.outputs["report"].path.read_text(encoding="utf-8"))
            self.assertEqual(len(report["per_example"]), 2)
            self.assertEqual(report["per_example"][0]["mse"], 0.0)
            self.assertEqual(report["per_example"][1]["mse"], 1.0)
            self.assertEqual(result.metrics["quality.mse"], 0.5)
            self.assertIn("per-image", report["aggregation"]["psnr_db"])

    def test_exact_psnr_sentinel_is_above_attainable_finite_float32_psnr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = np.zeros((2, 2, 2, 3), dtype=np.float32)
            reconstruction = reference.copy()
            reconstruction[1, 0, 0, 0] = np.nextafter(
                np.float32(0.0), np.float32(1.0)
            )
            result = ImageReconstructionMetricsOperation().run(
                self._context(
                    root,
                    {
                        "reference": self._image_artifact(root, "float_ref", reference),
                        "reconstruction": self._image_artifact(
                            root, "float_rec", reconstruction
                        ),
                    },
                    {"psnr_cap_db": 0.0},
                )
            )
            report = json.loads(result.outputs["report"].path.read_text(encoding="utf-8"))
            exact_psnr = report["per_example"][0]["psnr_db"]
            finite_psnr = report["per_example"][1]["psnr_db"]
            self.assertGreater(exact_psnr, finite_psnr)
            self.assertEqual(exact_psnr, report["psnr_cap_db"])
            self.assertEqual(result.metrics["quality.perfect_reconstruction_fraction"], 0.5)

    def test_ms_ssim_reports_total_eligible_and_evaluated_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = np.zeros((2, 200, 200, 3), dtype=np.uint8)
            metadata = {
                "sample_ids": ["large", "small"],
                "original_shapes": [[1, 200, 200, 3], [1, 100, 100, 3]],
            }
            with mock.patch(
                "noema_lab.ops.metrics.image._try_ms_ssim",
                return_value=[1.0, None],
            ):
                result = ImageReconstructionMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "reference": self._image_artifact(
                                root, "coverage_ref", images, metadata
                            ),
                            "reconstruction": self._image_artifact(
                                root, "coverage_rec", images, metadata
                            ),
                        },
                    )
                )
            coverage = result.metadata["ms_ssim_coverage"]
            self.assertEqual(coverage["total_image_count"], 2)
            self.assertEqual(coverage["eligible_image_count"], 1)
            self.assertEqual(coverage["evaluated_image_count"], 1)
            self.assertEqual(coverage["evaluated_fraction_of_total"], 0.5)
            self.assertEqual(coverage["evaluated_fraction_of_eligible"], 1.0)
            self.assertEqual(result.metrics["quality.ms_ssim"], 1.0)
            self.assertEqual(
                result.metrics["quality.ms_ssim.evaluated_image_count"], 1
            )

    def test_ms_ssim_only_treats_exact_optional_module_absence_as_unavailable(self):
        images = np.zeros((1, 8, 8, 3), dtype=np.uint8)
        missing_torch = ModuleNotFoundError(
            "No module named 'torch'",
            name="torch",
        )
        with mock.patch(
            "noema_lab.ops.metrics.image.importlib.import_module",
            side_effect=missing_torch,
        ):
            self.assertIsNone(
                _try_ms_ssim(images, images, [[8, 8, 3]], 255.0)
            )

        fake_torch = object()

        def missing_msssim(name):
            if name == "torch":
                return fake_torch
            raise ModuleNotFoundError(
                "No module named 'pytorch_msssim'",
                name="pytorch_msssim",
            )

        with mock.patch(
            "noema_lab.ops.metrics.image.importlib.import_module",
            side_effect=missing_msssim,
        ):
            self.assertIsNone(
                _try_ms_ssim(images, images, [[8, 8, 3]], 255.0)
            )

    def test_ms_ssim_rejects_broken_installed_optional_dependencies(self):
        images = np.zeros((1, 8, 8, 3), dtype=np.uint8)
        transitive_failure = ModuleNotFoundError(
            "No module named 'typing_extensions'",
            name="typing_extensions",
        )
        with mock.patch(
            "noema_lab.ops.metrics.image.importlib.import_module",
            side_effect=transitive_failure,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "torch is installed.*typing_extensions is missing",
            ):
                _try_ms_ssim(images, images, [[8, 8, 3]], 255.0)

        def broken_msssim(name):
            if name == "torch":
                return object()
            raise RuntimeError("binary ABI mismatch")

        with mock.patch(
            "noema_lab.ops.metrics.image.importlib.import_module",
            side_effect=broken_msssim,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "pytorch_msssim is installed.*binary ABI mismatch",
            ):
                _try_ms_ssim(images, images, [[8, 8, 3]], 255.0)

    def test_text_and_task_metrics_reject_missing_or_reordered_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "ID/order mismatch"):
                TextSemanticSimilarityMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "reference": self._json_artifact(
                                root,
                                "text_ref",
                                "text.batch",
                                [{"id": "a", "text": "one"}, {"id": "b", "text": "two"}],
                            ),
                            "candidate": self._json_artifact(
                                root,
                                "text_cand",
                                "text.batch",
                                [{"id": "b", "text": "two"}, {"id": "a", "text": "one"}],
                            ),
                        },
                    )
                )
            with self.assertRaisesRegex(RuntimeError, "requires a non-empty id"):
                ClassificationMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "reference": self._json_artifact(
                                root, "label_ref", "task.labels", [{"id": "a", "label": "x"}]
                            ),
                            "candidate": self._json_artifact(
                                root, "label_cand", "task.predictions", [{"prediction": "x"}]
                            ),
                        },
                    )
                )

            with self.assertRaisesRegex(RuntimeError, "requires a text value"):
                TextSemanticSimilarityMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "reference": self._json_artifact(
                                root, "missing_text_ref", "text.batch", [{"id": "a"}]
                            ),
                            "candidate": self._json_artifact(
                                root, "missing_text_cand", "text.batch", [{"id": "a"}]
                            ),
                        },
                    )
                )

            with self.assertRaisesRegex(RuntimeError, "requires one of label"):
                ClassificationMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "reference": self._json_artifact(
                                root, "missing_label_ref", "task.labels", [{"id": "a"}]
                            ),
                            "candidate": self._json_artifact(
                                root,
                                "missing_label_cand",
                                "task.predictions",
                                [{"id": "a"}],
                            ),
                        },
                    )
                )

            with self.assertRaisesRegex(RuntimeError, "requires one of caption, text"):
                CaptioningMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "reference": self._json_artifact(
                                root, "missing_caption_ref", "text.caption", [{"id": "a"}]
                            ),
                            "candidate": self._json_artifact(
                                root,
                                "missing_caption_cand",
                                "text.caption",
                                [{"id": "a"}],
                            ),
                        },
                    )
                )

    def test_exact_match_metrics_use_literal_string_equality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_result = TextSemanticSimilarityMetricsOperation().run(
                self._context(
                    root,
                    {
                        "reference": self._json_artifact(
                            root, "literal_ref", "text.batch", [{"id": "a", "text": "Hello!"}]
                        ),
                        "candidate": self._json_artifact(
                            root, "literal_cand", "text.batch", [{"id": "a", "text": "hello?"}]
                        ),
                    },
                )
            )
            self.assertEqual(text_result.metrics["text.exact_match"], 0.0)
            self.assertEqual(text_result.metrics["semantic.lexical_similarity"], 1.0)

            caption_result = CaptioningMetricsOperation().run(
                self._context(
                    root,
                    {
                        "reference": self._json_artifact(
                            root, "caption_ref", "text.caption", [{"id": "a", "caption": "Yes"}]
                        ),
                        "candidate": self._json_artifact(
                            root, "caption_cand", "text.caption", [{"id": "a", "caption": "yes"}]
                        ),
                    },
                )
            )
            self.assertEqual(caption_result.metrics["caption.exact_match"], 0.0)

            vqa_result = VqaMetricsOperation().run(
                self._context(
                    root,
                    {
                        "reference": self._json_artifact(
                            root, "literal_vqa_ref", "vqa.answers", [{"id": "q", "answer": "yes"}]
                        ),
                        "candidate": self._json_artifact(
                            root, "literal_vqa_cand", "vqa.answers", [{"id": "q", "answer": "YES"}]
                        ),
                    },
                )
            )
            self.assertEqual(vqa_result.metrics["vqa.single_reference_exact_match"], 0.0)

    def test_embedding_and_segmentation_metrics_reject_truncation_or_broadcasting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "Embedding count mismatch"):
                EmbeddingSimilarityMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "reference": self._embedding_artifact(
                                root, "emb_ref", np.ones((2, 3), dtype=np.float32)
                            ),
                            "candidate": self._embedding_artifact(
                                root, "emb_cand", np.ones((1, 3), dtype=np.float32)
                            ),
                        },
                    )
                )
            with self.assertRaisesRegex(RuntimeError, "Embedding sample ID/order mismatch"):
                EmbeddingSimilarityMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "reference": self._embedding_artifact(
                                root,
                                "emb_ref_ids",
                                np.ones((2, 3), dtype=np.float32),
                                ["a", "b"],
                            ),
                            "candidate": self._embedding_artifact(
                                root,
                                "emb_cand_ids",
                                np.ones((2, 3), dtype=np.float32),
                                ["b", "a"],
                            ),
                        },
                    )
                )
            ref_path = root / "mask_ref.npz"
            cand_path = root / "mask_cand.npz"
            np.savez_compressed(ref_path, masks=np.zeros((2, 3, 3), dtype=np.int32))
            np.savez_compressed(cand_path, masks=np.zeros((1, 3, 3), dtype=np.int32))
            with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
                SegmentationMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "reference": Artifact(
                                "vision.segmentation_mask.numpy", ref_path, {}
                            ),
                            "candidate": Artifact(
                                "vision.segmentation_mask.numpy", cand_path, {}
                            ),
                        },
                    )
                )

    def test_embedding_metric_names_match_cosine_or_dot_product_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = self._embedding_artifact(
                root, "dot_ref", np.asarray([[1.0, 2.0]], dtype=np.float32)
            )
            candidate = self._embedding_artifact(
                root, "dot_cand", np.asarray([[2.0, 2.0]], dtype=np.float32)
            )
            dot_result = EmbeddingSimilarityMetricsOperation().run(
                self._context(
                    root,
                    {"reference": reference, "candidate": candidate},
                    {"normalize": False, "label": "embedding"},
                )
            )
            self.assertEqual(
                dot_result.metrics["embedding.generic_generic.dot_product_mean"], 6.0
            )
            self.assertNotIn("embedding.generic_generic.cosine_mean", dot_result.metrics)

            cosine_result = EmbeddingSimilarityMetricsOperation().run(
                self._context(
                    root,
                    {"reference": reference, "candidate": candidate},
                    {"normalize": True, "label": "embedding"},
                )
            )
            self.assertAlmostEqual(
                cosine_result.metrics["embedding.generic_generic.cosine_mean"],
                6.0 / (np.sqrt(5.0) * np.sqrt(8.0)),
            )

            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                EmbeddingSimilarityMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "reference": self._embedding_artifact(
                                root, "finite_ref", np.asarray([[1.0, 0.0]])
                            ),
                            "candidate": self._embedding_artifact(
                                root, "nonfinite_cand", np.asarray([[np.inf, 0.0]])
                            ),
                        },
                    )
                )

            with self.assertRaisesRegex(RuntimeError, "zero-norm rows"):
                EmbeddingSimilarityMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "reference": self._embedding_artifact(
                                root, "zero_ref", np.zeros((1, 2), dtype=np.float32)
                            ),
                            "candidate": self._embedding_artifact(
                                root, "zero_cand", np.ones((1, 2), dtype=np.float32)
                            ),
                        },
                    )
                )

    def test_segmentation_requires_integer_masks_and_matching_sample_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masks = np.zeros((2, 3, 3), dtype=np.int32)
            with self.assertRaisesRegex(RuntimeError, "ID/order mismatch"):
                SegmentationMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "reference": self._mask_artifact(
                                root, "mask_ids_ref", masks, ["a", "b"]
                            ),
                            "candidate": self._mask_artifact(
                                root, "mask_ids_cand", masks, ["b", "a"]
                            ),
                        },
                    )
                )

            with self.assertRaisesRegex(RuntimeError, "integer dtype"):
                SegmentationMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "reference": self._mask_artifact(
                                root, "float_mask_ref", masks.astype(np.float32), ["a", "b"]
                            ),
                            "candidate": self._mask_artifact(
                                root, "float_mask_cand", masks.astype(np.float32), ["a", "b"]
                            ),
                        },
                    )
                )

    def test_retrieval_rejects_underspecified_rankings_and_names_mrr_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "at least one ranking"):
                RetrievalMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "rankings": self._json_artifact(
                                root, "empty_rankings", "retrieval.rankings", []
                            )
                        },
                    )
                )

            with self.assertRaisesRegex(RuntimeError, "requires one of target_id"):
                RetrievalMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "rankings": self._json_artifact(
                                root,
                                "missing_target",
                                "retrieval.rankings",
                                [{"id": "q", "ranked_ids": ["a"]}],
                            )
                        },
                        {"k_values": "1"},
                    )
                )

            with self.assertRaisesRegex(RuntimeError, "non-empty ranked_ids"):
                RetrievalMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "rankings": self._json_artifact(
                                root,
                                "empty_ranked_ids",
                                "retrieval.rankings",
                                [{"id": "q", "target_id": "a", "ranked_ids": []}],
                            )
                        },
                    )
                )

            result = RetrievalMetricsOperation().run(
                self._context(
                    root,
                    {
                        "rankings": self._json_artifact(
                            root,
                            "ranked_at_two",
                            "retrieval.rankings",
                            [
                                {"id": "q1", "target_id": "a", "ranked_ids": ["a", "b"]},
                                {"id": "q2", "target_id": "b", "ranked_ids": ["a", "b"]},
                            ],
                        )
                    },
                    {"k_values": "1,2"},
                )
            )
            self.assertEqual(result.metrics["retrieval.mrr_at_2"], 0.75)
            self.assertEqual(result.metrics["retrieval.hit_at_1"], 0.5)
            self.assertEqual(result.metrics["retrieval.hit_at_2"], 1.0)
            self.assertEqual(
                result.metrics["retrieval.recall_at_2"],
                result.metrics["retrieval.hit_at_2"],
            )
            self.assertNotIn("retrieval.mrr", result.metrics)

    def test_retrieval_binds_cutoffs_depth_and_candidate_universe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            examples = [
                {"id": "q1", "target_id": "a", "ranked_ids": ["a", "b"]},
                {"id": "q2", "target_id": "b", "ranked_ids": ["b", "a"]},
            ]
            artifact = self._json_artifact(
                root,
                "bound_rankings",
                "retrieval.rankings",
                examples,
                {
                    "candidate_ids": ["a", "b", "c"],
                    "candidate_count": 3,
                    "ranking_depth": 2,
                },
            )
            for invalid in ("1,1", "0,1", "1,-2", "1,", "1.5"):
                with self.subTest(k_values=invalid):
                    with self.assertRaisesRegex(RuntimeError, "k_values"):
                        RetrievalMetricsOperation().run(
                            self._context(
                                root,
                                {"rankings": artifact},
                                {"k_values": invalid},
                            )
                        )
            with self.assertRaisesRegex(RuntimeError, "exceed.*ranking_depth"):
                RetrievalMetricsOperation().run(
                    self._context(
                        root,
                        {"rankings": artifact},
                        {"k_values": "1,3"},
                    )
                )

            outside = self._json_artifact(
                root,
                "outside_pool",
                "retrieval.rankings",
                [{"id": "q", "target_id": "a", "ranked_ids": ["a", "z"]}],
                {
                    "candidate_ids": ["a", "b"],
                    "candidate_count": 2,
                    "ranking_depth": 2,
                },
            )
            with self.assertRaisesRegex(RuntimeError, "outside the declared candidate universe"):
                RetrievalMetricsOperation().run(
                    self._context(
                        root, {"rankings": outside}, {"k_values": "1,2"}
                    )
                )

            result = RetrievalMetricsOperation().run(
                self._context(
                    root, {"rankings": artifact}, {"k_values": "1,2"}
                )
            )
            self.assertTrue(
                all(
                    0.0 <= float(value) <= 1.0
                    for metric_id, value in result.metrics.items()
                    if metric_id.startswith("retrieval.")
                    and "candidate_count" not in metric_id
                )
            )
            self.assertEqual(result.metadata["candidate_count"], 3)
            self.assertEqual(
                result.metadata["candidate_universe_binding"],
                "declared_candidate_ids",
            )

    def test_embedding_metrics_require_matching_provenance_and_safe_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = self._embedding_artifact(
                root,
                "clip_text",
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                ["a"],
                modality="text",
                space_overrides={"space_family": "clip"},
            )
            candidate = self._embedding_artifact(
                root,
                "clip_image",
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                ["a"],
                modality="image",
                space_overrides={"space_family": "clip"},
            )
            result = EmbeddingSimilarityMetricsOperation().run(
                self._context(
                    root,
                    {"reference": reference, "candidate": candidate},
                    {"label": "generation.text_image_clip", "normalize": True},
                )
            )
            self.assertEqual(
                result.metrics["generation.text_image_clip.cosine_mean"], 1.0
            )

            mismatch = self._embedding_artifact(
                root,
                "other_image",
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                ["a"],
                modality="image",
                space_overrides={
                    "space_family": "clip",
                    "model_revision": "different-revision",
                },
            )
            with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                EmbeddingSimilarityMetricsOperation().run(
                    self._context(
                        root,
                        {"reference": reference, "candidate": mismatch},
                        {"label": "generation.text_image_clip"},
                    )
                )
            with self.assertRaisesRegex(RuntimeError, "namespace"):
                EmbeddingSimilarityMetricsOperation().run(
                    self._context(
                        root,
                        {"reference": reference, "candidate": candidate},
                        {"label": "pair"},
                    )
                )

    def test_retrieval_ranker_binds_pool_and_embedding_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_embeddings = self._embedding_artifact(
                root,
                "rank_images",
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                ["a", "b"],
                modality="image",
            )
            text_embeddings = self._embedding_artifact(
                root,
                "rank_text",
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                ["q1", "q2"],
                modality="text",
            )
            targets = self._json_artifact(
                root,
                "rank_targets",
                "retrieval.targets",
                [
                    {"id": "q1", "target_id": "a"},
                    {"id": "q2", "target_id": "b"},
                ],
            )
            ranked = ClipRetrievalRankOperation().run(
                self._context(
                    root,
                    {
                        "image_embeddings": image_embeddings,
                        "text_embeddings": text_embeddings,
                        "targets": targets,
                    },
                    {"normalize": True, "top_k": 2},
                )
            )
            payload = json.loads(
                ranked.outputs["rankings"].path.read_text(encoding="utf-8")
            )
            self.assertEqual(payload["candidate_ids"], ["a", "b"])
            self.assertEqual(payload["candidate_count"], 2)
            self.assertEqual(payload["ranking_depth"], 2)
            result = RetrievalMetricsOperation().run(
                self._context(
                    root,
                    {"rankings": ranked.outputs["rankings"]},
                    {"k_values": "1,2"},
                )
            )
            self.assertEqual(result.metrics["retrieval.hit_at_1"], 1.0)

            mismatched_text = self._embedding_artifact(
                root,
                "rank_text_mismatch",
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                ["q1", "q2"],
                modality="text",
                space_overrides={"model_id": "other/model"},
            )
            with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                ClipRetrievalRankOperation().run(
                    self._context(
                        root,
                        {
                            "image_embeddings": image_embeddings,
                            "text_embeddings": mismatched_text,
                            "targets": targets,
                        },
                        {"normalize": True, "top_k": 2},
                    )
                )
            with self.assertRaisesRegex(RuntimeError, "top_k must be a positive integer"):
                ClipRetrievalRankOperation().run(
                    self._context(
                        root,
                        {
                            "image_embeddings": image_embeddings,
                            "text_embeddings": text_embeddings,
                            "targets": targets,
                        },
                        {"normalize": True, "top_k": True},
                    )
                )

    def test_local_paired_embedders_emit_matching_space_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((1, 32, 32, 3), dtype=np.uint8)
            image[:, 8:24, 8:24, 0] = 220
            images = self._image_artifact(
                root, "local_embed_image", image, {"sample_ids": ["a"]}
            )
            texts = self._json_artifact(
                root,
                "local_embed_text",
                "text.batch",
                [{"id": "q", "text": "a red square"}],
            )
            image_result = ClipImageEmbeddingOperation().run(
                self._context(
                    root, {"images": images}, {"backend": "local_semantic"}
                )
            )
            text_result = ClipTextEmbeddingOperation().run(
                self._context(
                    root, {"texts": texts}, {"backend": "local_semantic"}
                )
            )
            image_meta = image_result.outputs["embeddings"].metadata
            text_meta = text_result.outputs["embeddings"].metadata
            self.assertEqual(image_meta["embedding_space"], text_meta["embedding_space"])
            self.assertEqual(image_meta["embedding_modality"], "image")
            self.assertEqual(text_meta["embedding_modality"], "text")
            self.assertEqual(
                image_meta["embedding_space"]["model_revision"],
                "noema_local_semantic_v1",
            )

    def test_faithfulness_separates_unsupported_assertions_and_omissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fact_a = {"subject": "s", "predicate": "has", "object": "a"}
            fact_b = {"subject": "s", "predicate": "has", "object": "b"}
            reference = self._semantic_state_artifact(
                root, "reference_state", [{"id": "x", "facts": [fact_a]}]
            )
            silent = self._semantic_state_artifact(
                root, "silent_state", [{"id": "x", "facts": []}]
            )
            asserted = self._semantic_state_artifact(
                root, "asserted_state", [{"id": "x", "facts": [fact_b]}]
            )
            kb = self._knowledge_base_artifact(root, [fact_a])

            silent_result = SemanticStateFaithfulnessMetricsOperation().run(
                self._context(
                    root,
                    {"reference": reference, "candidate": silent, "kb": kb},
                )
            )
            self.assertEqual(
                silent_result.metrics["faithfulness.unsupported_assertion_rate"],
                0.0,
            )
            self.assertEqual(
                silent_result.metrics["faithfulness.fact_omission_rate"], 1.0
            )
            self.assertNotIn("faithfulness.hallucination_rate", silent_result.metrics)

            asserted_result = SemanticStateFaithfulnessMetricsOperation().run(
                self._context(
                    root,
                    {"reference": reference, "candidate": asserted, "kb": kb},
                )
            )
            self.assertEqual(
                asserted_result.metrics["faithfulness.unsupported_assertion_rate"],
                1.0,
            )
            self.assertEqual(
                asserted_result.metrics["faithfulness.fact_omission_rate"], 1.0
            )

            duplicate = self._semantic_state_artifact(
                root,
                "duplicate_state",
                [{"id": "x", "facts": []}, {"id": "x", "facts": []}],
            )
            with self.assertRaisesRegex(ValueError, "must be unique"):
                SemanticStateFaithfulnessMetricsOperation().run(
                    self._context(
                        root,
                        {"reference": duplicate, "candidate": silent, "kb": kb},
                    )
                )

    def test_proxy_metrics_do_not_claim_standard_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_result = TextSemanticSimilarityMetricsOperation().run(
                self._context(
                    root,
                    {
                        "reference": self._json_artifact(
                            root, "proxy_ref", "text.batch", [{"id": "a", "text": "hello world"}]
                        ),
                        "candidate": self._json_artifact(
                            root, "proxy_cand", "text.batch", [{"id": "a", "text": "hello"}]
                        ),
                    },
                )
            )
            self.assertIn("text.unigram_bleu_proxy", text_result.metrics)
            self.assertNotIn("text.bleu", text_result.metrics)

            vqa_result = VqaMetricsOperation().run(
                self._context(
                    root,
                    {
                        "reference": self._json_artifact(
                            root, "vqa_ref", "vqa.answers", [{"id": "q", "answer": "yes"}]
                        ),
                        "candidate": self._json_artifact(
                            root, "vqa_cand", "vqa.answers", [{"id": "q", "answer": "YES"}]
                        ),
                    },
                )
            )
            self.assertEqual(vqa_result.metrics["vqa.single_reference_exact_match"], 0.0)
            self.assertNotIn("vqa.accuracy", vqa_result.metrics)

            detection_result = DetectionMetricsOperation().run(
                self._context(
                    root,
                    {
                        "reference": self._json_artifact(
                            root,
                            "det_ref",
                            "vision.detections",
                            [{"id": "i", "detections": [{"label": "x", "bbox": [0, 0, 2, 2]}]}],
                        ),
                        "candidate": self._json_artifact(
                            root,
                            "det_cand",
                            "vision.detections",
                            [{"id": "i", "detections": [{"label": "x", "bbox": [0, 0, 2, 2]}]}],
                        ),
                    },
                )
            )
            self.assertEqual(
                detection_result.metrics["detection.f1_at_iou_0p5"], 1.0
            )
            self.assertNotIn("detection.f1_at_iou", detection_result.metrics)
            self.assertNotIn("detection.map50", detection_result.metrics)

    def test_detection_threshold_zero_is_honored_and_encoded_in_metric_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = DetectionMetricsOperation().run(
                self._context(
                    root,
                    {
                        "reference": self._json_artifact(
                            root,
                            "det_zero_ref",
                            "vision.detections",
                            [{"id": "i", "detections": [{"label": "x", "bbox": [0, 0, 1, 1]}]}],
                        ),
                        "candidate": self._json_artifact(
                            root,
                            "det_zero_cand",
                            "vision.detections",
                            [{"id": "i", "detections": [{"label": "x", "bbox": [2, 2, 3, 3]}]}],
                        ),
                    },
                    {"iou_threshold": 0.0},
                )
            )
            self.assertEqual(result.metrics["detection.f1_at_iou_0"], 1.0)
            self.assertNotIn("detection.f1_at_iou_0p5", result.metrics)

    def test_detection_matching_is_order_independent_and_confidence_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_a = {"label": "x", "bbox": [0, 0, 2, 2]}
            ref_b = {"label": "x", "bbox": [1, 0, 3, 2]}
            shared = {"label": "x", "bbox": [0.5, 0, 2.5, 2], "score": 0.9}
            only_a = {"label": "x", "bbox": [0, 0, 1, 2], "score": 0.8}

            def evaluate(refs, candidates):
                return DetectionMetricsOperation().run(
                    self._context(
                        root,
                        {
                            "reference": self._json_artifact(
                                root,
                                "order_ref_%d" % len(refs),
                                "vision.detections",
                                [{"id": "i", "detections": refs}],
                            ),
                            "candidate": self._json_artifact(
                                root,
                                "order_cand_%d_%s" % (len(candidates), candidates[0]["bbox"][0]),
                                "vision.detections",
                                [{"id": "i", "detections": candidates}],
                            ),
                        },
                        {"iou_threshold": 0.4, "confidence_threshold": 0.0},
                    )
                )

            forward = evaluate([ref_a, ref_b], [shared, only_a])
            reversed_result = evaluate([ref_b, ref_a], [only_a, shared])
            canonical_id = "detection.f1_at_iou_0p4_conf_0"
            self.assertEqual(forward.metrics[canonical_id], 1.0)
            self.assertEqual(reversed_result.metrics[canonical_id], 1.0)
            self.assertIn(
                "maximum-cardinality", str(forward.metadata["matching"])
            )

            filtered = DetectionMetricsOperation().run(
                self._context(
                    root,
                    {
                        "reference": self._json_artifact(
                            root,
                            "confidence_ref",
                            "vision.detections",
                            [{"id": "i", "detections": [ref_a]}],
                        ),
                        "candidate": self._json_artifact(
                            root,
                            "confidence_cand",
                            "vision.detections",
                            [{
                                "id": "i",
                                "detections": [
                                    {"label": "x", "bbox": [0, 0, 2, 2], "score": 0.4}
                                ],
                            }],
                        ),
                    },
                    {"iou_threshold": 0.5, "confidence_threshold": 0.5},
                )
            )
            self.assertEqual(
                filtered.metrics["detection.f1_at_iou_0p5_conf_0p5"], 0.0
            )
            self.assertNotIn("detection.f1_at_iou_0p5", filtered.metrics)

    def test_segmentation_reports_pooled_semantic_raster_policy_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = np.asarray([[[0, 1], [1, 1]]], dtype=np.int32)
            candidate = np.asarray([[[0, 0], [1, 1]]], dtype=np.int32)
            result = SegmentationMetricsOperation().run(
                self._context(
                    root,
                    {
                        "reference": self._mask_artifact(
                            root, "pooled_ref", reference, ["a"]
                        ),
                        "candidate": self._mask_artifact(
                            root, "pooled_cand", candidate, ["a"]
                        ),
                    },
                )
            )
            self.assertIn(
                "segmentation.semantic_raster_pooled_miou", result.metrics
            )
            self.assertEqual(
                result.metrics["segmentation.semantic_raster_evaluated_pixel_count"],
                4,
            )
            self.assertEqual(result.metadata["coverage"]["mask_count"], 1)
            self.assertEqual(result.metadata["class_policy"]["included_labels"], [0, 1])
            self.assertIn("not instance segmentation", result.metadata["construct"])

    def test_confidence_intervals_use_student_t_and_are_unknown_for_one_sample(self):
        mean, stddev, ci95 = _mean_std_ci95([3.0])
        self.assertEqual(mean, 3.0)
        self.assertIsNone(stddev)
        self.assertIsNone(ci95)

        mean, stddev, ci95 = _mean_std_ci95([1.0, 3.0])
        self.assertEqual(mean, 2.0)
        self.assertAlmostEqual(stddev, np.sqrt(2.0))
        self.assertAlmostEqual(ci95, 12.706, places=6)

    def test_plot_rows_exclude_resource_rejections_and_compute_exact_paired_contrasts(self):
        result = {
            "benchmark": {"id": "publication", "version": "1"},
            "recipes": [
                {
                    "id": "bad",
                    "label": "candidate",
                    "role": "candidate",
                    "status": "rejected_resource_budget",
                    "resource_admission": {"admitted": False},
                    "metrics": {"channel.snr_db": 0.0, "quality.psnr_db": 1000.0},
                },
                {
                    "id": "baseline_1",
                    "label": "baseline",
                    "role": "baseline",
                    "run_id": "baseline_run_1",
                    "pairing_id": "1",
                    "aggregation_cell_id": "snr_0",
                    "statistical_unit": "paired_seed",
                    "status": "completed",
                    "metrics": {"channel.snr_db": 0.0, "quality.psnr_db": 10.0},
                },
                {
                    "id": "baseline_2",
                    "label": "baseline",
                    "role": "baseline",
                    "run_id": "baseline_run_2",
                    "pairing_id": "2",
                    "aggregation_cell_id": "snr_0",
                    "statistical_unit": "paired_seed",
                    "status": "completed",
                    "metrics": {"channel.snr_db": 0.0, "quality.psnr_db": 20.0},
                },
                {
                    "id": "candidate_1",
                    "label": "candidate",
                    "role": "candidate",
                    "run_id": "candidate_run_1",
                    "pairing_id": "1",
                    "aggregation_cell_id": "snr_0",
                    "statistical_unit": "paired_seed",
                    "status": "completed",
                    "metrics": {"channel.snr_db": 0.0, "quality.psnr_db": 12.0},
                },
                {
                    "id": "candidate_2",
                    "label": "candidate",
                    "role": "candidate",
                    "run_id": "candidate_run_2",
                    "pairing_id": "2",
                    "aggregation_cell_id": "snr_0",
                    "statistical_unit": "paired_seed",
                    "status": "completed",
                    "metrics": {"channel.snr_db": 0.0, "quality.psnr_db": 24.0},
                },
            ],
        }
        raw = _plot_rows(result, "graceful-degradation")
        self.assertNotIn("bad", {row["recipe_id"] for row in raw})
        aggregated = _aggregate_plot_rows(raw)
        candidate = next(row for row in aggregated if row["series"] == "candidate")
        self.assertEqual(candidate["paired_contrast_status"], "computed")
        self.assertEqual(candidate["paired_sample_count"], 2)
        self.assertEqual(candidate["paired_difference_mean"], 3.0)
        self.assertEqual(candidate["paired_ids"], "1|2")
        self.assertEqual(
            candidate["y_ci95_method"],
            "student_t_95_two_sided_tabulated_conservative_approximation",
        )

        for recipe in result["recipes"]:
            if recipe.get("role") == "candidate" and recipe.get("status") == "completed":
                recipe["aggregation_cell_id"] = "different_dataset"
        mismatched = _aggregate_plot_rows(
            _plot_rows(result, "graceful-degradation")
        )
        mismatched_candidate = next(
            row for row in mismatched if row["series"] == "candidate"
        )
        self.assertEqual(
            mismatched_candidate["paired_contrast_status"],
            "comparison_cell_mismatch",
        )
        self.assertIsNone(mismatched_candidate["paired_difference_mean"])

    def test_plot_metric_projection_is_common_and_numeric_for_every_method(self):
        mixed = {
            "benchmark": {"id": "mixed", "version": "1"},
            "recipes": [
                {
                    "id": "psnr",
                    "label": "psnr",
                    "role": "candidate",
                    "run_id": "run-psnr",
                    "status": "completed",
                    "metrics": {"channel.snr_db": 0.0, "quality.psnr_db": 20.0},
                },
                {
                    "id": "ssim",
                    "label": "ssim",
                    "role": "candidate",
                    "run_id": "run-ssim",
                    "status": "completed",
                    "metrics": {"channel.snr_db": 0.0, "quality.ms_ssim": 0.9},
                },
            ],
        }
        with self.assertRaisesRegex(BenchmarkError, "single y-axis metric"):
            _plot_rows(mixed, "graceful-degradation")

        methods = [
            {"id": "complete", "metrics": {"x": 1.0, "y": 2.0}},
            {"id": "missing", "metrics": {"x": 1.0}},
        ]
        with self.assertRaisesRegex(DemoExportError, "missing"):
            _validate_plot_metrics(
                {"id": "selective", "kind": "line", "x": "x", "y": "y"},
                methods,
            )
        with self.assertRaisesRegex(DemoExportError, "missing"):
            _demo_plot_rows(
                {"id": "selective", "kind": "line", "x": "x", "y": "y"},
                methods,
            )

    def test_plot_numeric_semantics_reject_boolean_and_string_numbers(self):
        for invalid in (True, "1.0"):
            result = {
                "benchmark": {"id": "numeric", "version": "1"},
                "recipes": [
                    {
                        "id": "candidate",
                        "label": "candidate",
                        "role": "candidate",
                        "run_id": "run",
                        "status": "completed",
                        "metrics": {"channel.snr_db": invalid, "quality.psnr_db": 1.0},
                    }
                ],
            }
            with self.subTest(invalid=invalid):
                with self.assertRaises(BenchmarkError):
                    _plot_rows(result, "graceful-degradation")
                with self.assertRaises(DemoExportError):
                    _demo_plot_rows(
                        {"id": "numeric", "kind": "line", "x": "x", "y": "y"},
                        [{"id": "candidate", "metrics": {"x": invalid, "y": 1.0}}],
                    )

    def test_contrast_status_checks_reference_before_comparison_metadata(self):
        result = {
            "benchmark": {"id": "candidate-only", "version": "1"},
            "recipes": [
                {
                    "id": "candidate",
                    "label": "candidate",
                    "role": "candidate",
                    "run_id": "run",
                    "status": "completed",
                    "metrics": {"channel.snr_db": 0.0, "quality.psnr_db": 1.0},
                }
            ],
        }
        benchmark_row = _aggregate_plot_rows(
            _plot_rows(result, "graceful-degradation")
        )[0]
        self.assertEqual(
            benchmark_row["paired_contrast_status"], "no_declared_reference"
        )
        demo_row = _demo_plot_rows(
            {"id": "candidate", "kind": "line", "x": "x", "y": "y"},
            [
                {
                    "id": "candidate",
                    "series": "candidate",
                    "role": "candidate",
                    "run_id": "run",
                    "metrics": {"x": 0.0, "y": 1.0},
                }
            ],
        )[0]
        self.assertEqual(
            demo_row["paired_contrast_status"], "no_declared_reference"
        )

    def test_demo_plot_order_and_bar_semantics_fail_closed(self):
        with self.assertRaisesRegex(DemoExportError, "unique"):
            _normalize_plot_specs(
                [
                    {
                        "id": "duplicate",
                        "kind": "line",
                        "x": "x",
                        "y": "y",
                        "method_order": ["A", "A"],
                    }
                ]
            )
        with self.assertRaisesRegex(DemoExportError, "cannot declare x"):
            _normalize_plot_specs(
                [{"id": "bar", "kind": "bar", "x": "x", "y": "y"}]
            )

    def test_series_summaries_do_not_pool_distinct_aggregation_cells(self):
        summaries = _series_summaries(
            [
                {
                    "id": "a",
                    "series": "candidate",
                    "role": "candidate",
                    "run_id": "run-a",
                    "paired_seed": "a",
                    "aggregation_cell_id": "dataset-a",
                    "statistical_unit": "image",
                    "metrics": {"score": 1.0},
                    "evidence": {"run": "a"},
                },
                {
                    "id": "b",
                    "series": "candidate",
                    "role": "candidate",
                    "run_id": "run-b",
                    "paired_seed": "b",
                    "aggregation_cell_id": "dataset-b",
                    "statistical_unit": "image",
                    "metrics": {"score": 3.0},
                    "evidence": {"run": "b"},
                },
            ],
            [{"id": "score"}],
        )
        self.assertEqual(summaries[0]["metrics"], {})
        self.assertEqual(len(summaries[0]["metrics_by_aggregation_cell"]), 2)
        self.assertEqual(summaries[0]["representative_evidence"], {})

    def test_plot_aggregation_rejects_undeclared_or_duplicate_statistical_units(self):
        result = {
            "benchmark": {"id": "publication", "version": "1"},
            "recipes": [
                {
                    "id": "candidate_1",
                    "label": "candidate",
                    "role": "candidate",
                    "run_id": "run_1",
                    "pairing_id": "same",
                    "status": "completed",
                    "metrics": {"channel.snr_db": 0.0, "quality.psnr_db": 12.0},
                },
                {
                    "id": "candidate_2",
                    "label": "candidate",
                    "role": "candidate",
                    "run_id": "run_2",
                    "pairing_id": "same",
                    "status": "completed",
                    "metrics": {"channel.snr_db": 0.0, "quality.psnr_db": 13.0},
                },
            ],
        }
        with self.assertRaisesRegex(BenchmarkError, "aggregation_cell_id"):
            _aggregate_plot_rows(_plot_rows(result, "graceful-degradation"))

        for row in result["recipes"]:
            row["aggregation_cell_id"] = "snr_0"
            row["statistical_unit"] = "paired_seed"
        with self.assertRaisesRegex(BenchmarkError, "unique pairing_id"):
            _aggregate_plot_rows(_plot_rows(result, "graceful-degradation"))

        methods = [
            {
                "id": "candidate_1",
                "series": "candidate",
                "role": "candidate",
                "run_id": "run_1",
                "paired_seed": 1,
                "metrics": {"channel.snr_db": 0.0, "task.score": 1.0},
            },
            {
                "id": "candidate_2",
                "series": "candidate",
                "role": "candidate",
                "run_id": "run_2",
                "paired_seed": 2,
                "metrics": {"channel.snr_db": 0.0, "task.score": 2.0},
            },
        ]
        with self.assertRaisesRegex(DemoExportError, "aggregation_cell_id"):
            _demo_plot_rows(
                {
                    "id": "invalid-aggregation",
                    "kind": "line",
                    "x": "channel.snr_db",
                    "y": "task.score",
                    "group": "method",
                    "style": {"aggregation": "mean_ci", "y_scale": "linear"},
                },
                methods,
            )

    def test_demo_plot_does_not_serialize_single_sample_uncertainty_as_zero(self):
        rows = _demo_plot_rows(
            {
                "id": "paired",
                "kind": "line",
                "x": "channel.snr_db",
                "y": "task.score",
                "group": "method",
                "style": {"aggregation": "mean_ci", "y_scale": "linear"},
            },
            [
                {
                    "id": "baseline",
                    "label": "baseline",
                    "series": "baseline",
                    "role": "baseline",
                    "run_id": "base_1",
                    "paired_seed": 1,
                    "aggregation_cell_id": "snr_0",
                    "statistical_unit": "paired_seed",
                    "metrics": {"channel.snr_db": 0.0, "task.score": 2.0},
                },
                {
                    "id": "candidate",
                    "label": "candidate",
                    "series": "candidate",
                    "role": "candidate",
                    "run_id": "candidate_1",
                    "paired_seed": 1,
                    "aggregation_cell_id": "snr_0",
                    "statistical_unit": "paired_seed",
                    "metrics": {"channel.snr_db": 0.0, "task.score": 3.0},
                },
            ],
        )
        candidate = next(row for row in rows if row["series_id"] == "candidate")
        self.assertIsNone(candidate["y_stddev"])
        self.assertIsNone(candidate["y_ci95"])
        self.assertEqual(candidate["y_ci95_method"], "not_estimable_n_lt_2")
        self.assertEqual(candidate["paired_difference_mean"], 1.0)
        self.assertIsNone(candidate["paired_difference_ci95"])

    def test_demo_svg_accessibility_text_does_not_claim_pairing(self):
        plot = {
            "title": "Single observation",
            "kind": "line",
            "style": {"aggregation": "mean_ci", "y_scale": "linear"},
        }
        rows = [
            {
                "series": "candidate",
                "method_id": "candidate",
                "label": "Candidate",
                "x_value": 0.0,
                "y_value": 1.0,
                "y_ci95": None,
            }
        ]
        svg = _render_plot_svg(plot, rows)
        self.assertNotIn("repeated paired seeds", svg)
        self.assertIn("arithmetic mean of the observations represented by its row", svg)


if __name__ == "__main__":
    unittest.main()
