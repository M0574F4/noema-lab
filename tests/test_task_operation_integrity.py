from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from noema_lab.core.artifacts import Artifact
from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.ops.foundation import (
    KnowledgeBaseSourceOperation,
    SemanticStateGroundOperation,
    SemanticStatePayloadDecodeOperation,
    SemanticStatePayloadEncodeOperation,
    SemanticStateToTextOperation,
    TextSemanticStateEncodeOperation,
)
from noema_lab.ops.channel.digital import SymbolPowerNormalizeOperation
from noema_lab.ops.metrics.task import DetectionMetricsOperation
from noema_lab.ops.models.text_codec import (
    BART_FRAME_FIXED_HEADER_SYMBOLS,
    TextBartJsccDecodeOperation,
    TextBartJsccEncodeOperation,
    _decode_bart_item_frames,
    _validated_bart_example_metadata,
)
from noema_lab.ops.source.coco_yolo import (
    _read_yolo_boxes,
    _read_yolo_segmentation_mask,
)
from noema_lab.ops.source.text_dataset import TEXT_SMOKE_EXAMPLES
from noema_lab.ops.source.vqa_manifest import _load_manifest_examples
from noema_lab.ops.vqa_goal import (
    VqaSemanticSelectOperation,
    VqaTransformersAnswerOperation,
)


class TaskOperationIntegrityTests(unittest.TestCase):
    def _context(
        self,
        root: Path,
        step: str,
        inputs: dict[str, Artifact] | None = None,
        params: dict | None = None,
    ) -> OperationContext:
        return OperationContext(
            recipe_name="task_operation_integrity",
            step_id=step,
            params=dict(params or {}),
            inputs=dict(inputs or {}),
            run_dir=root,
            step_dir=root / step,
        )

    def _json_artifact(
        self, root: Path, name: str, kind: str, payload: dict
    ) -> Artifact:
        path = root / (name + ".json")
        path.write_text(json.dumps(payload), encoding="utf-8")
        return Artifact(kind, path, {})

    def test_inline_facts_and_vqa_manifests_reject_ambiguous_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(OperationError, "Duplicate JSON object key"):
                KnowledgeBaseSourceOperation().run(
                    self._context(
                        root,
                        "knowledge_base",
                        params={
                            "kb_id": "inline_json",
                            "facts_json": (
                                '[{"subject":"one","subject":"two",'
                                '"predicate":"is","object":"value"}]'
                            ),
                        },
                    )
                )

            for suffix, content in (
                (
                    ".json",
                    (
                        '[{"image":"x.png","question":"first",'
                        '"question":"second","answer":"a"}]'
                    ),
                ),
                (
                    ".jsonl",
                    (
                        '{"image":"x.png","question":"first",'
                        '"question":"second","answer":"a"}\n'
                    ),
                ),
            ):
                path = root / ("vqa" + suffix)
                path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(
                    OperationError,
                    "Duplicate JSON object key",
                ):
                    _load_manifest_examples(path)

    def test_semantic_transport_redacts_answer_side_channels_and_counts_all_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_examples = TEXT_SMOKE_EXAMPLES[:2]
            canonicals = [str(item["text"]) for item in source_examples]
            sample_ids = [str(item["id"]) for item in source_examples]
            texts = self._json_artifact(
                root,
                "masked_text",
                "text.batch.json",
                {
                    "schema_version": 1,
                    "kind": "text.batch",
                    "examples": [
                        {"id": sample_id, "text": "[MASK]"}
                        for sample_id in sample_ids
                    ],
                },
            )
            encoded = TextSemanticStateEncodeOperation().run(
                self._context(
                    root,
                    "semantic_encode",
                    {"texts": texts},
                    {"include_text": True},
                )
            )
            state_payload = json.loads(
                encoded.outputs["state"].path.read_text(encoding="utf-8")
            )
            for state_item, sample_id, canonical in zip(
                state_payload["states"], sample_ids, canonicals
            ):
                state_item["source_id"] = sample_id
                state_item["source_text"] = canonical
                state_item["kb_matches"] = [
                    {
                        "fact_id": sample_id + ":canonical_text",
                        "overlap": ["mask"],
                        "fact": {
                            "subject": sample_id,
                            "predicate": "canonical_text",
                            "object": canonical,
                        },
                    }
                ]
            encoded.outputs["state"].path.write_text(
                json.dumps(state_payload), encoding="utf-8"
            )

            sent = SemanticStatePayloadEncodeOperation().run(
                self._context(
                    root,
                    "semantic_send",
                    {"state": encoded.outputs["state"]},
                )
            )
            with np.load(sent.outputs["bits"].path, allow_pickle=False) as archive:
                bits = archive["bits"]
                metadata = json.loads(str(archive["metadata_json"]))
            self.assertEqual(
                sum(metadata["source_item_payload_bit_counts"]), int(bits.size)
            )
            self.assertEqual(metadata["payload_bit_count"], int(bits.size))
            self.assertEqual(metadata["source_item_count"], 2)
            self.assertEqual(
                len(metadata["source_item_payload_bit_counts"]), 2
            )
            self.assertTrue(
                all(
                    value > 0
                    for value in metadata["source_item_payload_bit_counts"]
                )
            )
            self.assertGreater(metadata["semantic_transport_redacted_field_count"], 0)

            decoded = SemanticStatePayloadDecodeOperation().run(
                self._context(
                    root,
                    "semantic_decode",
                    {"bits": sent.outputs["bits"]},
                    {"on_error": "fail"},
                )
            )
            decoded_payload = json.loads(
                decoded.outputs["state"].path.read_text(encoding="utf-8")
            )
            decoded_json = json.dumps(decoded_payload, sort_keys=True)
            self.assertNotIn("source_id", decoded_json)
            self.assertNotIn("source_text", decoded_json)
            self.assertNotIn("kb_matches", decoded_json)
            for canonical in canonicals:
                self.assertNotIn(canonical, decoded_json)

            kb = KnowledgeBaseSourceOperation().run(
                self._context(root, "knowledge_base")
            )
            grounded = SemanticStateGroundOperation().run(
                self._context(
                    root,
                    "receiver_ground",
                    {"state": decoded.outputs["state"], "kb": kb.outputs["kb"]},
                )
            )
            reconstructed = SemanticStateToTextOperation().run(
                self._context(
                    root,
                    "semantic_receive",
                    {"state": grounded.outputs["state"], "kb": kb.outputs["kb"]},
                    {"generator": "kb_reconstruct", "prefer_source_text": False},
                )
            )
            generated = json.loads(
                reconstructed.outputs["texts"].path.read_text(encoding="utf-8")
            )["examples"]
            self.assertEqual(len(generated), 2)
            for item, canonical in zip(generated, canonicals):
                self.assertNotEqual(item["text"], canonical)

    def test_vqa_operations_reject_positional_identity_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "images.npz"
            image_metadata = {"image_ids": ["image-a", "image-b"]}
            np.savez_compressed(
                image_path,
                images=np.zeros((2, 4, 4, 3), dtype=np.uint8),
                metadata_json=json.dumps(image_metadata),
            )
            images = Artifact("image.batch.numpy", image_path, image_metadata)
            questions = self._json_artifact(
                root,
                "questions",
                "vqa.questions.json",
                {
                    "schema_version": 1,
                    "kind": "vqa.questions",
                    "examples": [
                        {
                            "id": "q-a",
                            "image_id": "image-b",
                            "question": "first?",
                        },
                        {
                            "id": "q-b",
                            "image_id": "image-a",
                            "question": "second?",
                        },
                    ],
                },
            )
            for index, operation in enumerate(
                (VqaSemanticSelectOperation(), VqaTransformersAnswerOperation())
            ):
                with self.subTest(operation=operation.id):
                    with self.assertRaisesRegex(
                        OperationError, "identity mismatch at index 0"
                    ):
                        operation.run(
                            self._context(
                                root,
                                "vqa_identity_%d" % index,
                                {"images": images, "questions": questions},
                            )
                        )

    def test_detection_metric_uses_upstream_model_prefilter_in_metric_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = self._json_artifact(
                root,
                "detection_reference",
                "vision.detections.json",
                {
                    "schema_version": 1,
                    "kind": "vision.detections",
                    "examples": [
                        {
                            "id": "image",
                            "detections": [
                                {"label": "car", "bbox": [0, 0, 2, 2]}
                            ],
                        }
                    ],
                },
            )
            candidate = self._json_artifact(
                root,
                "detection_candidate",
                "vision.detections.json",
                {
                    "schema_version": 1,
                    "kind": "vision.detections",
                    "inference_confidence_threshold": 0.25,
                    "predictions_pre_filtered_by_model": True,
                    "examples": [
                        {
                            "id": "image",
                            "detections": [
                                {
                                    "label": "car",
                                    "bbox": [0, 0, 2, 2],
                                    "score": 0.4,
                                }
                            ],
                        }
                    ],
                },
            )
            result = DetectionMetricsOperation().run(
                self._context(
                    root,
                    "detection_metrics",
                    {"reference": reference, "candidate": candidate},
                    {"iou_threshold": 0.5, "confidence_threshold": 0.0},
                )
            )
            self.assertIn(
                "detection.f1_at_iou_0p5_conf_0p25", result.metrics
            )
            self.assertNotIn("detection.f1_at_iou_0p5", result.metrics)
            self.assertEqual(
                result.metadata["candidate_prefilter_confidence_threshold"], 0.25
            )
            self.assertEqual(result.metadata["effective_confidence_threshold"], 0.25)

    def test_coco_labels_fail_closed_but_allow_explicit_empty_negatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.txt"
            with self.assertRaisesRegex(OperationError, "label file is missing"):
                _read_yolo_boxes(missing, 32, 32)
            with self.assertRaisesRegex(OperationError, "label file is missing"):
                _read_yolo_segmentation_mask(missing, 32, 32)

            empty = root / "empty.txt"
            empty.write_text("\n", encoding="utf-8")
            self.assertEqual(_read_yolo_boxes(empty, 32, 32), [])
            self.assertFalse(_read_yolo_segmentation_mask(empty, 32, 32).any())

            incomplete = root / "incomplete.txt"
            incomplete.write_text("0 0.5 0.5 0.25\n", encoding="utf-8")
            with self.assertRaisesRegex(OperationError, "expected exactly 5 fields"):
                _read_yolo_boxes(incomplete, 32, 32)

            malformed = root / "malformed.txt"
            malformed.write_text("0 0.1 0.1 0.2 nope\n", encoding="utf-8")
            with self.assertRaisesRegex(OperationError, "is not numeric"):
                _read_yolo_boxes(malformed, 32, 32)

            degenerate = root / "degenerate.txt"
            degenerate.write_text(
                "0 0.1 0.1 0.2 0.2 0.3 0.3\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(OperationError, "polygon is degenerate"):
                _read_yolo_segmentation_mask(degenerate, 32, 32)

    def test_bart_encoder_declares_additive_source_item_symbol_partition(self):
        class FakeTensor:
            def __init__(self, values):
                self.values = np.asarray(values)
                self.shape = self.values.shape

            def to(self, _device):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        class FakeTokenizer:
            def __call__(self, text, **_kwargs):
                length = 2 if text else 1
                return {
                    "input_ids": FakeTensor(
                        np.arange(length, dtype=np.int64).reshape(1, length)
                    ),
                    "attention_mask": FakeTensor(
                        np.ones((1, length), dtype=np.int64)
                    ),
                }

        class FakeEncoder:
            def eval(self):
                return None

            def __call__(self, input_ids, **_kwargs):
                length = int(input_ids.shape[-1])
                return SimpleNamespace(
                    last_hidden_state=FakeTensor(
                        np.arange(1, 1 + length * 4, dtype=np.float32).reshape(
                            1, length, 4
                        )
                    )
                )

        class FakeModel:
            def __init__(self):
                self.encoder = FakeEncoder()

            def get_encoder(self):
                return self.encoder

            def eval(self):
                return None

        fake_torch = SimpleNamespace(
            no_grad=contextlib.nullcontext,
            device=lambda value: value,
            cuda=SimpleNamespace(is_available=lambda: False),
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            texts = self._json_artifact(
                root,
                "bart_texts",
                "text.batch.json",
                {
                    "schema_version": 1,
                    "kind": "text.batch",
                    "examples": [
                        {"id": "one", "text": "alpha"},
                        {"id": "two", "text": ""},
                    ],
                },
            )
            with mock.patch(
                "noema_lab.ops.models.text_codec._load_seq2seq_model",
                return_value=(FakeTokenizer(), FakeModel(), fake_torch),
            ):
                result = TextBartJsccEncodeOperation().run(
                    self._context(
                        root,
                        "bart_encode",
                        {"texts": texts},
                        {"power_normalize": True},
                    )
                )
            metadata = result.outputs["symbols"].metadata
            with np.load(
                result.outputs["symbols"].path, allow_pickle=False
            ) as archive:
                transmitted_symbols = archive["symbols"].astype(
                    np.complex64, copy=False
                )
            received_frames = _decode_bart_item_frames(transmitted_symbols)
            self.assertEqual(metadata["source_item_count"], 2)
            self.assertEqual(metadata["source_item_ids"], ["one", "two"])
            self.assertEqual(
                sum(metadata["source_item_symbol_counts"]),
                metadata["symbol_count"],
            )
            self.assertTrue(metadata["source_item_use_counts_are_additive"])
            self.assertEqual(metadata["power_normalization_scope"], "none")
            self.assertEqual(
                metadata["encoder_semantic_power_normalization_scope"],
                "source_item",
            )
            self.assertTrue(
                metadata["decoder_side_information_rate_accounted"]
            )
            self.assertEqual(
                metadata["semantic_symbol_count"]
                + metadata["decoder_metadata_channel_use_count"],
                metadata["symbol_count"],
            )
            self.assertEqual(
                [item["decoder_metadata"]["id"] for item in received_frames],
                ["one", "two"],
            )
            self.assertEqual(
                received_frames[0]["decoder_metadata"]["hidden_shape"],
                [1, 2, 4],
            )
            self.assertEqual(
                received_frames[1]["decoder_metadata"]["hidden_shape"],
                [1, 1, 4],
            )
            for item in metadata["examples"]:
                self.assertNotIn("hidden_shape", item)
                self.assertNotIn("scale", item)
                self.assertNotIn("attention_mask", item)
                self.assertNotIn("input_token_count", item)
            _validated_bart_example_metadata(
                metadata,
                int(metadata["symbol_count"]),
                received_frames,
            )
            normalized = SymbolPowerNormalizeOperation().run(
                self._context(
                    root,
                    "bart_source_item_power",
                    {"symbols": result.outputs["symbols"]},
                    {
                        "target_power": 1.0,
                        "normalization_scope": "source_item",
                    },
                )
            )
            normalized_metadata = normalized.outputs["symbols"].metadata
            self.assertEqual(
                normalized_metadata["power_normalization_scope"], "source_item"
            )
            self.assertEqual(
                len(normalized_metadata["source_item_power_after"]), 2
            )
            for value in normalized_metadata["source_item_power_after"]:
                self.assertAlmostEqual(value, 1.0, places=6)
            with np.load(
                normalized.outputs["symbols"].path, allow_pickle=False
            ) as archive:
                normalized_symbols = archive["symbols"].astype(
                    np.complex64, copy=False
                )
            normalized_frames = _decode_bart_item_frames(normalized_symbols)
            self.assertEqual(
                [item["decoder_metadata"]["id"] for item in normalized_frames],
                ["one", "two"],
            )

            corrupted_symbols = transmitted_symbols.copy()
            corrupted_symbols[0] *= np.complex64(-1.0)
            with self.assertRaisesRegex(
                OperationError, "invalid or truncated decoder metadata"
            ):
                _decode_bart_item_frames(corrupted_symbols)
            crc_corrupted_symbols = transmitted_symbols.copy()
            crc_corrupted_symbols[
                BART_FRAME_FIXED_HEADER_SYMBOLS
            ] *= np.complex64(-1.0)
            with self.assertRaisesRegex(OperationError, "CRC32"):
                _decode_bart_item_frames(crc_corrupted_symbols)
            corrupted_path = root / "corrupted_bart_symbols.npz"
            np.savez_compressed(
                corrupted_path,
                symbols=corrupted_symbols,
                metadata_json=json.dumps(metadata),
            )
            corrupted_artifact = Artifact(
                "channel.symbols.complex_numpy",
                corrupted_path,
                metadata,
            )
            with mock.patch(
                "noema_lab.ops.models.text_codec._load_seq2seq_model"
            ) as model_loader:
                with self.assertRaisesRegex(
                    OperationError, "invalid or truncated decoder metadata"
                ):
                    TextBartJsccDecodeOperation().run(
                        self._context(
                            root,
                            "bart_corrupted_decode",
                            {"symbols": corrupted_artifact},
                        )
                    )
                model_loader.assert_not_called()
            corrupted = dict(metadata)
            corrupted["source_item_symbol_counts"] = [1, 1]
            with self.assertRaisesRegex(
                OperationError, "do not cover the received symbol stream"
            ):
                _validated_bart_example_metadata(
                    corrupted, int(metadata["symbol_count"])
                )


if __name__ == "__main__":
    unittest.main()
