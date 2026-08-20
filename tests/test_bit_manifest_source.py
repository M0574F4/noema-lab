from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from noema_lab.core.artifacts import file_sha256
from noema_lab.core.common_conditions import _source_evidence
from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.ops import build_registry
from noema_lab.ops.source.bit_manifest import (
    BIT_MANIFEST_KIND,
    BIT_PACKING,
    BitManifestSourceOperation,
    bit_payload_content_sha256,
)


def _item(item_id: str, group_id: str, bits: list[int]) -> dict:
    array = np.asarray(bits, dtype=np.uint8)
    packed_hex = np.packbits(array, bitorder="big").tobytes().hex()
    return {
        "item_id": item_id,
        "group_id": group_id,
        "bit_count": len(bits),
        "packed_hex": packed_hex,
        "content_sha256": bit_payload_content_sha256(len(bits), packed_hex),
    }


def _manifest() -> dict:
    items = [
        _item("dev-001", "trajectory-01", [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]),
        _item("dev-002", "trajectory-02", [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]),
        _item("test-001", "trajectory-03", [1, 1, 0, 0, 0, 1, 1, 0, 0, 1]),
    ]
    return {
        "schema_version": 1,
        "kind": BIT_MANIFEST_KIND,
        "id": "delayed-csi-payloads-v1",
        "version": "2026-08-11",
        "encoding": BIT_PACKING,
        "ordered_item_ids": [item["item_id"] for item in items],
        "splits": {
            "development": ["dev-001", "dev-002"],
            "heldout": ["test-001"],
        },
        "items": items,
    }


class BitManifestSourceOperationTests(unittest.TestCase):
    def _write_manifest(
        self,
        root: Path,
        manifest: dict,
        *,
        suffix: str = ".json",
        name: str = "payloads",
    ) -> Path:
        path = root / (name + suffix)
        if suffix == ".json":
            text = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
        else:
            text = yaml.safe_dump(manifest, sort_keys=False)
        path.write_text(text, encoding="utf-8")
        return path

    def _run(
        self,
        root: Path,
        manifest_path: Path,
        *,
        selection: str = "development",
        item_ids: list[str] | None = None,
        expected_sha256: str | None = None,
        step_name: str = "data",
    ):
        params = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": expected_sha256 or file_sha256(manifest_path),
            "selection": selection,
            "item_ids": list(item_ids or []),
        }
        return BitManifestSourceOperation().run(
            OperationContext(
                recipe_name="frozen-payload-test",
                step_id=step_name,
                params=params,
                inputs={},
                run_dir=root,
                step_dir=root / step_name,
            )
        )

    def test_named_split_materializes_canonical_bits_and_complete_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _manifest()
            path = self._write_manifest(root, manifest)
            result = self._run(root, path)
            with np.load(result.outputs["bits"].path, allow_pickle=False) as payload:
                bits = payload["bits"].copy()
                stored_metadata = json.loads(str(payload["metadata_json"]))

            expected = np.concatenate(
                [
                    np.unpackbits(
                        np.frombuffer(bytes.fromhex(item["packed_hex"]), dtype=np.uint8),
                        bitorder="big",
                    )[: item["bit_count"]]
                    for item in manifest["items"][:2]
                ]
            )
            np.testing.assert_array_equal(bits, expected)
            self.assertEqual(bits.dtype, np.uint8)
            self.assertEqual(bits.shape, (20,))

            metadata = result.outputs["bits"].metadata
            self.assertEqual(
                stored_metadata,
                {key: metadata[key] for key in stored_metadata},
            )
            self.assertEqual(metadata["source_item_ids"], ["dev-001", "dev-002"])
            self.assertEqual(
                metadata["source_group_ids"], ["trajectory-01", "trajectory-02"]
            )
            self.assertEqual(
                metadata["source_split_ids"], ["development", "development"]
            )
            self.assertEqual(metadata["source_item_payload_bit_counts"], [10, 10])
            self.assertEqual(metadata["capture_record_count"], 2)
            self.assertEqual(metadata["capture_record_shape"], [10])
            self.assertEqual(metadata["transport_block_size_bits"], 10)
            self.assertEqual(metadata["transport_block_count"], 2)
            self.assertEqual(metadata["dataset_manifest_sha256"], file_sha256(path))
            self.assertEqual(
                metadata["source_operation_contract_sha256"],
                canonical_json_sha256(metadata["source_operation_contract"]),
            )
            self.assertEqual(
                metadata["ordered_post_transform_sha256"],
                canonical_json_sha256(metadata["ordered_post_transform_items"]),
            )
            self.assertEqual(len(metadata["batch_tensor_sha256"]), 64)
            self.assertTrue(
                all(
                    row["source_sha256"] == row["post_transform_sha256"]
                    and row["split"] == "development"
                    and row["ancestry_ids"] == [row["source_id"]]
                    for row in metadata["source_items"]
                )
            )
            self.assertEqual(result.metrics["source.example_count"], 2)
            self.assertEqual(result.metrics["channel.payload_bit_count"], 20)
            source_evidence = _source_evidence(
                [
                    {
                        "op": "source.bit_manifest",
                        "outputs": {"bits": {"metadata": metadata}},
                    }
                ]
            )
            self.assertTrue(source_evidence["complete"])
            self.assertEqual(
                source_evidence["ordered_item_ids"], ["dev-001", "dev-002"]
            )

    def test_yaml_and_explicit_ordered_item_selection_are_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_manifest(root, _manifest(), suffix=".yaml")
            result = self._run(
                root,
                path,
                selection="",
                item_ids=["dev-002", "test-001"],
            )
            metadata = result.outputs["bits"].metadata
            self.assertEqual(metadata["source_item_ids"], ["dev-002", "test-001"])
            self.assertEqual(
                metadata["source_split_ids"], ["development", "heldout"]
            )
            self.assertIsNone(metadata["split"])
            self.assertIsNone(metadata["selection"])
            self.assertEqual(metadata["capture_record_shape"], [10])

    def test_operation_is_registered_with_source_binding_guarantees(self):
        operation = build_registry().get("source.bit_manifest")
        description = operation.describe()
        self.assertEqual(description["equivalence"]["type"], "exact")
        self.assertEqual(description["backends"]["benchmark_run"], ["numpy"])
        guarantees = description["output_metadata_guarantees"]["bits"]
        for field in (
            "dataset_manifest_sha256",
            "source_operation_contract_sha256",
            "ordered_post_transform_sha256",
            "source_item_ids",
            "source_item_content_sha256",
            "source_group_ids",
            "source_split_ids",
            "source_items",
        ):
            self.assertIn(field, guarantees)

    def test_rejects_unpinned_unsafe_or_ambiguous_selection(self):
        operation = BitManifestSourceOperation()
        with self.assertRaisesRegex(OperationError, "local path without a URI scheme"):
            operation.validate_preflight(
                {
                    "manifest_path": "https://example.test/payloads.json",
                    "manifest_sha256": "a" * 64,
                    "selection": "development",
                }
            )
        with self.assertRaisesRegex(OperationError, "suffix"):
            operation.validate_preflight(
                {
                    "manifest_path": "payloads.txt",
                    "manifest_sha256": "a" * 64,
                    "selection": "development",
                }
            )
        with self.assertRaisesRegex(OperationError, "exactly one"):
            operation.validate_preflight(
                {
                    "manifest_path": "payloads.json",
                    "manifest_sha256": "a" * 64,
                    "selection": "",
                    "item_ids": [],
                }
            )
        with self.assertRaisesRegex(OperationError, "exactly one"):
            operation.validate_preflight(
                {
                    "manifest_path": "payloads.json",
                    "manifest_sha256": "a" * 64,
                    "selection": "development",
                    "item_ids": ["dev-001"],
                }
            )
        with self.assertRaisesRegex(OperationError, "duplicates"):
            operation.validate_preflight(
                {
                    "manifest_path": "payloads.json",
                    "manifest_sha256": "a" * 64,
                    "item_ids": ["dev-001", "DEV-001"],
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_manifest(root, _manifest())
            with self.assertRaisesRegex(OperationError, "SHA-256 mismatch"):
                self._run(root, path, expected_sha256="0" * 64)
            with self.assertRaisesRegex(OperationError, "no selection"):
                self._run(root, path, selection="missing")
            with self.assertRaisesRegex(OperationError, "preserve.*order"):
                self._run(
                    root,
                    path,
                    selection="",
                    item_ids=["test-001", "dev-001"],
                )
            with self.assertRaisesRegex(OperationError, "Unknown"):
                self._run(
                    root,
                    path,
                    selection="",
                    item_ids=["unknown"],
                )
            symlink = root / "payload-link.json"
            symlink.symlink_to(path)
            with self.assertRaisesRegex(OperationError, "missing or unsafe"):
                self._run(root, symlink)

    def test_rejects_schema_order_content_and_split_failures(self):
        cases = []

        payload = _manifest()
        payload["unexpected"] = True
        cases.append((payload, "unknown field"))

        payload = _manifest()
        payload["ordered_item_ids"] = list(reversed(payload["ordered_item_ids"]))
        cases.append((payload, "exactly match items order"))

        payload = _manifest()
        payload["items"][0]["content_sha256"] = "0" * 64
        cases.append((payload, "content SHA-256 mismatch"))

        payload = _manifest()
        payload["items"][1]["item_id"] = "dev-001"
        cases.append((payload, "duplicate item IDs"))

        payload = _manifest()
        payload["items"][1]["packed_hex"] = payload["items"][0]["packed_hex"]
        payload["items"][1]["bit_count"] = payload["items"][0]["bit_count"]
        payload["items"][1]["content_sha256"] = payload["items"][0]["content_sha256"]
        cases.append((payload, "duplicate payload content"))

        payload = _manifest()
        payload["items"][0]["packed_hex"] = "b281"
        payload["items"][0]["content_sha256"] = bit_payload_content_sha256(10, "b281")
        cases.append((payload, "non-zero unused padding bits"))

        payload = _manifest()
        payload["splits"]["development"] = ["dev-002", "dev-001"]
        cases.append((payload, "does not preserve manifest item order"))

        payload = _manifest()
        payload["splits"]["heldout"] = ["dev-001", "test-001"]
        cases.append((payload, "appears in multiple splits"))

        payload = _manifest()
        payload["splits"]["heldout"] = ["missing"]
        cases.append((payload, "unknown item ID"))

        payload = _manifest()
        del payload["splits"]["heldout"]
        cases.append((payload, "not assigned to a split"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, (manifest, expected) in enumerate(cases):
                with self.subTest(expected=expected):
                    path = self._write_manifest(
                        root, manifest, name="invalid-%d" % index
                    )
                    with self.assertRaisesRegex(OperationError, expected):
                        self._run(root, path, step_name="invalid-%d" % index)

    def test_rejects_nonuniform_selection_and_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _manifest()
            short_bits = [1, 0, 1, 0, 0, 1, 0, 1]
            manifest["items"][1] = _item(
                "dev-002", "trajectory-02", short_bits
            )
            path = self._write_manifest(root, manifest, name="nonuniform")
            with self.assertRaisesRegex(OperationError, "uniform bit_count"):
                self._run(root, path)

            valid_text = json.dumps(_manifest(), separators=(",", ":"))
            duplicate_text = valid_text.replace(
                '"schema_version":1,',
                '"schema_version":1,"schema_version":1,',
                1,
            )
            duplicate_path = root / "duplicate.json"
            duplicate_path.write_text(duplicate_text, encoding="utf-8")
            with self.assertRaisesRegex(OperationError, "Duplicate JSON object key"):
                self._run(root, duplicate_path, step_name="duplicate")


if __name__ == "__main__":
    unittest.main()
