from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from noema_lab.ops.models import eflic, learned_codecs, upstream_lic


class RuntimeArtifactStrictInputTests(unittest.TestCase):
    def test_executable_bundle_manifests_reject_duplicate_component_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis = root / "analysis.bin"
            synthesis = root / "synthesis.bin"
            analysis.write_bytes(b"analysis")
            synthesis.write_bytes(b"synthesis")

            cases = (
                ("learned ONNX", learned_codecs._load_onnx_bundle, ""),
                ("learned AOTI", learned_codecs._load_aoti_bundle, ""),
                (
                    "EVC ONNX",
                    upstream_lic._load_onnx_bundle,
                    '"codec":"evc",',
                ),
                (
                    "EF-LIC ONNX",
                    eflic._load_eflic_onnx_bundle,
                    '"codec":"eflic","export_input_shape":[1,3,8,8],',
                ),
                (
                    "EF-LIC AOTI",
                    eflic._load_eflic_aoti_bundle,
                    '"codec":"eflic","runtime":"aot_inductor",'
                    '"export_input_shape":[1,3,8,8],',
                ),
            )
            for index, (label, loader, prefix) in enumerate(cases):
                with self.subTest(runtime=label):
                    manifest = root / ("bundle-%d.json" % index)
                    manifest.write_text(
                        "{%s"
                        '"analysis_path":%s,'
                        '"analysis_path":%s,'
                        '"synthesis_path":%s}'
                        % (
                            prefix,
                            json.dumps(str(root / "attacker-selected.bin")),
                            json.dumps(str(analysis)),
                            json.dumps(str(synthesis)),
                        ),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        RuntimeError, "Duplicate JSON object key `analysis_path`"
                    ):
                        loader(SimpleNamespace(path=manifest))

    def test_executable_bundle_manifests_require_an_object_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "bundle.json"
            manifest.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, "bundle manifest root must be a JSON object"
            ):
                learned_codecs._load_onnx_bundle(SimpleNamespace(path=manifest))

    def test_npz_runtime_metadata_rejects_duplicate_identity_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            duplicate_metadata = (
                '{"checkpoint":"attacker-selected.pt",'
                '"checkpoint":"expected.pt","shape":[1,2,2,3]}'
            )
            cases = (
                (
                    "learned images",
                    learned_codecs._load_images,
                    "images",
                    np.zeros((1, 2, 2, 3), dtype=np.uint8),
                    (),
                ),
                (
                    "learned bits",
                    learned_codecs._load_bits,
                    "bits",
                    np.zeros(8, dtype=np.uint8),
                    ({},),
                ),
                (
                    "learned latents",
                    learned_codecs._load_latents,
                    "latents",
                    np.zeros((1, 1, 1, 1), dtype=np.float32),
                    ({},),
                ),
                (
                    "EVC images",
                    upstream_lic._load_images,
                    "images",
                    np.zeros((1, 2, 2, 3), dtype=np.uint8),
                    (),
                ),
                (
                    "EVC bits",
                    upstream_lic._load_bits,
                    "bits",
                    np.zeros(8, dtype=np.uint8),
                    ({},),
                ),
                (
                    "EF-LIC images",
                    eflic._load_images,
                    "images",
                    np.zeros((1, 2, 2, 3), dtype=np.uint8),
                    (),
                ),
                (
                    "EF-LIC bits",
                    eflic._load_bits,
                    "bits",
                    np.zeros(8, dtype=np.uint8),
                    ({},),
                ),
                (
                    "EF-LIC indices",
                    eflic._load_eflic_indices,
                    "z_inds_0",
                    np.zeros((1, 1), dtype=np.int64),
                    ({},),
                ),
            )
            for index, (label, loader, field, values, args) in enumerate(cases):
                with self.subTest(artifact=label):
                    artifact_path = root / ("artifact-%d.npz" % index)
                    np.savez_compressed(
                        artifact_path,
                        **{field: values, "metadata_json": duplicate_metadata},
                    )
                    with self.assertRaisesRegex(
                        RuntimeError, "Duplicate JSON object key `checkpoint`"
                    ):
                        loader(artifact_path, *args)


if __name__ == "__main__":
    unittest.main()
