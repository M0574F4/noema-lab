from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

from noema_lab.ops.models.onnx_evidence import onnxruntime_native_evidence


class OnnxRuntimeEvidenceTests(unittest.TestCase):
    def test_reports_native_engine_and_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "onnxruntime"
            capi = root / "capi"
            include = root / "include"
            capi.mkdir(parents=True)
            include.mkdir()
            (root / "__init__.py").write_text("", encoding="utf-8")
            (capi / "libonnxruntime.so.1").write_bytes(b"native ort")
            (include / "onnxruntime_c_api.h").write_text("/* c api */", encoding="utf-8")

            fake_ort = types.SimpleNamespace(__file__=str(root / "__init__.py"))
            evidence = onnxruntime_native_evidence(fake_ort)

        self.assertEqual(evidence["runtime_engine"], "onnxruntime_cxx")
        self.assertEqual(evidence["runtime_execution_language"], "C++")
        self.assertEqual(evidence["runtime_api_binding"], "Python")
        self.assertTrue(evidence["native_inference_engine"])
        self.assertTrue(evidence["native_headers_available"])
        self.assertTrue(evidence["native_cxx_api_buildable"])
        self.assertTrue(evidence["native_library_path"].endswith("libonnxruntime.so.1"))


if __name__ == "__main__":
    unittest.main()
