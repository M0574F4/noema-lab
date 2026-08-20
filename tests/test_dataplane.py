import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.core import dataplane


class DataPlaneAutoPolicyTests(unittest.TestCase):
    def test_auto_is_python_numpy_even_when_native_module_is_available(self):
        bits = np.array([0, 1, 1, 0], dtype=np.uint8)
        with (
            patch.object(dataplane, "native_available", return_value=True),
            patch.object(
                dataplane,
                "require_native",
                side_effect=AssertionError(
                    "auto must not probe or invoke the native extension"
                ),
            ),
        ):
            self.assertEqual(
                dataplane.selected_backend(
                    {"data_plane_backend": "auto"},
                    "qpsk_modulate",
                    prefer_cpp=True,
                ),
                "python_numpy",
            )
            symbols, _padded, selected = dataplane.qpsk_modulate(bits, "auto")
            self.assertEqual(selected, "python_numpy")
            _decoded, selected = dataplane.qpsk_demodulate(symbols, "auto")
            self.assertEqual(selected, "python_numpy")
            _errors, selected = dataplane.bit_error_count(bits, bits, "auto")
            self.assertEqual(selected, "python_numpy")

    def test_kernel_calls_reject_unknown_backend_instead_of_using_numpy(self):
        with self.assertRaisesRegex(
            dataplane.OperationError,
            "Unknown data_plane_backend",
        ):
            dataplane.qpsk_modulate(
                np.array([0, 1], dtype=np.uint8),
                "host_fastest",
            )


class DataPlaneTests(unittest.TestCase):
    def setUp(self):
        if not dataplane.native_available():
            self.skipTest("native dataplane extension is not built")

    def test_indices_bits_roundtrip_matches_numpy(self):
        indices = np.array([[0, 1, 2, 7], [4, 5, 6, 3]], dtype=np.int64)
        py_bits, py_backend = dataplane.indices_to_bits(indices, 3, "python_numpy")
        cpp_bits, cpp_backend = dataplane.indices_to_bits(indices, 3, "cpp_native")
        self.assertEqual(py_backend, "python_numpy")
        self.assertEqual(cpp_backend, "cpp_native")
        np.testing.assert_array_equal(cpp_bits, py_bits)

        py_decoded, py_invalid, _ = dataplane.bits_to_indices(py_bits, 3, indices.shape, 8, "mod", "python_numpy")
        cpp_decoded, cpp_invalid, _ = dataplane.bits_to_indices(cpp_bits, 3, indices.shape, 8, "mod", "cpp_native")
        np.testing.assert_array_equal(cpp_decoded, py_decoded)
        np.testing.assert_array_equal(cpp_decoded, indices)
        self.assertEqual(cpp_invalid, py_invalid)

    def test_channel_kernels_match_numpy(self):
        bits = np.array([0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
        py_coded, _ = dataplane.repetition_encode(bits, 3, "python_numpy")
        cpp_coded, cpp_backend = dataplane.repetition_encode(bits, 3, "cpp_native")
        self.assertEqual(cpp_backend, "cpp_native")
        np.testing.assert_array_equal(cpp_coded, py_coded)

        py_decoded, _ = dataplane.repetition_decode(py_coded, 3, bits.size, "python_numpy")
        cpp_decoded, _ = dataplane.repetition_decode(cpp_coded, 3, bits.size, "cpp_native")
        np.testing.assert_array_equal(cpp_decoded, py_decoded)
        np.testing.assert_array_equal(cpp_decoded, bits)

        py_errors, _ = dataplane.bit_error_count(bits, 1 - bits, "python_numpy")
        cpp_errors, _ = dataplane.bit_error_count(bits, 1 - bits, "cpp_native")
        self.assertEqual(cpp_errors, py_errors)

    def test_modulation_and_awgn_match_numpy(self):
        bits = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        py_symbols, py_padded, _ = dataplane.qpsk_modulate(bits, "python_numpy")
        cpp_symbols, cpp_padded, cpp_backend = dataplane.qpsk_modulate(bits, "cpp_native")
        self.assertEqual(cpp_backend, "cpp_native")
        np.testing.assert_array_equal(cpp_padded, py_padded)
        np.testing.assert_allclose(cpp_symbols, py_symbols)

        noise_real = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        noise_imag = np.array([-0.4, 0.5, -0.6], dtype=np.float32)
        py_noisy, _ = dataplane.awgn_apply(py_symbols, noise_real, noise_imag, 0.25, "python_numpy")
        cpp_noisy, _ = dataplane.awgn_apply(cpp_symbols, noise_real, noise_imag, 0.25, "cpp_native")
        np.testing.assert_allclose(cpp_noisy, py_noisy)

        py_demod, _ = dataplane.qpsk_demodulate(py_noisy, "python_numpy")
        cpp_demod, _ = dataplane.qpsk_demodulate(cpp_noisy, "cpp_native")
        np.testing.assert_array_equal(cpp_demod, py_demod)

    def test_payload_and_image_conversion_roundtrips(self):
        payload = b"semantic-bytes"
        bits, backend = dataplane.bytes_to_bits(payload, "cpp_native")
        self.assertEqual(backend, "cpp_native")
        decoded, backend = dataplane.bits_to_bytes(bits, len(payload), "cpp_native")
        self.assertEqual(backend, "cpp_native")
        self.assertEqual(decoded, payload)

        images = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
        nchw, backend = dataplane.image_to_nchw(images, "cpp_native")
        self.assertEqual(backend, "cpp_native")
        restored, backend = dataplane.nchw_to_image(nchw, "cpp_native")
        self.assertEqual(backend, "cpp_native")
        np.testing.assert_array_equal(restored, images)

        values = np.array([0.0, 1.25, -3.5], dtype=np.float32)
        value_bits, backend = dataplane.float32_to_bits(values, "cpp_native")
        self.assertEqual(backend, "cpp_native")
        restored_values, backend = dataplane.bits_to_float32(value_bits, values.shape, "cpp_native")
        self.assertEqual(backend, "cpp_native")
        np.testing.assert_array_equal(restored_values, values)


if __name__ == "__main__":
    unittest.main()
