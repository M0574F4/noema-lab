from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from noema_lab.core.artifacts import Artifact
from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.ops.channel.digital import DigitalDemodulateOperation


def _write_npz_artifact(
    path: Path,
    kind: str,
    metadata: dict,
    **arrays: np.ndarray,
) -> Artifact:
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    return Artifact(kind, path, dict(metadata))


def _current_csi_inputs(
    root: Path,
    *,
    rx_symbols: np.ndarray,
    h_freq: np.ndarray,
    power: np.ndarray,
    noise_variance: float = 0.5,
    bit_count: int | None = None,
    rx_metadata_overrides: dict | None = None,
    state_metadata_overrides: dict | None = None,
    allocation_metadata_overrides: dict | None = None,
    allocation_arrays: dict | None = None,
) -> dict[str, Artifact]:
    rx = np.asarray(rx_symbols, dtype=np.complex64).reshape(-1)
    state = np.asarray(h_freq, dtype=np.complex64)
    allocation = np.asarray(power, dtype=np.float32)
    if state.ndim != 3:
        raise ValueError("test h_freq must be three-dimensional")
    capacity = int(state.size)
    rx_metadata = {
        "modulation": "qpsk",
        "bit_count": int(bit_count if bit_count is not None else 2 * rx.size),
        "payload_symbol_count": int(rx.size),
        "channel_use_count": capacity,
        "grid_padding_symbol_count": capacity - int(rx.size),
        "channel_equalized": True,
        "equalizer": "perfect_csi_ofdm_zero_forcing_one_tap",
        "channel_state_shared": True,
        "channel_state_seed": 17,
        "noise_variance": noise_variance,
    }
    rx_metadata.update(rx_metadata_overrides or {})
    state_metadata = {
        "csi_role": "actual_current_channel_state",
        "channel_application_state": True,
        "transmitter_visible": False,
        "channel_state_seed": 17,
        "noise_variance": noise_variance,
    }
    state_metadata.update(state_metadata_overrides or {})
    allocation_metadata = {
        "transport_mode": "fixed_modulation",
        "noise_variance": noise_variance,
    }
    allocation_metadata.update(allocation_metadata_overrides or {})
    allocation_payload = {"power": allocation}
    allocation_payload.update(allocation_arrays or {})
    return {
        "rx_symbols": _write_npz_artifact(
            root / "rx_symbols.npz",
            "channel.rx_symbols.complex_numpy",
            rx_metadata,
            symbols=rx,
        ),
        "allocation": _write_npz_artifact(
            root / "allocation.npz",
            "channel.power_allocation.numpy",
            allocation_metadata,
            **allocation_payload,
        ),
        "channel_state": _write_npz_artifact(
            root / "channel_state.npz",
            "channel.ofdm_channel_state.numpy",
            state_metadata,
            h_freq=state,
            gains=np.square(np.abs(state)).reshape(-1, state.shape[-1]),
        ),
    }


def _run_demodulator(
    root: Path,
    inputs: dict[str, Artifact],
    *,
    modulation: str = "qpsk",
):
    return DigitalDemodulateOperation().run(
        OperationContext(
            recipe_name="ofdm_current_csi_soft_demodulator_test",
            step_id="demodulator",
            params={
                "modulation": modulation,
                "data_plane_backend": "python_numpy",
            },
            inputs=inputs,
            run_dir=root,
            step_dir=root / "demodulator",
        )
    )


class OfdmCurrentCsiSoftDemodulatorTests(unittest.TestCase):
    def test_formula_sign_bit_order_scaling_and_zero_power(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rx = np.asarray(
                [0.25 - 0.5j, -0.3 + 0.1j, 100.0 - 100.0j, 0.2 + 0.4j],
                dtype=np.complex64,
            )
            h_freq = np.asarray(
                [[[1.0 + 0.0j, 0.5 + 0.5j, 0.0 + 2.0j, 1.0 - 1.0j]]],
                dtype=np.complex64,
            )
            power = np.asarray([[1.0, 4.0, 0.0, 0.25]], dtype=np.float32)
            noise_variance = 0.5
            result = _run_demodulator(
                root,
                _current_csi_inputs(
                    root,
                    rx_symbols=rx,
                    h_freq=h_freq,
                    power=power,
                    noise_variance=noise_variance,
                ),
            )

            h_flat = h_freq.reshape(-1).astype(np.complex128)
            scale = (
                2.0
                * np.sqrt(2.0 * power.reshape(-1))
                * np.square(np.abs(h_flat))
                / noise_variance
            )
            expected_pairs = np.stack(
                [scale * rx.real, scale * rx.imag], axis=-1
            )
            expected_pairs[power.reshape(-1) == 0.0] = 0.0
            expected_llr = expected_pairs.reshape(-1).astype(np.float32)
            expected_bits = (expected_llr < 0.0).astype(np.uint8)
            with np.load(result.outputs["llr"].path, allow_pickle=False) as payload:
                np.testing.assert_allclose(
                    payload["llr"], expected_llr, rtol=1e-6, atol=1e-7
                )
            with np.load(result.outputs["bits"].path, allow_pickle=False) as payload:
                np.testing.assert_array_equal(payload["bits"], expected_bits)

            self.assertEqual(expected_llr[4:6].tolist(), [0.0, 0.0])
            metadata = result.outputs["llr"].metadata
            self.assertEqual(
                metadata["llr_kind"], "max_log_qpsk_post_zf_current_csi"
            )
            self.assertEqual(metadata["llr_positive_value_bit"], 0)
            self.assertEqual(
                metadata["receiver_csi_role"], "actual_current_channel_state"
            )
            self.assertIn("sqrt(2*p(k))", metadata["llr_formula"])
            self.assertEqual(result.metrics["channel.current_csi_soft_demodulation"], 1)

    def test_source_item_qpsk_padding_is_removed_before_concatenation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rx = np.asarray(
                [1.0 - 1.0j, 1.0 - 1.0j, -1.0 + 1.0j, -1.0 + 1.0j],
                dtype=np.complex64,
            )
            inputs = _current_csi_inputs(
                root,
                rx_symbols=rx,
                h_freq=np.ones((1, 1, 4), dtype=np.complex64),
                power=np.ones((1, 4), dtype=np.float32),
                bit_count=7,
                rx_metadata_overrides={
                    "source_item_symbol_counts": [2, 2],
                    "source_item_modulator_input_bit_counts": [3, 4],
                    "source_item_padded_bit_counts": [4, 4],
                },
            )
            result = _run_demodulator(root, inputs)
            with np.load(result.outputs["bits"].path, allow_pickle=False) as payload:
                np.testing.assert_array_equal(
                    payload["bits"],
                    np.asarray([0, 1, 0, 1, 0, 1, 0], dtype=np.uint8),
                )
            self.assertEqual(
                result.outputs["bits"].metadata["source_item_symbol_counts"],
                [2, 2],
            )
            self.assertEqual(result.outputs["bits"].metadata["bit_count"], 7)

    def test_uniform_capture_layout_is_rewritten_to_trimmed_bits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _current_csi_inputs(
                root,
                rx_symbols=np.ones((4,), dtype=np.complex64),
                h_freq=np.ones((1, 1, 4), dtype=np.complex64),
                power=np.ones((1, 4), dtype=np.float32),
                bit_count=6,
                rx_metadata_overrides={
                    "capture_record_count": 2,
                    "capture_record_shape": [2],
                    "source_item_symbol_counts": [2, 2],
                    "source_item_modulator_input_bit_counts": [3, 3],
                    "source_item_padded_bit_counts": [4, 4],
                },
            )
            result = _run_demodulator(root, inputs)
            self.assertEqual(
                result.outputs["bits"].metadata["capture_record_shape"], [3]
            )
            self.assertEqual(
                result.outputs["bits"].metadata["capture_record_count"], 2
            )

    def test_rejects_missing_allocation_for_current_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _current_csi_inputs(
                root,
                rx_symbols=np.ones((2,), dtype=np.complex64),
                h_freq=np.ones((1, 1, 2), dtype=np.complex64),
                power=np.ones((1, 2), dtype=np.float32),
            )
            del inputs["allocation"]
            with self.assertRaisesRegex(OperationError, "requires allocation"):
                _run_demodulator(root, inputs)

    def test_rejects_non_qpsk_or_adaptive_transport(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _current_csi_inputs(
                root,
                rx_symbols=np.ones((2,), dtype=np.complex64),
                h_freq=np.ones((1, 1, 2), dtype=np.complex64),
                power=np.ones((1, 2), dtype=np.float32),
            )
            with self.assertRaisesRegex(OperationError, "modulation=qpsk"):
                _run_demodulator(root, inputs, modulation="bpsk")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _current_csi_inputs(
                root,
                rx_symbols=np.ones((2,), dtype=np.complex64),
                h_freq=np.ones((1, 1, 2), dtype=np.complex64),
                power=np.ones((1, 2), dtype=np.float32),
                allocation_arrays={
                    "modulation_order_bits_per_symbol": np.full(
                        (1, 2), 2, dtype=np.uint8
                    )
                },
            )
            with self.assertRaisesRegex(OperationError, "fixed QPSK transport"):
                _run_demodulator(root, inputs)

    def test_rejects_delayed_csi_role_and_non_zf_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _current_csi_inputs(
                root,
                rx_symbols=np.ones((2,), dtype=np.complex64),
                h_freq=np.ones((1, 1, 2), dtype=np.complex64),
                power=np.ones((1, 2), dtype=np.float32),
                state_metadata_overrides={
                    "csi_role": "delayed_noisy_transmitter_observation",
                    "channel_application_state": False,
                    "transmitter_visible": True,
                },
            )
            with self.assertRaisesRegex(
                OperationError, "actual_current_channel_state"
            ):
                _run_demodulator(root, inputs)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _current_csi_inputs(
                root,
                rx_symbols=np.ones((2,), dtype=np.complex64),
                h_freq=np.ones((1, 1, 2), dtype=np.complex64),
                power=np.ones((1, 2), dtype=np.float32),
                rx_metadata_overrides={"channel_equalized": False},
            )
            with self.assertRaisesRegex(OperationError, "post-ZF"):
                _run_demodulator(root, inputs)

    def test_rejects_state_allocation_shape_or_noise_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _current_csi_inputs(
                root,
                rx_symbols=np.ones((4,), dtype=np.complex64),
                h_freq=np.ones((1, 1, 4), dtype=np.complex64),
                power=np.ones((2, 2), dtype=np.float32),
            )
            with self.assertRaisesRegex(OperationError, "shape mismatch"):
                _run_demodulator(root, inputs)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _current_csi_inputs(
                root,
                rx_symbols=np.ones((2,), dtype=np.complex64),
                h_freq=np.ones((1, 1, 2), dtype=np.complex64),
                power=np.ones((1, 2), dtype=np.float32),
                allocation_metadata_overrides={"noise_variance": 0.25},
            )
            with self.assertRaisesRegex(OperationError, "contradicts"):
                _run_demodulator(root, inputs)


if __name__ == "__main__":
    unittest.main()
