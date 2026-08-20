from __future__ import annotations

import unittest

from noema_lab.core.verification import _CheckRecorder, _check_rate_accounting


class VerificationTaskRateAccountingTests(unittest.TestCase):
    def test_task_direct_recipe_does_not_emit_irrelevant_bpp_warning(self):
        recorder = _CheckRecorder()
        summary = {
            "recipe": {
                "metadata": {"research": {"task": {"id": "aoa_estimation"}}},
                "steps": [
                    {"id": "data", "op": "source.aoa_scene"},
                    {"id": "evaluation", "op": "metrics.aoa_estimation"},
                ],
            }
        }

        _check_rate_accounting(summary, {"aoa.rmse_deg": 0.2}, recorder)

        self.assertEqual(len(recorder.checks), 1)
        self.assertEqual(recorder.checks[0].status, "pass")
        self.assertIn("not applicable", recorder.checks[0].message)

    def test_image_recipe_still_requires_rate_accounting_evidence(self):
        recorder = _CheckRecorder()
        summary = {
            "recipe": {
                "metadata": {"research": {"task": {"id": "image_reconstruction"}}},
                "steps": [{"id": "data", "op": "source.image_dataset"}],
            }
        }

        _check_rate_accounting(summary, {"quality.psnr_db": 30.0}, recorder)

        self.assertEqual(len(recorder.checks), 1)
        self.assertEqual(recorder.checks[0].status, "warning")
        self.assertIn("recompute bpp", recorder.checks[0].message)

    def test_non_image_data_tensor_shape_is_not_misclassified_as_pixels(self):
        recorder = _CheckRecorder()
        summary = {
            "recipe": {
                "metadata": {
                    "research": {
                        "task": {
                            "id": "pilot_channel_estimation",
                            "modality": "wireless",
                        }
                    }
                },
                "steps": [
                    {"id": "data", "op": "source.ai_phy_channel_realization"},
                    {"id": "evaluation", "op": "metrics.channel_estimation"},
                ],
            },
            "steps": [
                {
                    "id": "data",
                    "outputs": {
                        "channel": {
                            "metadata": {
                                "shape": [32, 1, 1],
                            }
                        }
                    },
                }
            ],
        }

        _check_rate_accounting(
            summary,
            {"channel_estimation.nmse": 0.1},
            recorder,
        )

        self.assertEqual(len(recorder.checks), 1)
        self.assertEqual(recorder.checks[0].status, "pass")
        self.assertIn("not applicable", recorder.checks[0].message)

    def test_explicit_rate_boundaries_are_checked_independently(self):
        recorder = _CheckRecorder()
        metrics = {
            "source_pixel_count": 100,
            "codec.native_bit_count": 100,
            "rate.native_codec_bpp": 1.0,
            "codec.serialized_payload_bit_count": 120,
            "rate.serialized_payload_bpp": 1.2,
            "channel.framed_bit_count": 140,
            "rate.framed_bpp": 1.4,
            "channel.coded_bit_count": 280,
            "rate.coded_bpp": 2.8,
            "channel.transmitted_bit_count": 282,
            "rate.padded_bpp": 2.82,
        }

        _check_rate_accounting({}, metrics, recorder)

        self.assertEqual(len(recorder.checks), 5)
        self.assertTrue(all(check.status == "pass" for check in recorder.checks))
        self.assertEqual(
            {check.id for check in recorder.checks},
            {
                "accounting.native_codec_bpp",
                "accounting.serialized_payload_bpp",
                "accounting.framed_bpp",
                "accounting.coded_bpp",
                "accounting.transmitted_padded_bpp",
            },
        )

    def test_explicit_rate_boundary_tamper_is_rejected_at_that_boundary(self):
        recorder = _CheckRecorder()
        metrics = {
            "source_pixel_count": 100,
            "codec.native_bit_count": 100,
            "rate.native_codec_bpp": 1.0,
            "codec.serialized_payload_bit_count": 120,
            "rate.serialized_payload_bpp": 1.2,
            "channel.framed_bit_count": 140,
            "rate.framed_bpp": 1.4,
            "channel.coded_bit_count": 280,
            "rate.coded_bpp": 2.7,
            "channel.transmitted_bit_count": 282,
            "rate.padded_bpp": 2.82,
        }

        _check_rate_accounting({}, metrics, recorder)

        failures = [check for check in recorder.checks if check.status == "error"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].id, "accounting.coded_bpp")

    def test_named_boundaries_do_not_mask_forged_legacy_rate_bpp(self):
        recorder = _CheckRecorder()
        metrics = {
            "source_pixel_count": 100,
            "codec.native_bit_count": 100,
            "rate.native_codec_bpp": 1.0,
            "codec.serialized_payload_bit_count": 120,
            "rate.serialized_payload_bpp": 1.2,
            "channel.transmitted_bit_count": 282,
            "rate.padded_bpp": 2.82,
            "rate_bpp": 1.0,
        }

        _check_rate_accounting({}, metrics, recorder)

        failures = [check for check in recorder.checks if check.status == "error"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].id, "accounting.rate_bpp")

    def test_present_string_rate_metric_is_invalid_not_missing(self):
        recorder = _CheckRecorder()
        metrics = {
            "source_pixel_count": 100,
            "channel.transmitted_bit_count": "282",
            "rate.padded_bpp": 2.82,
        }

        _check_rate_accounting({}, metrics, recorder)

        self.assertEqual(len(recorder.checks), 1)
        self.assertEqual(recorder.checks[0].id, "accounting.metric_values")
        self.assertEqual(recorder.checks[0].status, "error")
        self.assertIn("channel.transmitted_bit_count", recorder.checks[0].message)


if __name__ == "__main__":
    unittest.main()
