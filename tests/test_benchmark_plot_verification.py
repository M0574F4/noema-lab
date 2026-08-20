import copy
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from noema_lab.core.artifacts import file_sha256
from noema_lab.core.benchmark_plots import (
    _benchmark_plot_renderer_identity,
    _benchmark_plot_renderer_matches,
    _plot_style,
    benchmark_plot_semantic_projection_sha256,
    plot_benchmark_result,
    reproduce_benchmark_plot_artifacts,
)
from noema_lab.core.benchmarks import BenchmarkError, write_benchmark_reports
from noema_lab.core.storage import LocalStore
from noema_lab.core.verification import (
    _CheckRecorder,
    _check_benchmark_expected_outputs,
    _check_benchmark_plot_artifacts,
    _check_benchmark_plot_sidecars,
    _check_benchmark_report_artifacts,
)


class BenchmarkPlotVerificationTests(unittest.TestCase):
    def test_renderer_identity_tracks_live_rendering_source(self):
        current = _benchmark_plot_renderer_identity()
        self.assertEqual(current["schema_version"], 3)
        self.assertRegex(current["implementation_sha256"], r"^[0-9a-f]{64}$")

        legacy = copy.deepcopy(current)
        legacy["schema_version"] = 2
        legacy["implementation_sha256"] = (
            "1c1fcd2ad112fcd0388a0bb98196067de77e1139ffc1df9d33f99747c4f67126"
        )
        self.assertTrue(_benchmark_plot_renderer_matches(legacy, current))

        changed_renderer = copy.deepcopy(current)
        changed_renderer["implementation_sha256"] = "0" * 64
        self.assertFalse(
            _benchmark_plot_renderer_matches(legacy, changed_renderer)
        )
        unknown_legacy = copy.deepcopy(legacy)
        unknown_legacy["implementation_sha256"] = "f" * 64
        self.assertFalse(
            _benchmark_plot_renderer_matches(unknown_legacy, current)
        )
        extra_null_legacy = copy.deepcopy(legacy)
        extra_null_legacy["undeclared"] = None
        self.assertFalse(
            _benchmark_plot_renderer_matches(extra_null_legacy, current)
        )

    def test_incomplete_or_skipped_comparison_cannot_be_plotted(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(Path(tmp) / ".noema")
            result_id = "incomplete-comparison"
            result_dir = store.get_benchmark_result_dir(result_id)
            result_dir.mkdir(parents=True)
            store.write_json(
                result_dir / "result.json",
                {
                    "status": "incomplete",
                    "benchmark": {"id": "fixture", "version": "1"},
                    "recipes": [
                        {
                            "id": "baseline",
                            "status": "completed",
                            "metrics": {
                                "channel.snr_db": 0.0,
                                "quality.psnr_db": 20.0,
                            },
                        },
                        {
                            "id": "candidate",
                            "status": "skipped",
                            "metrics": {},
                        },
                    ],
                },
            )

            with self.assertRaisesRegex(BenchmarkError, "status is incomplete"):
                plot_benchmark_result(
                    store,
                    result_id,
                    "graceful-degradation",
                    Path("plots/curve.png"),
                )

            self.assertFalse((result_dir / "plots" / "curve.png").exists())

    def test_plot_command_emits_a_semantically_reproducible_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(Path(tmp) / ".noema")
            store.ensure()
            result_id = "plot-command-fixture"
            result_dir = store.get_benchmark_result_dir(result_id)
            result_dir.mkdir(parents=True)
            result = {
                "status": "completed",
                "benchmark": {"id": "plot-fixture", "version": "1"},
                "recipes": [
                    {
                        "id": "candidate-one",
                        "label": "candidate",
                        "role": "candidate",
                        "run_id": "run-candidate-one",
                        "status": "completed",
                        "metrics": {
                            "channel.snr_db": 0.0,
                            "quality.psnr_db": 20.0,
                        },
                    }
                ],
            }
            store.write_json(result_dir / "result.json", result)
            sealed_sha256 = file_sha256(result_dir / "result.json")
            payload = plot_benchmark_result(
                store,
                result_id,
                "graceful-degradation",
                Path("plots/curve.png"),
            )
            stored = store.get_benchmark_result(result_id)
            self.assertEqual(
                file_sha256(result_dir / "result.json"),
                sealed_sha256,
            )
            self.assertNotIn("plots", stored)
            report = self._report(result_dir, stored)

        self.assertEqual(report["status"], "valid", report)
        self.assertEqual(
            payload["plot"]["renderer"]["id"],
            "noema.matplotlib_benchmark_plot",
        )
        self.assertEqual(
            payload["plot"]["selection_status"], "exploratory_post_hoc"
        )
        self.assertIsNone(payload["plot"]["selection_protocol"])
        self.assertRegex(
            payload["plot"]["semantic_projection_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(
            payload["plot"]["semantic_projection_sha256"],
            benchmark_plot_semantic_projection_sha256(
                stored,
                payload["plot"],
            ),
        )
        self.assertTrue(
            any(
                check["id"] == "benchmark.plot.semantic_projection"
                and check["status"] == "pass"
                for check in report["checks"]
            ),
            report,
        )

    def test_exact_image_and_csv_bindings_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir, result = self._fixture(Path(tmp))
            report = self._report(result_dir, result)

        self.assertEqual(report["status"], "valid", report)

    def test_legacy_plot_without_semantic_projection_identity_still_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir, result = self._fixture(Path(tmp))
            result["plots"][0].pop("semantic_projection_sha256")
            report = self._report(result_dir, result)

        self.assertEqual(report["status"], "valid", report)

    def test_semantic_projection_rejects_selection_tampering_with_same_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir, result = self._fixture(Path(tmp))
            plot = result["plots"][0]
            original_image_sha = plot["sha256"]
            original_data_sha = plot["data_csv_sha256"]
            plot["requested_x_metric"] = plot["x_metric"]
            report = self._report(result_dir, result)

        self.assertEqual(plot["sha256"], original_image_sha)
        self.assertEqual(plot["data_csv_sha256"], original_data_sha)
        self.assertEqual(report["status"], "invalid", report)
        self.assertTrue(
            any("semantic projection identity differs" in item for item in report["errors"]),
            report,
        )

    def test_semantic_projection_rejects_row_tampering_with_recomputed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir, result = self._fixture(Path(tmp))
            plot = result["plots"][0]
            stale_semantic_sha = plot.pop("semantic_projection_sha256")
            old_data_sha = plot["data_csv_sha256"]
            result["recipes"][0]["metrics"]["quality.psnr_db"] = 21.0
            image_path = result_dir / plot["relative_path"]
            data_path = result_dir / plot["data_csv_relative_path"]
            reproduce_benchmark_plot_artifacts(
                result,
                plot,
                image_path=image_path,
                data_path=data_path,
            )
            plot["semantic_projection_sha256"] = stale_semantic_sha
            for role, path, sha_key, size_key in (
                ("image", image_path, "sha256", "size_bytes"),
                (
                    "data_csv",
                    data_path,
                    "data_csv_sha256",
                    "data_csv_size_bytes",
                ),
            ):
                digest = file_sha256(path)
                size = path.stat().st_size
                plot[sha_key] = digest
                plot[size_key] = size
                plot["artifacts"][role]["sha256"] = digest
                plot["artifacts"][role]["size_bytes"] = size
            report = self._report(result_dir, result)

        self.assertNotEqual(plot["data_csv_sha256"], old_data_sha)
        self.assertEqual(report["status"], "invalid", report)
        self.assertTrue(
            any("semantic projection identity differs" in item for item in report["errors"]),
            report,
        )

    def test_semantic_projection_identity_is_separate_from_representation_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _result_dir, result = self._fixture(Path(tmp))
            plot = result["plots"][0]
            semantic_sha = benchmark_plot_semantic_projection_sha256(result, plot)
            representation_variant = copy.deepcopy(plot)
            representation_variant["renderer"]["matplotlib_version"] = "other"
            representation_variant["style_options"]["dpi"] = 72
            representation_variant["format"] = "pdf"
            representation_variant["relative_path"] = "plots/other.pdf"
            representation_variant["sha256"] = "0" * 64
            representation_variant["artifacts"]["image"].update(
                {
                    "relative_path": "plots/other.pdf",
                    "sha256": "0" * 64,
                    "size_bytes": 1,
                }
            )
            outage_variant = copy.deepcopy(plot)
            outage_variant["outage_markers"] = not plot["outage_markers"]
            packet_panel_variant = copy.deepcopy(plot)
            packet_panel_variant["packet_success_panel"] = not plot[
                "packet_success_panel"
            ]

        self.assertEqual(plot["semantic_projection_sha256"], semantic_sha)
        self.assertEqual(
            benchmark_plot_semantic_projection_sha256(
                result,
                representation_variant,
            ),
            semantic_sha,
        )
        self.assertNotEqual(semantic_sha, plot["sha256"])
        self.assertNotEqual(semantic_sha, plot["data_csv_sha256"])
        self.assertNotEqual(
            benchmark_plot_semantic_projection_sha256(result, outage_variant),
            semantic_sha,
        )
        self.assertNotEqual(
            benchmark_plot_semantic_projection_sha256(
                result,
                packet_panel_variant,
            ),
            semantic_sha,
        )

    def test_changed_plot_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir, result = self._fixture(Path(tmp))
            (result_dir / "plots" / "curve.png").write_bytes(b"substituted")
            report = self._report(result_dir, result)

        self.assertEqual(report["status"], "invalid", report)
        self.assertTrue(any("hash mismatch" in item for item in report["errors"]), report)

    def test_unsafe_absolute_plot_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir, result = self._fixture(Path(tmp))
            outside = Path(tmp) / "outside.png"
            outside.write_bytes(b"same-image")
            plot = result["plots"][0]
            plot["relative_path"] = str(outside)
            plot["artifacts"]["image"]["relative_path"] = str(outside)
            report = self._report(result_dir, result)

        self.assertEqual(report["status"], "invalid", report)
        self.assertTrue(any("safe bundle-relative" in item for item in report["errors"]), report)

    def test_csv_point_count_and_metric_projection_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir, result = self._fixture(Path(tmp))
            result["plots"][0]["point_count"] = 2
            result["plots"][0]["x_metric"] = "different.metric"
            report = self._report(result_dir, result)

        self.assertEqual(report["status"], "invalid", report)
        self.assertTrue(any("point_count" in item for item in report["errors"]), report)
        self.assertTrue(any("x_metric" in item for item in report["errors"]), report)

    def test_forged_csv_values_are_rejected_even_when_rehashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir, result = self._fixture(Path(tmp))
            data_path = result_dir / "plots" / "curve.csv"
            rows = list(csv.DictReader(data_path.read_text(encoding="utf-8").splitlines()))
            rows[0]["y_value"] = "999.0"
            with data_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            plot = result["plots"][0]
            digest = file_sha256(data_path)
            size = data_path.stat().st_size
            plot["data_csv_sha256"] = digest
            plot["data_csv_size_bytes"] = size
            plot["artifacts"]["data_csv"]["sha256"] = digest
            plot["artifacts"]["data_csv"]["size_bytes"] = size
            report = self._report(result_dir, result)

        self.assertEqual(report["status"], "invalid", report)
        self.assertTrue(any("reproduced" in item for item in report["errors"]), report)

    def test_substituted_plot_image_is_rejected_even_when_rehashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir, result = self._fixture(Path(tmp))
            forged_result = copy.deepcopy(result)
            forged_result["recipes"][0]["metrics"]["quality.psnr_db"] = 21.0
            forged_image = result_dir / "plots" / "forged.png"
            forged_data = result_dir / "plots" / "forged.csv"
            forged_plot = copy.deepcopy(result["plots"][0])
            forged_plot.pop("semantic_projection_sha256")
            reproduce_benchmark_plot_artifacts(
                forged_result,
                forged_plot,
                image_path=forged_image,
                data_path=forged_data,
            )
            image_path = result_dir / "plots" / "curve.png"
            image_path.write_bytes(forged_image.read_bytes())
            plot = result["plots"][0]
            digest = file_sha256(image_path)
            size = image_path.stat().st_size
            plot["sha256"] = digest
            plot["size_bytes"] = size
            plot["artifacts"]["image"]["sha256"] = digest
            plot["artifacts"]["image"]["size_bytes"] = size
            report = self._report(result_dir, result)

        self.assertEqual(report["status"], "invalid", report)
        self.assertTrue(any("rendered image" in item for item in report["errors"]), report)

    def test_plot_reproduction_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir, result = self._fixture(Path(tmp))
            first_image = result_dir / "plots" / "rebuild-one.png"
            first_data = result_dir / "plots" / "rebuild-one.csv"
            second_image = result_dir / "plots" / "rebuild-two.png"
            second_data = result_dir / "plots" / "rebuild-two.csv"
            reproduce_benchmark_plot_artifacts(
                result,
                result["plots"][0],
                image_path=first_image,
                data_path=first_data,
            )
            reproduce_benchmark_plot_artifacts(
                result,
                result["plots"][0],
                image_path=second_image,
                data_path=second_data,
            )
            first_image_sha = file_sha256(first_image)
            second_image_sha = file_sha256(second_image)
            first_data_sha = file_sha256(first_data)
            second_data_sha = file_sha256(second_data)

        self.assertEqual(first_image_sha, second_image_sha)
        self.assertEqual(first_data_sha, second_data_sha)

    def test_renderer_environment_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir, result = self._fixture(Path(tmp))
            result["plots"][0]["renderer"]["matplotlib_version"] = "different"
            report = self._report(result_dir, result)

        self.assertEqual(report["status"], "invalid", report)
        self.assertTrue(any("renderer" in item for item in report["errors"]), report)

    def test_plot_output_must_be_safe_supported_and_bundle_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalStore(root / ".noema")
            store.ensure()
            result_id = "unsafe-output-fixture"
            result_dir = store.get_benchmark_result_dir(result_id)
            result_dir.mkdir(parents=True)
            store.write_json(
                result_dir / "result.json",
                {
                    "status": "completed",
                    "benchmark": {"id": "fixture", "version": "1"},
                    "recipes": [
                        {
                            "id": "candidate",
                            "label": "candidate",
                            "role": "candidate",
                            "run_id": "run",
                            "status": "completed",
                            "metrics": {
                                "channel.snr_db": 0.0,
                                "quality.psnr_db": 20.0,
                            },
                        }
                    ],
                },
            )
            unsafe = [
                root / "outside.png",
                Path("../outside.png"),
                Path("plots/no-suffix"),
                Path("plots/collision.csv"),
                Path("plots/unsupported.jpg"),
            ]
            for output in unsafe:
                with self.subTest(output=output):
                    with self.assertRaises(BenchmarkError):
                        plot_benchmark_result(
                            store,
                            result_id,
                            "graceful-degradation",
                            output,
                        )
            outside_dir = root / "outside-directory"
            outside_dir.mkdir()
            (result_dir / "linked-output").symlink_to(
                outside_dir, target_is_directory=True
            )
            with self.assertRaisesRegex(BenchmarkError, "symbolic link"):
                plot_benchmark_result(
                    store,
                    result_id,
                    "graceful-degradation",
                    Path("linked-output/curve.png"),
                )
            self.assertFalse((root / "outside.png").exists())
            self.assertFalse((outside_dir / "curve.png").exists())

    def test_duplicate_method_order_and_cross_plot_path_reuse_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(Path(tmp) / ".noema")
            store.ensure()
            result_id = "order-fixture"
            result_dir = store.get_benchmark_result_dir(result_id)
            result_dir.mkdir(parents=True)
            store.write_json(
                result_dir / "result.json",
                {
                    "status": "completed",
                    "benchmark": {"id": "fixture", "version": "1"},
                    "recipes": [
                        {
                            "id": "a",
                            "label": "A",
                            "role": "candidate",
                            "run_id": "run-a",
                            "status": "completed",
                            "metrics": {
                                "channel.snr_db": 0.0,
                                "quality.psnr_db": 20.0,
                                "channel.packet_success_rate": 1.0,
                            },
                        }
                    ],
                },
            )
            with self.assertRaisesRegex(BenchmarkError, "unique"):
                plot_benchmark_result(
                    store,
                    result_id,
                    "graceful-degradation",
                    Path("plots/curve.png"),
                    method_order="A,A",
                )
            plot_benchmark_result(
                store,
                result_id,
                "graceful-degradation",
                Path("plots/curve.png"),
            )
            with self.assertRaisesRegex(BenchmarkError, "another recorded plot"):
                plot_benchmark_result(
                    store,
                    result_id,
                    "packet-success",
                    Path("plots/curve.png"),
                )

    def test_declared_plot_image_csv_and_sidecar_are_all_mandatory(self):
        missing_paths = (
            "plots/curve.png",
            "plots/curve.csv",
            "plots/curve.png.plot.json",
        )
        for missing_path in missing_paths:
            with self.subTest(missing_path=missing_path):
                with tempfile.TemporaryDirectory() as tmp:
                    store = LocalStore(Path(tmp) / ".noema")
                    store.ensure()
                    result_id = "declared-plot-fixture"
                    result_dir = store.get_benchmark_result_dir(result_id)
                    result_dir.mkdir(parents=True)
                    benchmark = {
                        "id": "plot-fixture",
                        "version": "1",
                        "metadata": {
                            "expected_outputs": [
                                "plots/curve.png",
                                "plots/curve.csv",
                            ]
                        },
                    }
                    result = {
                        "status": "completed",
                        "benchmark": benchmark,
                        "recipes": [
                            {
                                "id": "candidate",
                                "label": "candidate",
                                "role": "candidate",
                                "run_id": "run",
                                "status": "completed",
                                "metrics": {
                                    "channel.snr_db": 0.0,
                                    "quality.psnr_db": 20.0,
                                },
                            }
                        ],
                    }
                    store.write_json(result_dir / "result.json", result)
                    plot_benchmark_result(
                        store,
                        result_id,
                        "graceful-degradation",
                        Path("plots/curve.png"),
                    )
                    (result_dir / missing_path).unlink()
                    recorder = _CheckRecorder()
                    declarations = _check_benchmark_expected_outputs(
                        result_dir,
                        result,
                        benchmark,
                        recorder,
                    )
                    _check_benchmark_plot_sidecars(
                        result_dir,
                        result,
                        recorder,
                        declared_plot_outputs=declarations["plot_artifacts"],
                        required_plot_sidecars=declarations["plot_sidecars"],
                    )
                    report = recorder.report(
                        target_type="benchmark_result",
                        target_id=result_id,
                        path=result_dir,
                    )

                self.assertEqual(report["status"], "invalid", report)
                self.assertTrue(
                    any(
                        "declared output" in message
                        or "declared plot" in message
                        for message in report["errors"]
                    ),
                    report,
                )

    def test_plots_remain_optional_without_a_plot_output_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            result_dir.mkdir()
            result = {
                "benchmark": {
                    "id": "no-plot-fixture",
                    "version": "1",
                    "metadata": {
                        "expected_outputs": [
                            "result.json",
                        ]
                    },
                },
                "recipes": [],
            }
            (result_dir / "result.json").write_text("{}", encoding="utf-8")
            recorder = _CheckRecorder()
            declarations = _check_benchmark_expected_outputs(
                result_dir,
                result,
                None,
                recorder,
            )
            _check_benchmark_plot_sidecars(
                result_dir,
                result,
                recorder,
                declared_plot_outputs=declarations["plot_artifacts"],
                required_plot_sidecars=declarations["plot_sidecars"],
            )
            report = recorder.report(
                target_type="benchmark_result",
                target_id="no-plot-fixture",
                path=result_dir,
            )

        self.assertEqual(report["status"], "valid", report)
        self.assertEqual(declarations["plot_artifacts"], [])
        self.assertEqual(declarations["plot_sidecars"], [])

    def test_rehashed_forged_human_report_is_rejected_by_semantic_reproduction(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            result = {
                "benchmark": {"id": "reports", "version": "1"},
                "recipes": [
                    {
                        "id": "candidate",
                        "label": "candidate",
                        "role": "candidate",
                        "run_id": "run",
                        "status": "completed",
                        "metrics": {"task.score": 1.0},
                    }
                ],
            }
            write_benchmark_reports(result_dir, result)
            metrics_path = result_dir / "metrics.csv"
            metrics_path.write_text(
                metrics_path.read_text(encoding="utf-8").replace("1.0", "999.0"),
                encoding="utf-8",
            )
            descriptor = result["reports"]["metrics_csv"]
            descriptor["sha256"] = file_sha256(metrics_path)
            descriptor["size_bytes"] = metrics_path.stat().st_size
            recorder = _CheckRecorder()
            _check_benchmark_report_artifacts(result_dir, result, recorder)
            report = recorder.report(
                target_type="benchmark_result",
                target_id="reports",
                path=result_dir,
            )

        self.assertEqual(report["status"], "invalid", report)
        self.assertTrue(any("projection" in item for item in report["errors"]), report)

    @staticmethod
    def _report(result_dir, result):
        recorder = _CheckRecorder()
        _check_benchmark_plot_artifacts(result_dir, result, recorder)
        _check_benchmark_plot_sidecars(result_dir, result, recorder)
        return recorder.report(
            target_type="benchmark_result",
            target_id="plot-fixture",
            path=result_dir,
        )

    @staticmethod
    def _fixture(root):
        result_dir = root / "result"
        plot_dir = result_dir / "plots"
        plot_dir.mkdir(parents=True)
        image_path = plot_dir / "curve.png"
        data_path = plot_dir / "curve.csv"
        result = {
            "benchmark": {"id": "plot-fixture", "version": "1"},
            "recipes": [
                {
                    "id": "candidate-one",
                    "label": "candidate",
                    "role": "candidate",
                    "run_id": "run-candidate-one",
                    "status": "completed",
                    "metrics": {
                        "channel.snr_db": 0.0,
                        "quality.psnr_db": 20.0,
                    },
                }
            ],
        }
        plot = {
            "id": "graceful-degradation-curve",
            "plot": "graceful-degradation",
            "relative_path": "plots/curve.png",
            "data_csv_relative_path": "plots/curve.csv",
            "point_count": 1,
            "x_metric": "channel.snr_db",
            "y_metric": "quality.psnr_db",
            "requested_x_metric": None,
            "requested_y_metric": None,
            "group_by": "method",
            "method_order": [],
            "format": "png",
            "style": "noema",
            "style_options": _plot_style("noema", None),
            "renderer": _benchmark_plot_renderer_identity(),
            "selection_status": "exploratory_post_hoc",
            "selection_protocol": None,
            "outage_markers": True,
            "packet_success_panel": False,
        }
        rows = reproduce_benchmark_plot_artifacts(
            result,
            plot,
            image_path=image_path,
            data_path=data_path,
        )
        plot["point_count"] = len(rows)
        plot["semantic_projection_sha256"] = (
            benchmark_plot_semantic_projection_sha256(result, plot)
        )
        image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
        data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
        plot.update(
            {
                "sha256": image_sha,
                "size_bytes": image_path.stat().st_size,
                "data_csv_sha256": data_sha,
                "data_csv_size_bytes": data_path.stat().st_size,
                "artifacts": {
                    "image": {
                        "relative_path": "plots/curve.png",
                        "sha256": image_sha,
                        "size_bytes": image_path.stat().st_size,
                    },
                    "data_csv": {
                        "relative_path": "plots/curve.csv",
                        "sha256": data_sha,
                        "size_bytes": data_path.stat().st_size,
                    },
                },
            }
        )
        result["plots"] = [plot]
        return result_dir, result


if __name__ == "__main__":
    unittest.main()
