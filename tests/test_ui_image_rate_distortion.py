from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"


def _artifact(kind: str, metadata: dict) -> dict:
    return {"kind": kind, "metadata": metadata, "path": "/tmp/test-artifact.npz"}


def _summary(
    run_id: str,
    *,
    psnr_db: float,
    snr_db: float,
    source_metadata: dict,
    channel_uses: int,
    transmitted_bits: int | None = None,
) -> dict:
    recipe_steps = [
        {"id": "image_source", "op": "source.image_dataset", "params": {}},
        {"id": "sender", "op": "model.deepjscc_external_encode", "params": {}},
        {
            "id": "wireless_channel",
            "op": "wireless.channel",
            "params": {"channel": "awgn", "snr_db": snr_db},
        },
        {"id": "evaluation", "op": "metrics.image_reconstruction", "params": {}},
    ]
    steps = [
        {
            "id": "image_source",
            "outputs": {"images": _artifact("image.batch.numpy", source_metadata)},
        },
        {
            "id": "sender",
            "outputs": {
                "symbols": _artifact(
                    "channel.symbols.complex_numpy",
                    {"symbol_count": channel_uses},
                )
            },
        },
    ]
    if transmitted_bits is not None:
        recipe_steps.insert(
            2,
            {"id": "tx_bit_boundary", "op": "channel.bit_boundary", "params": {}},
        )
        steps.append(
            {
                "id": "tx_bit_boundary",
                "outputs": {
                    "bits": _artifact(
                        "channel.bits.numpy",
                        {"fixed_point_bit_count": transmitted_bits},
                    )
                },
            }
        )
    steps.extend(
        [
            {
                "id": "wireless_channel",
                "metrics": {
                    "channel.reference_snr_db": snr_db,
                    "channel.channel_use_count": channel_uses,
                },
                "outputs": {
                    "rx_symbols": _artifact(
                        "channel.rx_symbols.complex_numpy",
                        {"channel_use_count": channel_uses},
                    )
                },
            },
            {
                "id": "evaluation",
                "metrics": {"quality.psnr_db": psnr_db},
                "outputs": {},
            },
        ]
    )
    return {
        "run_id": run_id,
        "status": "completed",
        "recipe": {
            "name": run_id,
            "metadata": {
                "research": {"task": {"id": "image_reconstruction"}},
                "sweep_values": {"wireless_channel.params.snr_db": snr_db},
            },
            "steps": recipe_steps,
        },
        "steps": steps,
    }


@unittest.skipUnless(shutil.which("node"), "Node.js is required for the UI metric regression test")
class UiImageRateDistortionTests(unittest.TestCase):
    def test_continuous_symbols_use_bandwidth_ratio_without_synthetic_bits(self):
        padded_source = {
            "shape": [2, 12, 12, 3],
            "original_shapes": [[1, 10, 8, 3], [1, 10, 8, 3]],
        }
        deep_summaries = [
            _summary(
                "deep-snr-4",
                psnr_db=24.5,
                snr_db=4.0,
                source_metadata=padded_source,
                channel_uses=100,
            ),
            _summary(
                "deep-snr-10",
                psnr_db=29.0,
                snr_db=10.0,
                source_metadata=padded_source,
                channel_uses=100,
            ),
        ]
        digital_summary = _summary(
            "digital-snr-10",
            psnr_db=31.0,
            snr_db=10.0,
            source_metadata={"shape": [1, 10, 10, 3]},
            channel_uses=100,
            transmitted_bits=200,
        )
        rows = [
            {
                "recipe": {
                    "key": "deep",
                    "displayName": "DeepJSCC",
                    "color": "#6aa5ff",
                    "recipe": deep_summaries[0]["recipe"],
                },
                "summaries": deep_summaries,
            },
            {
                "recipe": {
                    "key": "digital",
                    "displayName": "Digital baseline",
                    "color": "#ff9d5c",
                    "recipe": digital_summary["recipe"],
                },
                "summary": digital_summary,
            },
        ]
        script = textwrap.dedent(
            rf"""
            const fs = require("fs");
            const vm = require("vm");
            let source = fs.readFileSync({json.dumps(str(APP_JS))}, "utf8");
            source = source.replace(/\ninit\(\);\s*$/, "\n");
            const sandbox = {{
              console,
              localStorage: {{ getItem: () => null, setItem: () => {{}} }},
              document: {{ documentElement: {{ dataset: {{}} }}, getElementById: () => null }},
              window: {{ CSS: null }},
              setTimeout,
              clearTimeout,
            }};
            sandbox.globalThis = sandbox;
            vm.createContext(sandbox);
            vm.runInContext(source + `
              const rows = ${{JSON.stringify({json.dumps(rows)})}};
              state.resultRows = rows;
              const deepDetails = collectRunDetails(rows[0].summaries[0]);
              const digitalDetails = collectRunDetails(rows[1].summary);
              const presentation = rateDistortionPresentation(rows);
              const digitalPresentation = rateDistortionPresentation([rows[1]]);
              const definitions = scatterMetricDefinitions(rows);
              const settings = normalizedRdSettings(definitions, presentation.settings, presentation.context);
              const labels = recipeDisplayLabels(rows.map((row) => row.recipe));
              const deepPoints = rdChartPoints(rows[0], labels, settings, definitions);
              const digitalPoints = rdChartPoints(rows[1], labels, settings, definitions);
              const xLabel = scatterMetricLabel(definitions, settings.x, settings.xStat);
              const yLabel = scatterMetricLabel(definitions, settings.y, settings.yStat);
              globalThis.__result = {{
                deep: {{
                  pixels: sourcePixelCount(deepDetails),
                  uses: transmittedSymbolCount(deepDetails),
                  bits: transmittedBitCount(deepDetails),
                  ratio: channelUsesPerSourcePixel(deepDetails),
                  pointX: deepPoints.map((point) => point.x),
                  titles: deepPoints.map((point) => rdPointTitle(point, xLabel, yLabel, "fixed point size")),
                }},
                digital: {{
                  pixels: sourcePixelCount(digitalDetails),
                  uses: transmittedSymbolCount(digitalDetails),
                  bits: transmittedBitCount(digitalDetails),
                  ratio: channelUsesPerSourcePixel(digitalDetails),
                  bpp: transmittedBitsPerSourcePixel(digitalDetails),
                  titles: digitalPoints.map((point) => rdPointTitle(point, xLabel, yLabel, "fixed point size")),
                }},
                presentation,
                digitalPresentation,
                markup: overviewPrimaryMetricPlot(rows),
                digitalMarkup: overviewPrimaryMetricPlot([rows[1]]),
              }};
            `, sandbox);
            process.stdout.write(JSON.stringify(sandbox.__result));
            """
        )
        completed = subprocess.run(
            ["node", "-e", script],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            self.fail(completed.stderr)
        result = json.loads(completed.stdout)

        # Source pixels use the two original 10x8 extents, not the padded
        # 2x12x12 storage tensor, and RGB channels are not pixels.
        self.assertEqual(result["deep"]["pixels"], 160)
        self.assertEqual(result["deep"]["uses"], 100)
        self.assertIsNone(result["deep"]["bits"])
        self.assertAlmostEqual(result["deep"]["ratio"], 0.625)
        self.assertEqual(result["deep"]["pointX"], [0.625, 0.625])
        self.assertTrue(all("digital bits" not in title for title in result["deep"]["titles"]))

        self.assertEqual(result["digital"]["pixels"], 100)
        self.assertEqual(result["digital"]["uses"], 100)
        self.assertEqual(result["digital"]["bits"], 200)
        self.assertAlmostEqual(result["digital"]["ratio"], 1.0)
        self.assertAlmostEqual(result["digital"]["bpp"], 2.0)
        self.assertIn("Transmitted digital bits 200", result["digital"]["titles"][0])

        self.assertEqual(
            result["presentation"]["context"],
            "image-reconstruction-channel-resource",
        )
        self.assertEqual(
            result["presentation"]["settings"]["x"],
            "channel_uses_per_pixel",
        )
        self.assertEqual(
            result["presentation"]["title"],
            "Image Reconstruction · Rate–Distortion",
        )
        self.assertIn("Image Reconstruction · Rate–Distortion", result["markup"])
        self.assertIn("Bandwidth ratio (complex channel uses/pixel)", result["markup"])
        self.assertIn("common physical-channel resource", result["presentation"]["note"])
        self.assertIn("continuous-symbol JSCC has no transmitted bitstream", result["presentation"]["note"])
        self.assertNotIn('class="rd-series-line"', result["markup"])
        self.assertNotIn("Task Performance", result["markup"])

        # A purely digital image-compression result follows the conventional
        # measured bitstream-bpp axis, even when its modulation-stage channel
        # use count is also available. The channel-use axis is reserved for a
        # continuous-symbol or mixed digital/continuous comparison.
        self.assertEqual(
            result["digitalPresentation"]["context"],
            "image-reconstruction-digital-rate",
        )
        self.assertEqual(result["digitalPresentation"]["settings"]["x"], "rate_bpp")
        self.assertIn("measured transmit-boundary bits", result["digitalPresentation"]["note"])
        self.assertIn("on-air coded bpp", result["digitalPresentation"]["note"])
        self.assertIn("compression-only source-codec bpp", result["digitalPresentation"]["note"])
        self.assertIn("Transmit rate (bits/pixel)", result["digitalMarkup"])
        self.assertNotIn("Bandwidth ratio (complex channel uses/pixel)", result["digitalMarkup"])


if __name__ == "__main__":
    unittest.main()
