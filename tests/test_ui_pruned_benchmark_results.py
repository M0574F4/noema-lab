from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest

from noema_lab.core.artifacts import file_sha256
from noema_lab.core.benchmark_run_evidence import (
    RUN_EVIDENCE_SCHEMA_VERSION,
    snapshot_benchmark_run_evidence,
)
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.storage import LocalStore
from noema_lab.ui.server import (
    _benchmark_result_run_ids_from_path,
    _benchmark_result_run_payload,
)


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"


class UiPrunedBenchmarkResultServerTests(unittest.TestCase):
    def test_pruned_run_payload_comes_from_authenticated_compressed_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalStore(root / ".noema")
            result_id = "result-with-pruned-run"
            run_id = "run-pruned-after-snapshot"
            result_dir = store.get_benchmark_result_dir(result_id)
            run_dir = store.runs_dir / run_id
            result_dir.mkdir(parents=True)
            run_dir.mkdir(parents=True)

            recipe = {
                "schema_version": 1,
                "name": "learned_delayed_csi",
                "metadata": {},
                "steps": [
                    {
                        "id": "evaluation",
                        "op": "metrics.resource_reliability",
                        "params": {},
                    }
                ],
            }
            recipe_sha256 = canonical_json_sha256(recipe)
            execution_plan = {
                "schema_version": 1,
                "kind": "noema.execution_plan",
                "runner": "local",
                "recipe": {"sha256": recipe_sha256},
            }
            execution_plan["sha256"] = canonical_json_sha256(execution_plan)
            report_path = run_dir / "artifacts" / "evaluation" / "report.json"
            report_path.parent.mkdir(parents=True)
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "metrics": {
                            "resource.expected_finite_blocklength_goodput": 1.234,
                        },
                    }
                ),
                encoding="utf-8",
            )
            summary = {
                "schema_version": 1,
                "run_id": run_id,
                "recipe_name": recipe["name"],
                "status": "completed",
                "recipe_sha256": recipe_sha256,
                "metrics": {
                    "resource.expected_finite_blocklength_goodput": 1.234,
                },
                "steps": [
                    {
                        "id": "evaluation",
                        "metrics": {
                            "resource.expected_finite_blocklength_goodput": 1.234,
                        },
                        "outputs": {
                            "report": {
                                "kind": "metrics.report",
                                "path": str(report_path),
                                "metadata": {
                                    "method": "learned_allocator",
                                    "power_budget": 0.8,
                                },
                            }
                        },
                    }
                ],
            }
            manifest = {
                "schema_version": 1,
                "run_id": run_id,
                "recipe_name": recipe["name"],
                "status": "completed",
                "recipe": {
                    "sha256": recipe_sha256,
                    "authored_sha256": recipe_sha256,
                },
                "execution_plan": {
                    "schema_version": 1,
                    "sha256": execution_plan["sha256"],
                    "runner": "local",
                },
                "artifacts": [
                    {
                        "step_id": "evaluation",
                        "output_name": "report",
                        "kind": "metrics.report",
                        "path": str(report_path),
                        "relative_path": "artifacts/evaluation/report.json",
                        "sha256": file_sha256(report_path),
                        "metadata": {
                            "method": "learned_allocator",
                            "power_budget": 0.8,
                        },
                    }
                ],
            }
            for filename, payload in (
                ("recipe.json", recipe),
                ("recipe.authored.json", recipe),
                ("summary.json", summary),
                ("manifest.json", manifest),
                ("execution-plan.json", execution_plan),
            ):
                (run_dir / filename).write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )

            entry = {
                "id": "learned",
                "label": "Learned allocator",
                "recipe_name": recipe["name"],
                "run_id": run_id,
                "status": "completed",
                "recipe_sha256": recipe_sha256,
                "semantic_recipe_sha256": recipe_sha256,
                "metrics": {
                    **dict(summary["metrics"]),
                    "steps.evaluation.resource.expected_finite_blocklength_goodput": 1.234,
                },
                "metric_provenance": {
                    "resource.expected_finite_blocklength_goodput": {
                        "source_scope": "step",
                        "source_step": "evaluation",
                    }
                },
            }
            entry["run_evidence_snapshot"] = snapshot_benchmark_run_evidence(
                result_dir,
                entry_id=entry["id"],
                entry_index=0,
                run_dir=run_dir,
                run_id=run_id,
                semantic_recipe_sha256=recipe_sha256,
                metric_producer_steps={"evaluation"},
            )
            result = {
                "schema_version": 1,
                "kind": "noema.benchmark_result",
                "benchmark": {"id": "delayed-csi", "version": "2"},
                "status": "completed",
                "recipes": [entry],
            }
            (result_dir / "result.json").write_text(
                json.dumps(result),
                encoding="utf-8",
            )

            snapshot_root = result_dir / entry["run_evidence_snapshot"]["root"]
            self.assertEqual(
                entry["run_evidence_snapshot"]["schema_version"],
                RUN_EVIDENCE_SCHEMA_VERSION,
            )
            self.assertTrue((snapshot_root / "summary.json.gz").is_file())
            shutil.rmtree(run_dir)

            payload = _benchmark_result_run_payload(store, result_id, run_id)

            self.assertEqual(payload["recipe"], recipe)
            self.assertEqual(
                payload["metrics"]["resource.expected_finite_blocklength_goodput"],
                1.234,
            )
            artifact = payload["steps"][0]["outputs"]["report"]
            self.assertEqual(artifact["metadata"]["method"], "learned_allocator")
            self.assertTrue(Path(artifact["path"]).is_file())
            self.assertTrue(
                Path(artifact["path"]).is_relative_to(result_dir)
            )
            self.assertEqual(
                payload["benchmark_run_evidence"]["source"],
                "result_local_snapshot",
            )
            self.assertEqual(
                payload["benchmark_run_evidence"]["verification"]["status"],
                "valid",
            )

            self.assertEqual(
                _benchmark_result_run_ids_from_path(
                    "/api/benchmarks/results/%s/runs/%s"
                    % (result_id, run_id)
                ),
                (result_id, run_id),
            )


@unittest.skipUnless(
    shutil.which("node"),
    "Node.js is required for benchmark Results UI tests",
)
class UiPrunedBenchmarkResultClientTests(unittest.TestCase):
    def test_results_rows_fall_back_to_pruned_run_snapshot(self):
        script = textwrap.dedent(
            f"""
            const fs = require("fs");
            const vm = require("vm");
            let source = fs.readFileSync({json.dumps(str(APP_JS))}, "utf8");
            source = source.replace(/\\ninit\\(\\);\\s*$/, "\\n");
            const sandbox = {{
              console,
              localStorage: {{ getItem: () => null, setItem: () => {{}} }},
              document: {{
                documentElement: {{ dataset: {{}} }},
                getElementById: () => null,
                querySelectorAll: () => [],
              }},
              window: {{ CSS: null }},
              setTimeout,
              clearTimeout,
            }};
            sandbox.globalThis = sandbox;
            vm.createContext(sandbox);
            const promise = vm.runInContext(source + `
              state.selectedBenchmarkResultId = "benchmark-v2";
              state.runSummaries = {{}};
              const calls = [];
              api = async (path) => {{
                calls.push(path);
                if (path === "/api/benchmarks/results/benchmark-v2") {{
                  return {{
                    result: {{
                      benchmark: {{ id: "delayed-csi", version: "2" }},
                      status: "completed",
                      recipes: [{{
                        id: "learned",
                        label: "Learned allocator",
                        recipe_name: "learned_delayed_csi",
                        run_id: "pruned-run",
                        status: "completed",
                      }}],
                    }},
                  }};
                }}
                if (path === "/api/runs/pruned-run") {{
                  throw new Error("run not found");
                }}
                if (path === "/api/runs/legacy-retained-run") {{
                  return {{
                    run_id: "legacy-retained-run",
                    recipe_name: "legacy_recipe",
                    status: "completed",
                    metrics: {{ legacy: 1 }},
                    steps: [],
                  }};
                }}
                if (
                  path
                  === "/api/benchmarks/results/benchmark-v2/runs/pruned-run"
                ) {{
                  return {{
                    run_id: "pruned-run",
                    recipe_name: "learned_delayed_csi",
                    status: "completed",
                    metrics: {{
                      "resource.expected_finite_blocklength_goodput": 1.234,
                    }},
                    recipe: {{
                      name: "learned_delayed_csi",
                      metadata: {{}},
                      steps: [{{ id: "evaluation", params: {{}} }}],
                    }},
                    steps: [{{
                      id: "evaluation",
                      metrics: {{}},
                      outputs: {{
                        report: {{
                          kind: "metrics.report",
                          metadata: {{
                            method: "learned_allocator",
                            power_budget: 0.8,
                          }},
                        }},
                      }},
                    }}],
                    benchmark_run_evidence: {{
                      source: "result_local_snapshot",
                    }},
                  }};
                }}
                throw new Error("unexpected API call: " + path);
              }};
              (async () => {{
                const rows = await loadBenchmarkResultRows();
                const details = collectRunDetails(rows[0].summary);
                const legacy = await loadBenchmarkRunSummary(
                  "legacy-benchmark",
                  "legacy-retained-run",
                );
                return {{
                  calls,
                  error: rows[0].error || "",
                  recipeName: rows[0].recipe.name,
                  metric: details.metrics.find(
                    (row) =>
                      row.metric
                      === "resource.expected_finite_blocklength_goodput"
                  ),
                  artifact: details.artifacts[0],
                  source:
                    rows[0].summary.benchmark_run_evidence.source,
                  legacyMetric: legacy.metrics.legacy,
                }};
              }})()
            `, sandbox);
            Promise.resolve(promise).then(
              (result) => process.stdout.write(JSON.stringify(result)),
              (error) => {{ console.error(error); process.exit(1); }},
            );
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
        payload = json.loads(completed.stdout)

        self.assertEqual(
            payload["calls"],
            [
                "/api/benchmarks/results/benchmark-v2",
                "/api/runs/pruned-run",
                "/api/benchmarks/results/benchmark-v2/runs/pruned-run",
                "/api/runs/legacy-retained-run",
            ],
        )
        self.assertEqual(payload["error"], "")
        self.assertEqual(payload["recipeName"], "learned_delayed_csi")
        self.assertEqual(payload["metric"]["value"], 1.234)
        self.assertEqual(
            payload["artifact"]["artifact"]["metadata"]["method"],
            "learned_allocator",
        )
        self.assertEqual(payload["source"], "result_local_snapshot")
        self.assertEqual(payload["legacyMetric"], 1)


if __name__ == "__main__":
    unittest.main()
