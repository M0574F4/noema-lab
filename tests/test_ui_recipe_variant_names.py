from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"


@unittest.skipUnless(shutil.which("node"), "Node.js is required for the UI naming regression test")
class UiRecipeVariantNameTests(unittest.TestCase):
    def _labels(self, recipes: list[dict]) -> dict[str, str]:
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
              const __recipes = ${{JSON.stringify({json.dumps(recipes)})}};
              const __labels = recipeDisplayLabels(__recipes);
              globalThis.__result = Object.fromEntries(__recipes.map((recipe) => [recipe.key, recipeDisplayName(recipe, __labels)]));
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
        return json.loads(completed.stdout)

    def _next_duplicate_name(self, source_name: str, existing_names: list[str]) -> str:
        recipes = [
            {"name": name, "recipe": {"name": name}}
            for name in existing_names
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
              state.recipes = ${{JSON.stringify({json.dumps(recipes)})}};
              globalThis.__result = nextDuplicateRecipeName({json.dumps(source_name)});
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
        return json.loads(completed.stdout)

    @staticmethod
    def _recipe(key: str, tab_label: str, name: str, policy: str, target_power: float = 1.0) -> dict:
        return {
            "key": key,
            "tabLabel": tab_label,
            "name": name,
            "recipe": {
                "name": name,
                "metadata": {"ui_configured": False},
                "steps": [
                    {
                        "id": "tx_power",
                        "op": "model.symbol_power_allocator",
                        "params": {
                            "policy": policy,
                            "granularity": "per_subcarrier",
                            "budget_mode": "fixed_average",
                            "target_power": target_power,
                        },
                    }
                ],
            },
        }

    def test_loaded_template_duplicates_are_named_by_allocator_policy(self):
        recipes = [
            self._recipe("water", "Resource water filling baseline", "resource_water_filling_baseline", "water_filling"),
            self._recipe("equal", "Recipe 2", "resource_water_filling_baseline_copy", "fixed"),
        ]

        labels = self._labels(recipes)

        self.assertEqual(labels["water"], "water filling")
        self.assertEqual(labels["equal"], "equal power")
        self.assertNotIn("Recipe 2", labels.values())

    def test_loaded_template_power_variants_are_named_by_power_budget(self):
        recipes = [
            self._recipe("one", "Resource allocation", "resource_allocation", "fixed", 1.0),
            self._recipe("two", "Recipe 2", "resource_allocation_copy", "fixed", 2.0),
        ]

        labels = self._labels(recipes)

        self.assertEqual(labels["one"], "power 1")
        self.assertEqual(labels["two"], "power 2")

    def test_explicit_display_name_is_preserved(self):
        named = self._recipe("named", "Recipe 2", "generated_copy", "fixed")
        named["displayName"] = "My measured allocator"
        other = self._recipe("water", "Recipe 1", "generated", "water_filling")

        labels = self._labels([named, other])

        self.assertEqual(labels["named"], "My measured allocator")

    def test_duplicate_names_use_incrementing_numeric_suffixes(self):
        self.assertEqual(self._next_duplicate_name("name", ["name"]), "name_1")
        self.assertEqual(
            self._next_duplicate_name("name", ["name", "name_1", "name_2"]),
            "name_3",
        )
        self.assertEqual(
            self._next_duplicate_name("name_10", ["name_10"]),
            "name_11",
        )
        self.assertEqual(
            self._next_duplicate_name("name_10", ["name_10", "name_11"]),
            "name_12",
        )


if __name__ == "__main__":
    unittest.main()
