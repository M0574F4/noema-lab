from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
import urllib.error
import urllib.request

from noema_lab.ui.server import start_ui_server_in_thread


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
INDEX_HTML = ROOT / "src" / "noema_lab" / "ui" / "static" / "index.html"


def _post_json(url: str, payload: object) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _project_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


class RecipeValidateEndpointTests(unittest.TestCase):
    def test_validate_rejects_duplicate_request_keys_before_recipe_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            project_root.mkdir()
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                root / "workspace",
                project_root,
            )
            host, port = server.server_address
            request = urllib.request.Request(
                "http://%s:%d/api/recipe/validate" % (host, port),
                data=b'{"recipe":{},"recipe":{"schema_version":1}}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(request, timeout=10)
                response = json.loads(
                    caught.exception.read().decode("utf-8")
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

        self.assertEqual(caught.exception.code, 400)
        self.assertIn("Duplicate JSON object key", response["error"])

    def test_validate_strictly_normalizes_authored_recipe_without_writing(self) -> None:
        recipe = {
            "schema_version": 1,
            "name": "raw_editor_matrix",
            "metadata": {
                "sweeps": {"source.bit_count": "8,16"},
                "ui_sweeps": {"source.bit_count": "32,64"},
            },
            "steps": [
                {
                    "id": "source",
                    "op": "source.random_bits",
                    "inputs": {},
                    "params": {"bit_count": 8},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            recipes_dir = project_root / "recipes"
            recipes_dir.mkdir(parents=True)
            sentinel = recipes_dir / "existing.yaml"
            sentinel.write_text("sentinel: unchanged\n", encoding="utf-8")
            before = _project_snapshot(project_root)
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                root / "workspace",
                project_root,
            )
            host, port = server.server_address
            try:
                status, response = _post_json(
                    "http://%s:%d/api/recipe/validate" % (host, port),
                    {"recipe": recipe},
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

            self.assertEqual(status, 200)
            self.assertEqual(response["status"], "valid")
            self.assertEqual(response["validation"]["status"], "valid")
            self.assertEqual(response["validation"]["mode"], "strict")
            self.assertIsInstance(
                response["validation"]["defaults_materialized"], bool
            )
            self.assertEqual(response["recipe"]["name"], "raw_editor_matrix")
            metadata = response["recipe"]["metadata"]
            self.assertNotIn("sweeps", metadata)
            self.assertNotIn("ui_sweeps", metadata)
            self.assertIn(
                [8, 16],
                list(metadata["matrix"]["dimensions"].values()),
            )
            diagnostic_codes = {
                item["code"] for item in response["validation"]["diagnostics"]
            }
            self.assertIn("legacy_sweep_precedence", diagnostic_codes)
            self.assertIn("legacy_sweep_normalized", diagnostic_codes)
            self.assertEqual(_project_snapshot(project_root), before)

    def test_validate_rejects_invalid_json_object_without_writing(self) -> None:
        invalid_recipe = {
            "schema_version": 1,
            "name": "raw_editor_typo",
            "metdata": {"seed": 7},
            "steps": [
                {
                    "id": "source",
                    "op": "source.random_bits",
                    "inputs": {},
                    "params": {"bit_count": 8},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            recipes_dir = project_root / "recipes"
            recipes_dir.mkdir(parents=True)
            sentinel = recipes_dir / "existing.yaml"
            sentinel.write_text("sentinel: unchanged\n", encoding="utf-8")
            before = _project_snapshot(project_root)
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                root / "workspace",
                project_root,
            )
            host, port = server.server_address
            base = "http://%s:%d/api/recipe/validate" % (host, port)
            try:
                typo_status, typo_response = _post_json(
                    base, {"recipe": invalid_recipe}
                )
                scalar_status, scalar_response = _post_json(
                    base, {"recipe": "not an object"}
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

            self.assertEqual(typo_status, 400)
            self.assertEqual(typo_response["status"], "error")
            self.assertIn("Unknown recipe field `metdata`", typo_response["error"])
            self.assertEqual(scalar_status, 400)
            self.assertEqual(scalar_response["status"], "error")
            self.assertIn("recipe must be a JSON object", scalar_response["error"])
            self.assertEqual(_project_snapshot(project_root), before)

    def test_validate_rejects_invalid_capture_contract_without_writing(self) -> None:
        base_recipe = {
            "schema_version": 1,
            "name": "raw_editor_capture",
            "steps": [
                {
                    "id": "source",
                    "op": "source.random_bits",
                    "inputs": {},
                    "params": {"bit_count": 8},
                }
            ],
        }
        missing_output = {
            **base_recipe,
            "dataset_capture": {
                "taps": [{"id": "signal", "from": "source.missing"}]
            },
        }
        reserved_id = {
            **base_recipe,
            "dataset_capture": {
                "taps": [{"id": "metadata_json", "from": "source.bits"}]
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            recipes_dir = project_root / "recipes"
            recipes_dir.mkdir(parents=True)
            sentinel = recipes_dir / "existing.yaml"
            sentinel.write_text("sentinel: unchanged\n", encoding="utf-8")
            before = _project_snapshot(project_root)
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                root / "workspace",
                project_root,
            )
            host, port = server.server_address
            base = "http://%s:%d/api/recipe/validate" % (host, port)
            try:
                missing_status, missing_response = _post_json(
                    base, {"recipe": missing_output}
                )
                reserved_status, reserved_response = _post_json(
                    base, {"recipe": reserved_id}
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

            self.assertEqual(missing_status, 400)
            self.assertIn(
                "references unknown output source.missing",
                missing_response["error"],
            )
            self.assertEqual(reserved_status, 400)
            self.assertIn(
                "Tap id is reserved: metadata_json",
                reserved_response["error"],
            )
            self.assertEqual(_project_snapshot(project_root), before)


class RawRecipeEditorStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app_js = APP_JS.read_text(encoding="utf-8")
        cls.index_html = INDEX_HTML.read_text(encoding="utf-8")

    @staticmethod
    def _function_source(source: str, name: str) -> str:
        match = re.search(
            rf"(?:async\s+)?function\s+{re.escape(name)}\s*\([^)]*\)\s*\{{",
            source,
        )
        if not match:
            raise AssertionError(f"missing JavaScript function {name}")
        next_function = re.search(
            r"\n(?:async\s+)?function\s+[A-Za-z_$][\w$]*\s*\(",
            source[match.end() :],
        )
        end = (
            match.end() + next_function.start()
            if next_function
            else len(source)
        )
        return source[match.start() : end]

    def test_header_action_is_raw_editor_instead_of_copy_only_action(self) -> None:
        self.assertRegex(
            self.index_html,
            r'<button\s+id="rawRecipeEditorButton"[^>]*aria-label="Edit raw recipe JSON"',
        )
        self.assertNotIn('id="copyRecipeConfigButton"', self.index_html)
        self.assertIn(
            'rawRecipeEditorButton: document.getElementById("rawRecipeEditorButton")',
            self.app_js,
        )
        self.assertRegex(
            self.app_js,
            r"rawRecipeEditorButton\.addEventListener\(\s*[\"']click[\"']\s*,"
            r"\s*openRawRecipeEditor\s*\)",
        )

    def test_editor_supports_edit_copy_paste_and_validated_apply(self) -> None:
        open_source = self._function_source(self.app_js, "openRawRecipeEditor")
        copy_source = self._function_source(self.app_js, "copyRawRecipeText")
        paste_source = self._function_source(self.app_js, "pasteRawRecipeText")
        apply_source = self._function_source(self.app_js, "applyRawRecipeText")

        serialization_source = self._function_source(
            self.app_js, "currentRecipeJsonText"
        )
        self.assertIn("currentRecipeJsonText", open_source)
        self.assertIn("JSON.stringify", serialization_source)
        self.assertIn("stripRecipeWorkingStateMetadata", serialization_source)
        self.assertIn("navigator.clipboard.writeText", copy_source)
        self.assertIn("navigator.clipboard.readText", paste_source)
        self.assertIn("strictJsonParse", apply_source)
        self.assertIn("recipeJsonNumberProblem", apply_source)
        self.assertIn("/api/recipe/validate", apply_source)
        self.assertNotIn("/api/recipe/save", apply_source)
        self.assertIn("applyBuiltRecipe", apply_source)
        self.assertRegex(apply_source, r"\b(?:payload|response)\.recipe\b")


@unittest.skipUnless(shutil.which("node"), "Node.js is required for raw editor UI tests")
class RawRecipeEditorBehaviorTests(unittest.TestCase):
    def _run_node(self, body: str) -> dict:
        script = r"""
const fs = require("fs");
const vm = require("vm");
let source = fs.readFileSync(__APP_PATH__, "utf8");
source = source.replace(/\ninit\(\);\s*$/, "\n");
const sandbox = {
  console,
  localStorage: { getItem: () => null, setItem: () => {} },
  document: { documentElement: { dataset: {} }, getElementById: () => null },
  window: { CSS: null },
  navigator: { clipboard: {} },
  setTimeout,
  clearTimeout,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
const promise = vm.runInContext(
  source + "\n;(async () => {\n" + __BODY__ + "\n})()",
  sandbox,
);
Promise.resolve(promise).then(
  (result) => process.stdout.write(JSON.stringify(result)),
  (error) => { console.error(error); process.exit(1); },
);
"""
        script = script.replace("__APP_PATH__", json.dumps(str(APP_JS)))
        script = script.replace("__BODY__", json.dumps(body))
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

    def test_apply_uses_validated_payload_and_updates_only_open_working_copy(self) -> None:
        result = self._run_node(
            r"""
const original = {
  schema_version: 1,
  name: "before",
  metadata: {},
  steps: [{ id: "source", op: "source.random_bits", inputs: {}, params: { bit_count: 8 } }],
};
const parsed = {
  schema_version: 1,
  name: "pasted",
  metadata: {},
  steps: [{ id: "source", op: "source.random_bits", inputs: {}, params: { bit_count: 16 } }],
};
const normalized = JSON.parse(JSON.stringify(parsed));
normalized.description = "normalized by server";
const textarea = { value: JSON.stringify(parsed) };
const status = { textContent: "", hidden: true, className: "" };
const applyButton = {
  disabled: false,
  setAttribute: () => {},
  removeAttribute: () => {},
};
els.settingsBody = {
  querySelector: (selector) => {
    if (selector.includes("status")) return status;
    if (selector.includes("apply")) return applyButton;
    if (selector.includes("text")) return textarea;
    return null;
  },
};
els.settingsOverlay = { hidden: false };
state.editRecipe = original;
state.editRecipeKey = "working:one";
state.selectedRecipeKey = "working:one";
state.selectedRecipe = {
  key: "working:one",
  name: "before",
  recipe: original,
  workingCopy: true,
  configuratorView: "experiment",
};
state.recipes = [state.selectedRecipe];
state.settingsSection = "raw-recipe";
state.rawRecipeEditorGeneration = 1;
state.rawRecipeEditorSession = {
  generation: 1,
  recipe: original,
  envelope: state.selectedRecipe,
  selectedRecipeKey: "working:one",
  controller: null,
  pending: false,
};
let request = null;
let applied = null;
let rendered = null;
api = async (path, options) => {
  request = { path, options };
  return { status: "valid", recipe: normalized, validation: { status: "valid" } };
};
applyBuiltRecipe = (recipe, renderControls) => {
  applied = recipe;
  rendered = renderControls;
  state.editRecipe = recipe;
  state.selectedRecipe.recipe = recipe;
};
notify = () => {};
await applyRawRecipeText();
return {
  path: request && request.path,
  method: request && request.options && request.options.method,
  posted: request && JSON.parse(request.options.body),
  applied,
  rendered,
  modalHidden: els.settingsOverlay.hidden,
  selectedRecipe: state.selectedRecipe.recipe,
};
"""
        )

        self.assertEqual(result["path"], "/api/recipe/validate")
        self.assertEqual(result["method"], "POST")
        self.assertEqual(result["posted"]["recipe"]["name"], "pasted")
        self.assertEqual(result["applied"]["name"], "pasted")
        self.assertEqual(
            result["applied"]["description"], "normalized by server"
        )
        self.assertEqual(result["selectedRecipe"], result["applied"])
        self.assertTrue(result["rendered"])
        self.assertTrue(result["modalHidden"])

    def test_late_validation_result_is_ignored_after_editor_closes(self) -> None:
        result = self._run_node(
            r"""
const original = {
  schema_version: 1,
  name: "original",
  metadata: {},
  steps: [{ id: "source", op: "source.random_bits", inputs: {}, params: { bit_count: 8 } }],
};
const normalized = { ...original, name: "late-response" };
const textarea = { value: JSON.stringify(normalized) };
const status = { textContent: "", hidden: true, className: "" };
const applyButton = { disabled: false, setAttribute: () => {}, removeAttribute: () => {} };
els.settingsBody = {
  querySelector: (selector) => {
    if (selector.includes("status")) return status;
    if (selector.includes("apply")) return applyButton;
    if (selector.includes("text")) return textarea;
    return null;
  },
};
els.settingsOverlay = { hidden: false };
const firstTab = { key: "working:one", recipe: original, workingCopy: true };
const secondRecipe = { ...original, name: "other-tab" };
const secondTab = { key: "working:two", recipe: secondRecipe, workingCopy: true };
state.recipes = [firstTab, secondTab];
state.selectedRecipe = firstTab;
state.selectedRecipeKey = firstTab.key;
state.editRecipe = original;
state.editRecipeKey = firstTab.key;
state.settingsSection = "raw-recipe";
state.rawRecipeEditorGeneration = 1;
state.rawRecipeEditorSession = {
  generation: 1,
  recipe: original,
  envelope: firstTab,
  selectedRecipeKey: firstTab.key,
  controller: null,
  pending: false,
};
let resolveValidation;
api = () => new Promise((resolve) => { resolveValidation = resolve; });
let applied = false;
applyBuiltRecipe = () => { applied = true; };
renderRecipeControls = () => {};
notify = () => {};
const applying = applyRawRecipeText();
await Promise.resolve();
closeSettingsDialog();
state.selectedRecipe = secondTab;
state.selectedRecipeKey = secondTab.key;
state.editRecipe = secondRecipe;
state.editRecipeKey = secondTab.key;
resolveValidation({ status: "valid", recipe: normalized, validation: { status: "valid" } });
await applying;
return {
  applied,
  firstName: firstTab.recipe.name,
  secondName: secondTab.recipe.name,
  generation: state.rawRecipeEditorGeneration,
  sessionCleared: state.rawRecipeEditorSession === null,
};
"""
        )

        self.assertFalse(result["applied"])
        self.assertEqual(result["firstName"], "original")
        self.assertEqual(result["secondName"], "other-tab")
        self.assertGreater(result["generation"], 1)
        self.assertTrue(result["sessionCleared"])

    def test_apply_rejects_unsafe_json_integer_before_validation(self) -> None:
        result = self._run_node(
            r"""
const original = {
  schema_version: 1,
  name: "original",
  metadata: { seed: 7 },
  steps: [{ id: "source", op: "source.random_bits", inputs: {}, params: { bit_count: 8 } }],
};
const textarea = {
  value: '{"schema_version":1,"name":"unsafe","metadata":{"seed":9007199254740993},"steps":[{"id":"source","op":"source.random_bits","inputs":{},"params":{"bit_count":8}}]}',
};
const status = { textContent: "", hidden: true, className: "" };
const applyButton = { disabled: false, setAttribute: () => {}, removeAttribute: () => {} };
els.settingsBody = {
  querySelector: (selector) => {
    if (selector.includes("status")) return status;
    if (selector.includes("apply")) return applyButton;
    if (selector.includes("text")) return textarea;
    return null;
  },
};
const tab = { key: "working:one", recipe: original, workingCopy: true };
state.recipes = [tab];
state.selectedRecipe = tab;
state.selectedRecipeKey = tab.key;
state.editRecipe = original;
state.editRecipeKey = tab.key;
state.settingsSection = "raw-recipe";
state.rawRecipeEditorGeneration = 1;
state.rawRecipeEditorSession = {
  generation: 1,
  recipe: original,
  envelope: tab,
  selectedRecipeKey: tab.key,
  controller: null,
  pending: false,
};
let apiCalls = 0;
let applied = false;
api = async () => { apiCalls += 1; return { recipe: {} }; };
applyBuiltRecipe = () => { applied = true; };
await applyRawRecipeText();
return {
  apiCalls,
  applied,
  unchanged: state.editRecipe === original && tab.recipe === original,
  status: status.textContent,
  pending: state.rawRecipeEditorSession.pending,
};
"""
        )

        self.assertEqual(result["apiCalls"], 0)
        self.assertFalse(result["applied"])
        self.assertTrue(result["unchanged"])
        self.assertFalse(result["pending"])
        self.assertIn("integer exceeds JavaScript's exact JSON range", result["status"])

    def test_user_json_parser_rejects_duplicate_keys_and_nonfinite_numbers(self) -> None:
        result = self._run_node(
            r"""
function parse(raw) {
  try {
    return { ok: true, value: strictJsonParse(raw) };
  } catch (error) {
    return { ok: false, error: error.message };
  }
}
return {
  valid: parse('{"outer":{"seed":7},"items":[1,2]}'),
  nestedDuplicate: parse('{"outer":{"seed":1,"seed":9}}'),
  escapedDuplicate: parse('{"a":1,"\\u0061":2}'),
  overflow: parse('{"value":1e400}'),
};
"""
        )

        self.assertEqual(
            result["valid"]["value"],
            {"outer": {"seed": 7}, "items": [1, 2]},
        )
        self.assertFalse(result["nestedDuplicate"]["ok"])
        self.assertIn("duplicate JSON object key", result["nestedDuplicate"]["error"])
        self.assertFalse(result["escapedDuplicate"]["ok"])
        self.assertIn("duplicate JSON object key", result["escapedDuplicate"]["error"])
        self.assertFalse(result["overflow"]["ok"])
        self.assertIn("finite JSON number", result["overflow"]["error"])


if __name__ == "__main__":
    unittest.main()
