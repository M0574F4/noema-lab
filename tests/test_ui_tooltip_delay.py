from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"


class UiTooltipDelayTests(unittest.TestCase):
    def test_native_and_custom_tooltips_share_the_delegated_controller(self):
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('root.querySelectorAll("[title]")', source)
        self.assertIn('target.closest("[data-result-tooltip], [data-ui-tooltip], [title]")', source)
        self.assertIn('document.addEventListener("pointerover", showUiTooltipFromEvent)', source)
        self.assertIn('document.addEventListener("focusin", showUiTooltipFromEvent)', source)
        self.assertIn("const UI_TOOLTIP_POINTER_DELAY_MS = 1000;", source)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the tooltip timing test")
    def test_pointer_dwell_is_delayed_and_cancelable_while_focus_is_immediate(self):
        script = textwrap.dedent(
            rf"""
            const fs = require("fs");
            const vm = require("vm");
            const app = fs.readFileSync({json.dumps(str(APP_JS))}, "utf8");
            const start = app.indexOf("function ensureResultValueTooltip()");
            const end = app.indexOf("const SVG_EXPORT_STYLE_PROPERTIES", start);
            const source = app.slice(start, end);

            let nextTimerId = 1;
            const timers = new Map();
            const scheduledDelays = [];
            const fakeSetTimeout = (callback, delay) => {{
              const id = nextTimerId++;
              timers.set(id, callback);
              scheduledDelays.push(delay);
              return id;
            }};
            const fakeClearTimeout = (id) => timers.delete(id);
            const runNextTimer = () => {{
              const entry = timers.entries().next().value;
              if (!entry) return false;
              timers.delete(entry[0]);
              entry[1]();
              return true;
            }};
            const classList = () => {{
              const values = new Set();
              return {{
                add: (...names) => names.forEach((name) => values.add(name)),
                remove: (...names) => names.forEach((name) => values.delete(name)),
                contains: (name) => values.has(name),
              }};
            }};
            const tooltip = {{
              hidden: true,
              textContent: "",
              classList: classList(),
              style: {{}},
              getBoundingClientRect: () => ({{ width: 120, height: 32 }}),
            }};
            const makeTarget = (text) => {{
              const attributes = new Map([["data-ui-tooltip", text]]);
              const target = {{
                hovered: false,
                isConnected: true,
                classList: classList(),
                closest: () => target,
                contains: (candidate) => candidate === target,
                getAttribute: (name) => attributes.get(name) || "",
                hasAttribute: (name) => attributes.has(name),
                matches: (selector) => selector === ":hover" ? target.hovered : false,
                getBoundingClientRect: () => ({{ left: 20, top: 30, width: 80, height: 20 }}),
              }};
              return target;
            }};
            const document = {{
              querySelector: (selector) => selector === ".ui-value-tooltip" ? tooltip : null,
            }};
            const sandbox = {{
              document,
              window: {{ innerWidth: 1024, innerHeight: 768 }},
              setTimeout: fakeSetTimeout,
              clearTimeout: fakeClearTimeout,
              makeTarget,
              runNextTimer,
              scheduledDelays,
              timers,
              tooltip,
            }};
            sandbox.globalThis = sandbox;
            vm.createContext(sandbox);
            vm.runInContext(source + `
              const pointerTarget = makeTarget("pointer details");
              pointerTarget.hovered = true;
              showUiTooltipFromEvent({{
                type: "pointerover",
                target: pointerTarget,
                relatedTarget: null,
                clientX: 40,
                clientY: 50,
              }});
              const beforeDwell = {{
                hidden: tooltip.hidden,
                active: pointerTarget.classList.contains("is-tooltip-active"),
                timerCount: timers.size,
              }};
              runNextTimer();
              const afterDwell = {{
                hidden: tooltip.hidden,
                text: tooltip.textContent,
                active: pointerTarget.classList.contains("is-tooltip-active"),
              }};
              hideUiTooltipFromPointerEvent({{
                type: "pointerout",
                target: pointerTarget,
                relatedTarget: null,
              }});

              const cancelledTarget = makeTarget("should not appear");
              cancelledTarget.hovered = true;
              showUiTooltipFromEvent({{
                type: "pointerover",
                target: cancelledTarget,
                relatedTarget: null,
                clientX: 60,
                clientY: 70,
              }});
              cancelledTarget.hovered = false;
              hideUiTooltipFromPointerEvent({{
                type: "pointerout",
                target: cancelledTarget,
                relatedTarget: null,
              }});
              const cancelled = {{ hidden: tooltip.hidden, timerCount: timers.size }};

              const focusTarget = makeTarget("keyboard details");
              const focusControl = {{ closest: () => focusTarget }};
              focusTarget.contains = (candidate) => candidate === focusTarget || candidate === focusControl;
              showUiTooltipFromEvent({{
                type: "focusin",
                target: focusControl,
                relatedTarget: null,
              }});
              const focused = {{
                hidden: tooltip.hidden,
                text: tooltip.textContent,
                active: focusTarget.classList.contains("is-tooltip-active"),
                timerCount: timers.size,
              }};
              hideUiTooltipFromFocusEvent({{
                type: "focusout",
                target: focusControl,
                relatedTarget: null,
              }});
              globalThis.__result = {{
                delay: scheduledDelays[0],
                beforeDwell,
                afterDwell,
                cancelled,
                focused,
                focusDismissed: tooltip.hidden,
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
        self.assertEqual(result["delay"], 1000)
        self.assertEqual(
            result["beforeDwell"],
            {"hidden": True, "active": False, "timerCount": 1},
        )
        self.assertEqual(
            result["afterDwell"],
            {"hidden": False, "text": "pointer details", "active": True},
        )
        self.assertEqual(result["cancelled"], {"hidden": True, "timerCount": 0})
        self.assertEqual(
            result["focused"],
            {
                "hidden": False,
                "text": "keyboard details",
                "active": True,
                "timerCount": 0,
            },
        )
        self.assertTrue(result["focusDismissed"])


if __name__ == "__main__":
    unittest.main()
