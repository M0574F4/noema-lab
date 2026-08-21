#!/usr/bin/env python3
"""Drive the built flagship comparison demo in a real Chrome browser."""

from __future__ import annotations

import argparse
import base64
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import traceback
from typing import Any, Mapping, Optional
import urllib.parse

from ui_browser_probe import (
    CDP,
    ProbeFailure,
    find_chrome,
    free_loopback_port,
    http_json,
    no_proxy_opener,
    require,
    wait_for_http_json,
)


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


def _evaluate_state(cdp: CDP) -> Mapping[str, Any]:
    value = cdp.evaluate(
        """(() => {
          const root = document.getElementById('noema-break-comparison');
          const checks = Object.fromEntries(
            ['condition', 'aggregation', 'metric', 'role'].map(
              name => [name, document.getElementById(`btc-check-${name}`).dataset.check]
            )
          );
          return {
            state: root.dataset.state,
            busy: root.getAttribute('aria-busy'),
            verdict: document.getElementById('btc-result-title').textContent,
            detail: document.getElementById('btc-verdict-detail').textContent,
            claim: document.getElementById('btc-claim-value').textContent,
            diagnosis: document.getElementById('btc-diagnosis-list').textContent,
            disclosure: document.getElementById('btc-disclosure').textContent,
            checked: [...root.querySelectorAll('input[data-fault]')]
              .filter(input => input.checked).map(input => input.dataset.fault),
            checks,
            overflow: document.documentElement.scrollWidth - window.innerWidth,
            errors: window.__noemaProbeErrors || [],
          };
        })()"""
    )
    require(isinstance(value, dict), "Browser did not return demo state")
    return value


def _select_fault(cdp: CDP, fault: str) -> Mapping[str, Any]:
    cdp.evaluate(
        """(() => {
          document.querySelectorAll('#noema-break-comparison input[data-fault]')
            .forEach(input => { input.checked = input.dataset.fault === %s; });
          const selected = document.getElementById('btc-fault-' + %s);
          selected.dispatchEvent(new Event('change', { bubbles: true }));
          selected.focus();
        })()""" % (json.dumps(fault), json.dumps(fault))
    )
    state = _evaluate_state(cdp)
    require(state["state"] == "broken", "%s fault did not break the comparison" % fault)
    require(state["checks"][fault] == "fail", "%s audit row did not fail" % fault)
    require(state["checked"] == [fault], "%s fault was not isolated" % fault)
    return state


def run(url: str, chrome: str, timeout: float, screenshot: Optional[Path]) -> Mapping[str, Any]:
    opener = no_proxy_opener()
    devtools_port = free_loopback_port()
    # Chrome helper processes can briefly touch the profile after the browser
    # process exits. A cleanup race must not turn a successful UI probe into a
    # failed CI job.
    with tempfile.TemporaryDirectory(
        prefix="noema-break-comparison-probe-",
        ignore_cleanup_errors=True,
    ) as raw:
        temporary = Path(raw)
        chrome_log = temporary / "chrome.log"
        command = [
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--disable-background-networking",
            "--disable-extensions",
            "--no-first-run",
            "--remote-allow-origins=*",
            "--remote-debugging-address=127.0.0.1",
            "--remote-debugging-port=%d" % devtools_port,
            "--user-data-dir=%s" % (temporary / "profile"),
            "about:blank",
        ]
        with chrome_log.open("wb") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        cdp: Optional[CDP] = None
        try:
            devtools = "http://127.0.0.1:%d" % devtools_port
            wait_for_http_json(
                opener,
                devtools + "/json/version",
                "Chrome DevTools endpoint",
                timeout=timeout,
            )
            target = http_json(
                opener,
                devtools + "/json/new?" + urllib.parse.quote("about:blank", safe=""),
                method="PUT",
            )
            cdp = CDP(str(target["webSocketDebuggerUrl"]), timeout)
            cdp.connect()
            cdp.call("Page.enable")
            cdp.call("Runtime.enable")
            cdp.call(
                "Page.addScriptToEvaluateOnNewDocument",
                {
                    "source": """
                      window.__noemaProbeErrors = [];
                      window.addEventListener('error', event => {
                        window.__noemaProbeErrors.push(event.message || 'window error');
                      });
                      window.addEventListener('unhandledrejection', event => {
                        window.__noemaProbeErrors.push(String(event.reason));
                      });
                    """,
                },
            )
            cdp.call(
                "Emulation.setDeviceMetricsOverride",
                {"width": 1440, "height": 1000, "deviceScaleFactor": 1, "mobile": False},
            )
            cdp.call("Page.navigate", {"url": url})
            cdp.wait_for(
                "document.getElementById('noema-break-comparison')?.dataset.state === 'intact'",
                "intact demo state",
            )

            initial = _evaluate_state(cdp)
            require(initial["busy"] == "false", "Demo remained busy after evidence load")
            require(initial["verdict"] == "Contract intact", "Unexpected initial verdict")
            require(initial["overflow"] <= 0, "Desktop page has horizontal overflow")
            require(not initial["errors"], "Browser errors: %r" % initial["errors"])
            require(
                "not a publication-ready" in initial["disclosure"],
                "Experimental disclosure is not visible",
            )

            fault_states = {fault: _select_fault(cdp, fault) for fault in (
                "condition",
                "aggregation",
                "metric",
                "role",
            )}
            require(
                fault_states["metric"]["claim"] == "Not comparable",
                "Metric mismatch still produced a percentage",
            )

            cdp.evaluate("document.getElementById('btc-reset').click()")
            reset = _evaluate_state(cdp)
            require(
                reset["state"] == "intact" and not reset["checked"],
                "Reset did not restore contract",
            )

            cdp.call(
                "Emulation.setDeviceMetricsOverride",
                {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True},
            )
            mobile = _evaluate_state(cdp)
            require(mobile["overflow"] <= 0, "Mobile page has horizontal overflow")

            if screenshot is not None:
                screenshot.parent.mkdir(parents=True, exist_ok=True)
                capture = cdp.call(
                    "Page.captureScreenshot",
                    {"format": "png", "captureBeyondViewport": False},
                )
                screenshot.write_bytes(base64.b64decode(str(capture["data"])))

            return {
                "url": url,
                "initial": initial,
                "fault_verdicts": {
                    fault: state["verdict"] for fault, state in fault_states.items()
                },
                "metric_mismatch_claim": fault_states["metric"]["claim"],
                "reset_state": reset["state"],
                "mobile_overflow_px": mobile["overflow"],
                "screenshot": str(screenshot) if screenshot is not None else None,
            }
        except Exception as exc:
            log = chrome_log.read_text(encoding="utf-8", errors="replace")
            raise ProbeFailure(
                "%s\nChrome log tail:\n%s"
                % (exc, "\n".join(log.splitlines()[-30:]))
            ) from exc
        finally:
            if cdp is not None:
                try:
                    cdp.call("Browser.close")
                except Exception:
                    pass
                cdp.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Built demo URL served over HTTP")
    source.add_argument(
        "--site-dir",
        type=Path,
        help="Built Sphinx HTML directory to serve on a temporary loopback port",
    )
    parser.add_argument("--chrome", default="", help="Chrome/Chromium executable")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--screenshot", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    server: Optional[ThreadingHTTPServer] = None
    server_thread: Optional[threading.Thread] = None
    try:
        url = args.url
        if args.site_dir is not None:
            site_dir = args.site_dir.resolve()
            require(
                (site_dir / "break_the_comparison.html").is_file(),
                "Built flagship page is missing from %s" % site_dir,
            )
            handler = functools.partial(_QuietHandler, directory=str(site_dir))
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            url = "http://127.0.0.1:%d/break_the_comparison.html" % server.server_port
        require(url, "A URL or built site directory is required")
        payload = run(
            str(url),
            find_chrome(args.chrome),
            max(5.0, args.timeout),
            args.screenshot,
        )
    except Exception as exc:
        print("Flagship browser probe failed: %s" % exc, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=5)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
