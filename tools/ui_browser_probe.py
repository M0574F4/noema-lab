#!/usr/bin/env python3
"""Dependency-free, real-Chrome accessibility and responsive UI probe.

The probe starts an isolated Noema UI server and a temporary headless Chrome
profile, drives Chrome through the DevTools Protocol, and exits nonzero on the
first failed assertion.  It intentionally uses only the Python standard
library so the same command can run in the core CI environment.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, List, Mapping, Optional
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

DEPENDENCY_LIGHT_RECIPE = {
    "schema_version": 1,
    "name": "ui_browser_probe_bits",
    "description": "Dependency-light browser edit/run/results release probe.",
    "metadata": {
        "ui_configured": False,
        "research": {
            "dataset": {"id": "synthetic_random_bits", "modality": "bits", "split": "fixed_seed"},
            "task": {
                "id": "neural_receiver_demapping",
                "kind": "transport_integrity",
                "modality": "bits",
                "target": "demodulator.bits",
                "metrics": ["channel.payload.ber"],
            },
            "metrics": ["channel.payload.ber"],
        },
    },
    "steps": [
        {
            "id": "data",
            "op": "source.random_bits",
            "params": {"bit_count": 256, "batch_size": 1, "seed": 101},
        },
        {
            "id": "modulator",
            "op": "modulation.digital_modulate",
            "inputs": {"bits": "data.bits"},
            "params": {"modulation": "qpsk"},
        },
        {
            "id": "wireless_channel",
            "op": "wireless.channel",
            "inputs": {"symbols": "modulator.symbols"},
            "params": {"channel": "awgn", "snr_db": 0, "wireless_backend": "numpy", "seed": 303},
        },
        {
            "id": "demodulator",
            "op": "demodulation.digital_demodulate",
            "inputs": {"rx_symbols": "wireless_channel.rx_symbols"},
            "params": {"modulation": "auto"},
        },
        {
            "id": "payload_ber",
            "op": "metrics.bit_error_rate",
            "inputs": {"reference": "data.bits", "candidate": "demodulator.bits"},
            "params": {"label": "payload"},
        },
    ],
}


class ProbeFailure(AssertionError):
    """Raised when a browser-observed UI contract is not satisfied."""


def _strict_json(raw_text: str) -> Any:
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ProbeFailure("Duplicate JSON object key: %s" % key)
            value[key] = item
        return value

    def reject_constant(value):
        raise ProbeFailure("Non-finite JSON constant: %s" % value)

    payload = json.loads(
        raw_text,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )

    def require_finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ProbeFailure("Non-finite JSON number")
        if isinstance(value, dict):
            for item in value.values():
                require_finite(item)
        elif isinstance(value, list):
            for item in value:
                require_finite(item)

    require_finite(payload)
    return payload


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ProbeFailure(message)


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def no_proxy_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_bytes(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    method: str = "GET",
    timeout: float = 3.0,
) -> bytes:
    request = urllib.request.Request(url, method=method)
    with opener.open(request, timeout=timeout) as response:
        return response.read()


def http_json(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    method: str = "GET",
    timeout: float = 3.0,
) -> Any:
    return _strict_json(
        http_bytes(opener, url, method=method, timeout=timeout).decode("utf-8")
    )


def wait_for_http_json(
    opener: urllib.request.OpenerDirector,
    url: str,
    label: str,
    *,
    timeout: float,
) -> Any:
    deadline = time.monotonic() + timeout
    last_error: Optional[BaseException] = None
    while time.monotonic() < deadline:
        try:
            return http_json(opener, url)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(0.1)
    raise ProbeFailure(f"Timed out waiting for {label}: {last_error}")


class WebSocketClient:
    """Small RFC 6455 client sufficient for local Chrome CDP traffic."""

    def __init__(self, url: str, timeout: float) -> None:
        self.url = url
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self.buffer = bytearray()

    def connect(self) -> None:
        parsed = urllib.parse.urlsplit(self.url)
        require(parsed.scheme == "ws", f"Unsupported DevTools WebSocket URL: {self.url}")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        sock = socket.create_connection((host, port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        request = (
            f"GET {target} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        ).encode("ascii")
        sock.sendall(request)
        response = bytearray()
        while b"\r\n\r\n" not in response:
            chunk = sock.recv(4096)
            if not chunk:
                raise ProbeFailure("Chrome closed the DevTools WebSocket handshake")
            response.extend(chunk)
            require(len(response) <= 65536, "Oversized DevTools WebSocket handshake")
        header_bytes, remainder = bytes(response).split(b"\r\n\r\n", 1)
        header_lines = header_bytes.decode("iso-8859-1").split("\r\n")
        require(" 101 " in f" {header_lines[0]} ", f"DevTools WebSocket upgrade failed: {header_lines[0]}")
        headers: Dict[str, str] = {}
        for line in header_lines[1:]:
            if ":" in line:
                name, value = line.split(":", 1)
                headers[name.strip().lower()] = value.strip()
        expected_accept = base64.b64encode(
            hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()
        ).decode("ascii")
        require(
            headers.get("sec-websocket-accept") == expected_accept,
            "Chrome returned an invalid DevTools WebSocket accept key",
        )
        self.sock = sock
        self.buffer.extend(remainder)

    def close(self) -> None:
        if self.sock is None:
            return
        try:
            self._send_frame(0x8, b"")
        except OSError:
            pass
        try:
            self.sock.close()
        finally:
            self.sock = None

    def send_text(self, text: str) -> None:
        self._send_frame(0x1, text.encode("utf-8"))

    def receive_text(self) -> str:
        fragments: List[bytes] = []
        message_opcode: Optional[int] = None
        while True:
            fin, opcode, payload = self._receive_frame()
            if opcode == 0x8:
                raise ProbeFailure("Chrome closed the DevTools WebSocket")
            if opcode == 0x9:
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            if opcode in (0x1, 0x2):
                require(message_opcode is None, "Nested DevTools WebSocket message")
                message_opcode = opcode
                fragments = [payload]
            elif opcode == 0x0:
                require(message_opcode is not None, "Unexpected DevTools continuation frame")
                fragments.append(payload)
            else:
                continue
            if fin:
                require(message_opcode == 0x1, "Unexpected binary DevTools WebSocket message")
                return b"".join(fragments).decode("utf-8")

    def _read_exact(self, length: int) -> bytes:
        require(self.sock is not None, "DevTools WebSocket is not connected")
        while len(self.buffer) < length:
            chunk = self.sock.recv(max(4096, length - len(self.buffer)))
            if not chunk:
                raise ProbeFailure("Chrome closed the DevTools WebSocket")
            self.buffer.extend(chunk)
        result = bytes(self.buffer[:length])
        del self.buffer[:length]
        return result

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        require(self.sock is not None, "DevTools WebSocket is not connected")
        first = 0x80 | opcode
        length = len(payload)
        if length < 126:
            header = struct.pack("!BB", first, 0x80 | length)
        elif length < (1 << 16):
            header = struct.pack("!BBH", first, 0x80 | 126, length)
        else:
            header = struct.pack("!BBQ", first, 0x80 | 127, length)
        mask = secrets.token_bytes(4)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        self.sock.sendall(header + mask + masked)

    def _receive_frame(self) -> tuple[bool, int, bytes]:
        first, second = struct.unpack("!BB", self._read_exact(2))
        fin = bool(first & 0x80)
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        mask = self._read_exact(4) if masked else b""
        payload = self._read_exact(length)
        if masked:
            payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        return fin, opcode, payload


class CDP:
    def __init__(self, websocket_url: str, timeout: float) -> None:
        self.websocket = WebSocketClient(websocket_url, timeout)
        self.timeout = timeout
        self.next_id = 1

    def connect(self) -> None:
        self.websocket.connect()

    def close(self) -> None:
        self.websocket.close()

    def call(self, method: str, params: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        command_id = self.next_id
        self.next_id += 1
        self.websocket.send_text(
            json.dumps(
                {
                    "id": command_id,
                    "method": method,
                    "params": dict(params or {}),
                },
                separators=(",", ":"),
            )
        )
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            message = _strict_json(self.websocket.receive_text())
            if message.get("id") != command_id:
                continue
            if message.get("error"):
                raise ProbeFailure(f"CDP {method} failed: {message['error']}")
            return message.get("result") or {}
        raise ProbeFailure(f"Timed out waiting for CDP {method}")

    def evaluate(self, expression: str) -> Any:
        response = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": True,
                "returnByValue": True,
                "userGesture": True,
            },
        )
        if response.get("exceptionDetails"):
            details = response["exceptionDetails"]
            exception = details.get("exception") or {}
            description = exception.get("description") or details.get("text")
            raise ProbeFailure(f"Browser evaluation failed: {description}")
        return (response.get("result") or {}).get("value")

    def wait_for(self, expression: str, label: str, timeout: Optional[float] = None) -> Any:
        deadline = time.monotonic() + (timeout or self.timeout)
        last_value: Any = False
        last_error: Optional[BaseException] = None
        while time.monotonic() < deadline:
            try:
                last_value = self.evaluate(expression)
                if last_value:
                    return last_value
            except (OSError, ProbeFailure, socket.timeout) as exc:
                last_error = exc
            time.sleep(0.1)
        suffix = f"; last error={last_error}" if last_error else f"; last value={last_value!r}"
        raise ProbeFailure(f"Timed out waiting for {label}{suffix}")

    def dispatch_key(
        self,
        key: str,
        code: str,
        virtual_key: int,
        *,
        modifiers: int = 0,
    ) -> None:
        common = {
            "key": key,
            "code": code,
            "windowsVirtualKeyCode": virtual_key,
            "nativeVirtualKeyCode": virtual_key,
            "modifiers": modifiers,
        }
        self.call("Input.dispatchKeyEvent", {"type": "keyDown", **common})
        self.call("Input.dispatchKeyEvent", {"type": "keyUp", **common})


class ProbeEnvironment:
    def __init__(self, chrome: str, timeout: float) -> None:
        self.chrome = chrome
        self.timeout = timeout
        self.tempdir_context = tempfile.TemporaryDirectory(prefix="noema-ui-browser-probe-")
        self.tempdir = Path(self.tempdir_context.name)
        self.ui_port = free_loopback_port()
        self.devtools_port = free_loopback_port()
        self.ui_process: Optional[subprocess.Popen[bytes]] = None
        self.chrome_process: Optional[subprocess.Popen[bytes]] = None
        self.ui_log = self.tempdir / "ui-server.log"
        self.chrome_log = self.tempdir / "chrome.log"
        self.opener = no_proxy_opener()

    @property
    def ui_base_url(self) -> str:
        return f"http://127.0.0.1:{self.ui_port}"

    @property
    def devtools_base_url(self) -> str:
        return f"http://127.0.0.1:{self.devtools_port}"

    def start(self) -> None:
        workspace = self.tempdir / "workspace"
        workspace.mkdir()
        environment = dict(os.environ)
        source_path = str(ROOT / "src")
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            source_path
            if not existing_pythonpath
            else source_path + os.pathsep + existing_pythonpath
        )
        ui_command = [
            sys.executable,
            "-m",
            "noema_lab.cli.main",
            "--workspace",
            str(workspace),
            "ui",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.ui_port),
            "--project-root",
            str(ROOT),
        ]
        self.ui_process = self._start_process(
            ui_command,
            self.ui_log,
            environment=environment,
        )
        health = wait_for_http_json(
            self.opener,
            self.ui_base_url + "/api/health",
            "Noema UI server",
            timeout=self.timeout,
        )
        require(health.get("status") == "ok", f"Noema UI health check failed: {health}")

        profile = self.tempdir / "chrome-profile"
        chrome_command = [
            self.chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--disable-background-networking",
            "--disable-breakpad",
            "--disable-component-update",
            "--disable-default-apps",
            "--disable-extensions",
            "--disable-sync",
            "--metrics-recording-only",
            "--no-first-run",
            "--remote-allow-origins=*",
            "--remote-debugging-address=127.0.0.1",
            f"--remote-debugging-port={self.devtools_port}",
            f"--user-data-dir={profile}",
            "about:blank",
        ]
        self.chrome_process = self._start_process(chrome_command, self.chrome_log)
        wait_for_http_json(
            self.opener,
            self.devtools_base_url + "/json/version",
            "Chrome DevTools endpoint",
            timeout=self.timeout,
        )

    def stop(self) -> None:
        self._stop_process(self.chrome_process)
        self._stop_process(self.ui_process)
        self.chrome_process = None
        self.ui_process = None
        self.tempdir_context.cleanup()

    def diagnostics(self) -> str:
        sections = []
        for label, path in (("UI server", self.ui_log), ("Chrome", self.chrome_log)):
            if not path.exists():
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            sections.append(f"{label} log tail:\n" + "\n".join(lines[-40:]))
        return "\n\n".join(sections)

    @staticmethod
    def _start_process(
        command: List[str],
        log_path: Path,
        *,
        environment: Optional[Mapping[str, str]] = None,
    ) -> subprocess.Popen[bytes]:
        with log_path.open("wb") as stream:
            return subprocess.Popen(
                command,
                cwd=ROOT,
                env=dict(environment) if environment is not None else None,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )

    @staticmethod
    def _stop_process(process: Optional[subprocess.Popen[bytes]]) -> None:
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def probe_edit_run_results(cdp: CDP, name: str) -> Mapping[str, Any]:
    recipe_json = json.dumps(DEPENDENCY_LIGHT_RECIPE, sort_keys=True)
    cdp.evaluate(
        """
        (() => {
          const trigger = document.getElementById("rawRecipeEditorButton");
          trigger.focus();
          trigger.click();
        })()
        """
    )
    cdp.wait_for(
        """
        !document.getElementById("settingsOverlay").hidden &&
        Boolean(document.querySelector("[data-raw-recipe-text]"))
        """,
        f"{name} raw recipe editor open",
    )
    cdp.evaluate(
        """
        (() => {
          const textarea = document.querySelector("[data-raw-recipe-text]");
          textarea.value = %s;
          textarea.dispatchEvent(new Event("input", { bubbles: true }));
          document.querySelector("[data-raw-recipe-apply]").click();
        })()
        """ % json.dumps(recipe_json)
    )
    cdp.wait_for(
        """
        document.getElementById("settingsOverlay").hidden &&
        state.editRecipe && state.editRecipe.name === "ui_browser_probe_bits" &&
        !document.getElementById("runButton").disabled
        """,
        f"{name} raw recipe edit applied",
    )
    cdp.evaluate('document.getElementById("runButton").click()')
    try:
        cdp.wait_for(
            """
            !state.batchRunning &&
            state.selectedRecipe &&
            Boolean(state.runJobs[state.selectedRecipe.key]) &&
            isTerminalJobStatus(state.runJobs[state.selectedRecipe.key].status)
            """,
            f"{name} dependency-light run terminal state",
            timeout=max(20.0, cdp.timeout),
        )
    except ProbeFailure as exc:
        diagnostic = cdp.evaluate(
            """
            (() => {
              const recipe = state.selectedRecipe || {};
              const job = state.runJobs[recipe.key] || {};
              return {
                batchRunning: state.batchRunning,
                batchStopping: state.batchStopping,
                recipeKey: recipe.key || "",
                transientStatus: recipe.transientStatus || "",
                lastError: recipe.lastError || "",
                activeJobIds: state.activeJobIds,
                jobStatus: job.status || "",
                jobError: job.error || "",
                eventCount: (job.events || []).length,
                events: (job.events || []).slice(-8)
              };
            })()
            """
        )
        raise ProbeFailure(f"{exc}; browser run state={diagnostic}") from exc
    terminal = cdp.evaluate(
        """
        (() => {
          const recipe = state.selectedRecipe;
          const job = state.runJobs[recipe.key] || {};
          return {
            status: job.status || "",
            error: job.error || recipe.lastError || "",
            runId: recipe.lastRunId || job.run_id || "",
            events: (job.events || []).slice(-8).map((event) => ({
              seq: event.seq, kind: event.kind, message: event.message
            }))
          };
        })()
        """
    )
    require(
        terminal["status"] == "completed" and terminal["runId"],
        f"{name} dependency-light run failed: {terminal}",
    )
    cdp.wait_for(
        """
        Boolean(state.runSummaries[state.selectedRecipe.lastRunId]) &&
        state.runSummaries[state.selectedRecipe.lastRunId].status === "completed"
        """,
        f"{name} dependency-light run summary",
    )
    cdp.evaluate('setActiveView("results", true, "replace")')
    try:
        cdp.wait_for(
            """
            location.hash === "#results" &&
            document.getElementById("resultsSummaryLabel").textContent.includes("1 completed recipe shown in figures") &&
            Boolean(document.querySelector("[data-results-overview-figure], [data-domain-chart], [data-rd-chart]"))
            """,
            f"{name} completed evidence in results",
        )
    except ProbeFailure as exc:
        diagnostic = cdp.evaluate(
            """
            (() => {
              const rows = state.resultRows || [];
              const eligible = completedFigureResultRows(rows, { includeHidden: true });
              return {
                activeView: state.activeView,
                hash: location.hash,
                summaryLabel: document.getElementById("resultsSummaryLabel").textContent,
                rowCount: rows.length,
                rowStatuses: rows.map((row) => resultSummaries(row).map((summary) => summary.status)),
                eligibleCount: eligible.length,
                evidence: rows.map((row) => resultRowHasFigureEvidence(row)),
                resultText: document.getElementById("resultsComparison").innerText.slice(0, 800)
              };
            })()
            """
        )
        raise ProbeFailure(f"{exc}; results state={diagnostic}") from exc
    result = cdp.evaluate(
        """
        (() => {
          const recipe = state.selectedRecipe;
          const runId = recipe.lastRunId;
          const job = state.runJobs[recipe.key] || {};
          return {
            recipeName: state.editRecipe.name,
            runId,
            runStatus: state.runSummaries[runId].status,
            resultSummary: document.getElementById("resultsSummaryLabel").textContent,
            figureCount: document.querySelectorAll("[data-results-overview-figure]").length,
            eventCount: Array.isArray(job.events) ? job.events.length : 0,
            eventCursor: runJobEventCursor(job),
            activeView: state.activeView
          };
        })()
        """
    )
    require(result["eventCount"] > 0, f"{name} completed run has no job events: {result}")
    require(result["eventCursor"] > 0, f"{name} completed run did not advance its event cursor: {result}")
    return result


def probe_viewport(cdp: CDP, base_url: str, spec: Mapping[str, Any]) -> Mapping[str, Any]:
    name = str(spec["name"])
    width = int(spec["width"])
    height = int(spec["height"])
    cdp.call(
        "Emulation.setDeviceMetricsOverride",
        {
            "width": width,
            "height": height,
            "deviceScaleFactor": 1,
            "mobile": bool(spec["mobile"]),
            "screenWidth": width,
            "screenHeight": height,
        },
    )
    cdp.call("Network.setBlockedURLs", {"urls": []})
    cdp.call(
        "Page.navigate",
        {
            "url": (
                f"{base_url}/?probe={urllib.parse.quote(name)}-"
                f"{time.time_ns()}#graph"
            )
        },
    )
    cdp.wait_for(
        """
        document.readyState === "complete" &&
        document.querySelectorAll('.node[role="button"]').length > 0 &&
        document.getElementById("startupStatusBanner").hidden
        """,
        f"{name} application bootstrap",
    )

    layout = cdp.evaluate(
        """
        (() => {
          const ids = ["themeToggleButton", "saveRecipeButton", "logButton", "runButton"];
          const controls = Object.fromEntries(ids.map((id) => {
            const element = document.getElementById(id);
            const rect = element.getBoundingClientRect();
            const style = getComputedStyle(element);
            element.focus({ preventScroll: true });
            return [id, {
              visible:
                !element.hidden &&
                style.display !== "none" &&
                style.visibility !== "hidden" &&
                rect.width > 0 &&
                rect.height > 0,
              focusable: document.activeElement === element,
              disabled: Boolean(element.disabled),
              tabIndex: element.tabIndex,
              left: rect.left,
              right: rect.right,
              top: rect.top,
              bottom: rect.bottom,
              label:
                element.getAttribute("aria-label") ||
                element.textContent.trim()
            }];
          }));
          const brandRect = document.querySelector(".brand").getBoundingClientRect();
          const actionRect = document.querySelector(".topbar-actions").getBoundingClientRect();
          const brandActionsOverlap = !(
            brandRect.right <= actionRect.left ||
            actionRect.right <= brandRect.left ||
            brandRect.bottom <= actionRect.top ||
            actionRect.bottom <= brandRect.top
          );
          return {
            innerWidth,
            innerHeight,
            documentScrollWidth: document.documentElement.scrollWidth,
            bodyScrollWidth: document.body.scrollWidth,
            controls,
            brandActionsOverlap,
            brandRect: { left: brandRect.left, right: brandRect.right, top: brandRect.top, bottom: brandRect.bottom },
            actionRect: { left: actionRect.left, right: actionRect.right, top: actionRect.top, bottom: actionRect.bottom }
          };
        })()
        """
    )
    require(layout["innerWidth"] == width, f"{name} viewport width is {layout['innerWidth']}, expected {width}")
    for control_id, row in layout["controls"].items():
        require(row["visible"], f"{name} {control_id} is not visible")
        require(row["focusable"], f"{name} {control_id} cannot receive keyboard focus")
        require(not row["disabled"], f"{name} {control_id} is disabled")
        require(row["tabIndex"] >= 0, f"{name} {control_id} is outside the tab order")
        require(row["label"], f"{name} {control_id} has no accessible label")
        require(
            row["left"] >= -0.5 and row["right"] <= layout["innerWidth"] + 0.5,
            f"{name} {control_id} is horizontally clipped: {row}",
        )
        require(
            row["top"] >= -0.5 and row["bottom"] <= layout["innerHeight"] + 0.5,
            f"{name} {control_id} is vertically unreachable: {row}",
        )
    require(
        layout["documentScrollWidth"] <= layout["innerWidth"] + 1,
        f"{name} document has horizontal overflow: {layout}",
    )
    require(
        layout["bodyScrollWidth"] <= layout["innerWidth"] + 1,
        f"{name} body has horizontal overflow: {layout}",
    )
    require(
        not layout["brandActionsOverlap"],
        f"{name} brand overlaps top-bar actions: {layout}",
    )

    execution_controls = cdp.evaluate(
        """
        (() => {
          const selectors = [
            "[data-execution-parallel-workers]",
            "[data-execution-backend]",
            "[data-execution-implementation]",
            "[data-execution-strict-lint]",
            "[data-execution-use-plan-cache]"
          ];
          return selectors.map((selector) => {
            const element = document.querySelector(selector);
            if (!element) return { selector, exists: false };
            const label =
              element.getAttribute("aria-label") ||
              (element.closest("label") &&
                element.closest("label").innerText.trim()) ||
              "";
            const rect = element.getBoundingClientRect();
            const labelRect = element.closest("label").getBoundingClientRect();
            return {
              selector,
              exists: true,
              label,
              width: rect.width,
              height: rect.height,
              left: labelRect.left,
              right: labelRect.right,
              top: labelRect.top,
              bottom: labelRect.bottom
            };
          });
        })()
        """
    )
    for row in execution_controls:
        require(row["exists"], f"{name} missing execution control {row['selector']}")
        require(row["label"], f"{name} execution control lacks a label: {row['selector']}")
        require(
            row["width"] > 0 and row["height"] > 0,
            f"{name} execution control is not rendered: {row['selector']}",
        )
        require(
            row["left"] >= -0.5
            and row["right"] <= width + 0.5
            and row["top"] >= -0.5
            and row["bottom"] <= height + 0.5,
            f"{name} execution control is clipped or unreachable: {row}",
        )

    fresh_results = cdp.evaluate(
        """
        (() => {
          renderResultsDashboard(emptyCurrentRecipeResultRows());
          return document.getElementById("resultsSummaryLabel").textContent;
        })()
        """
    )
    require(
        fresh_results.startswith("0 completed recipes shown in figures")
        and "without completed figure evidence" in fresh_results,
        f"{name} fresh results claim unrun evidence is figure-ready: {fresh_results}",
    )

    cdp.evaluate(
        """
        (() => {
          const trigger = document.getElementById("recipeSettingsButton");
          trigger.focus();
          trigger.click();
        })()
        """
    )
    cdp.wait_for(
        '!document.getElementById("settingsOverlay").hidden',
        f"{name} settings modal open",
    )
    initial_focus = cdp.wait_for(
        """
        (() => {
          const overlay = document.getElementById("settingsOverlay");
          return overlay.contains(document.activeElement) &&
            (document.activeElement.id || document.activeElement.tagName);
        })()
        """,
        f"{name} modal initial focus",
    )
    cdp.evaluate(
        """
        (() => {
          const elements = modalFocusableElements(
            document.getElementById("settingsOverlay")
          );
          elements[elements.length - 1].focus();
        })()
        """
    )
    cdp.dispatch_key("Tab", "Tab", 9)
    wrapped_forward = cdp.evaluate(
        """
        (() => {
          const elements = modalFocusableElements(
            document.getElementById("settingsOverlay")
          );
          return document.activeElement === elements[0];
        })()
        """
    )
    require(wrapped_forward, f"{name} modal Tab did not wrap last to first")
    cdp.evaluate(
        """
        (() => {
          const elements = modalFocusableElements(
            document.getElementById("settingsOverlay")
          );
          elements[0].focus();
        })()
        """
    )
    cdp.dispatch_key("Tab", "Tab", 9, modifiers=8)
    wrapped_backward = cdp.evaluate(
        """
        (() => {
          const elements = modalFocusableElements(
            document.getElementById("settingsOverlay")
          );
          return document.activeElement === elements[elements.length - 1];
        })()
        """
    )
    require(wrapped_backward, f"{name} modal Shift+Tab did not wrap first to last")
    cdp.dispatch_key("Escape", "Escape", 27)
    cdp.wait_for(
        'document.getElementById("settingsOverlay").hidden',
        f"{name} settings modal close",
    )
    cdp.wait_for(
        'document.activeElement && document.activeElement.id === "recipeSettingsButton"',
        f"{name} modal focus restoration",
    )

    cdp.evaluate(
        """
        setActiveView("graph", true, "replace");
        document.getElementById("graphTabButton").focus();
        """
    )
    history_before = int(cdp.evaluate("history.length"))
    cdp.dispatch_key("ArrowRight", "ArrowRight", 39)
    cdp.wait_for(
        """
        location.hash === "#training" &&
        document.getElementById("trainingTabButton")
          .getAttribute("aria-selected") === "true" &&
        document.activeElement.id === "trainingTabButton"
        """,
        f"{name} ArrowRight tab navigation",
    )
    history_after = int(cdp.evaluate("history.length"))
    require(
        history_after == history_before + 1,
        f"{name} tab navigation did not push exactly one history entry: "
        f"{history_before} -> {history_after}",
    )
    cdp.evaluate("history.back()")
    cdp.wait_for(
        """
        location.hash === "#graph" &&
        document.getElementById("graphTabButton")
          .getAttribute("aria-selected") === "true"
        """,
        f"{name} tab history back",
    )
    cdp.evaluate("history.forward()")
    cdp.wait_for(
        """
        location.hash === "#training" &&
        document.getElementById("trainingTabButton")
          .getAttribute("aria-selected") === "true"
        """,
        f"{name} tab history forward",
    )
    cdp.evaluate('setActiveView("graph", true, "replace")')

    topology = cdp.evaluate(
        """
        (() => {
          const svg = document.getElementById("recipeSvg");
          const controls = [...svg.querySelectorAll("[data-graph-control]")];
          const nodes = [...svg.querySelectorAll('.node[role="button"]')];
          const outputs = [...svg.querySelectorAll('[data-node-output][role="button"]')];
          const inputs = [...svg.querySelectorAll('[data-node-input][role="button"]')];
          const edges = [...svg.querySelectorAll('.edge-hit-area[role="button"]')];
          return {
            nodes: nodes.length,
            outputs: outputs.length,
            inputs: inputs.length,
            edges: edges.length,
            compositeTabIndex: svg.tabIndex,
            childTabStops: controls.filter((control) => control.tabIndex >= 0).length,
            childProgrammaticStops: controls.filter((control) => control.tabIndex === -1).length,
            graphLabel: svg.getAttribute("aria-label") || "",
            maxNodeLabelLength: Math.max(...nodes.map((node) => (node.getAttribute("aria-label") || "").length)),
            maxControlLabelLength: Math.max(...controls.map((control) => (control.getAttribute("aria-label") || "").length))
          };
        })()
        """
    )
    require(topology["nodes"] > 0, f"{name} graph has no keyboard nodes")
    require(topology["outputs"] > 0, f"{name} graph has no keyboard output ports")
    require(topology["inputs"] > 0, f"{name} graph has no keyboard input ports")
    require(topology["edges"] > 0, f"{name} graph has no keyboard edges")
    require(topology["compositeTabIndex"] == 0, f"{name} graph composite is not in the tab order: {topology}")
    require(topology["childTabStops"] == 0, f"{name} graph exposes redundant normal tab stops: {topology}")
    require(
        topology["childProgrammaticStops"]
        == topology["nodes"] + topology["outputs"] + topology["inputs"] + topology["edges"],
        f"{name} graph controls are not available through composite navigation: {topology}",
    )
    require("Tab to skip" in topology["graphLabel"], f"{name} graph lacks skip guidance: {topology}")
    require(topology["maxNodeLabelLength"] <= 160, f"{name} graph node label is too verbose: {topology}")
    require(topology["maxControlLabelLength"] <= 240, f"{name} graph control label is too verbose: {topology}")

    cdp.evaluate('document.getElementById("recipeSvg").focus()')
    cdp.dispatch_key("ArrowRight", "ArrowRight", 39)
    cdp.wait_for(
        'Boolean(document.activeElement && document.activeElement.matches(\'.node[role="button"]\'))',
        f"{name} graph composite ArrowRight navigation",
    )
    cdp.dispatch_key("Enter", "Enter", 13)
    cdp.wait_for(
        'Boolean(document.querySelector(\'.node[aria-pressed="true"]\'))',
        f"{name} node keyboard activation",
    )
    cdp.evaluate(
        'document.querySelector(\'.edge-hit-area[role="button"]\').focus()'
    )
    cdp.dispatch_key("Enter", "Enter", 13)
    cdp.wait_for(
        """
        Boolean(
          document.querySelector('.edge-hit-area[aria-pressed="true"]')
        )
        """,
        f"{name} edge keyboard activation",
    )

    wheel_behavior = cdp.evaluate(
        """
        (() => {
          const surface = document.getElementById("graphSurface");
          const before = state.graphZoom;
          const ordinary = new WheelEvent("wheel", { deltaY: -120, bubbles: true, cancelable: true });
          surface.dispatchEvent(ordinary);
          const afterOrdinary = state.graphZoom;
          const modified = new WheelEvent("wheel", { deltaY: -120, altKey: true, bubbles: true, cancelable: true });
          surface.dispatchEvent(modified);
          return {
            before,
            afterOrdinary,
            afterModified: state.graphZoom,
            ordinaryPrevented: ordinary.defaultPrevented,
            modifiedPrevented: modified.defaultPrevented
          };
        })()
        """
    )
    require(
        wheel_behavior["before"] == wheel_behavior["afterOrdinary"]
        and not wheel_behavior["ordinaryPrevented"],
        f"{name} ordinary wheel scrolling is hijacked: {wheel_behavior}",
    )
    require(
        wheel_behavior["afterModified"] != wheel_behavior["afterOrdinary"]
        and wheel_behavior["modifiedPrevented"],
        f"{name} Alt+wheel graph zoom is unavailable: {wheel_behavior}",
    )

    port_candidate = cdp.evaluate(
        """
        (() => {
          const candidates = [
            ...document.querySelectorAll(
              '[data-node-output][role="button"]'
            )
          ].map((port) => ({
            step: port.getAttribute("data-node-output"),
            output: port.getAttribute("data-port-name"),
            kind: port.getAttribute("data-port-kind")
          }));
          for (const candidate of candidates) {
            startPendingGraphLink(
              candidate.step,
              candidate.output,
              candidate.kind
            );
            if (document.querySelector(".node-input-port.compatible")) {
              window.__noemaProbePortCandidate = candidate;
              cancelPendingGraphLink();
              return candidate;
            }
            cancelPendingGraphLink();
          }
          return null;
        })()
        """
    )
    require(port_candidate, f"{name} graph has no keyboard-connectable port pair")
    cdp.evaluate(
        """
        (() => {
          const candidate = window.__noemaProbePortCandidate;
          const port = [
            ...document.querySelectorAll('[data-node-output][role="button"]')
          ].find((item) =>
            item.getAttribute("data-node-output") === candidate.step &&
            item.getAttribute("data-port-name") === candidate.output
          );
          port.focus();
        })()
        """
    )
    cdp.dispatch_key("Enter", "Enter", 13)
    cdp.wait_for(
        """
        Boolean(document.getElementById("pendingGraphEdge")) &&
        Boolean(document.querySelector(".node-input-port.compatible"))
        """,
        f"{name} output port keyboard activation",
    )
    input_label = cdp.evaluate(
        """
        (() => {
          const input = document.querySelector(
            '.node-input-port.compatible[role="button"]'
          );
          input.focus();
          return input.getAttribute("aria-label");
        })()
        """
    )
    require(input_label, f"{name} compatible input port has no accessible label")
    cdp.dispatch_key("Enter", "Enter", 13)
    cdp.wait_for(
        """
        !document.getElementById("pendingGraphEdge") &&
        !document.querySelector(".node-output-port.active")
        """,
        f"{name} input port keyboard activation",
    )

    cdp.call("Network.setBlockedURLs", {"urls": ["*api/health*"]})
    cdp.call(
        "Page.navigate",
        {
            "url": (
                f"{base_url}/?probe=failure-{urllib.parse.quote(name)}-"
                f"{time.time_ns()}#graph"
            )
        },
    )
    cdp.wait_for(
        """
        document.readyState === "complete" &&
        !document.getElementById("startupStatusBanner").hidden
        """,
        f"{name} durable startup failure banner",
    )
    failure_before = cdp.evaluate(
        """
        (() => {
          const banner = document.getElementById("startupStatusBanner");
          const button = document.getElementById("startupRetryButton");
          const rect = banner.getBoundingClientRect();
          return {
            role: banner.getAttribute("role"),
            live: banner.getAttribute("aria-live"),
            message:
              document.getElementById("startupStatusMessage").textContent,
            buttonText: button.textContent.trim(),
            buttonDisabled: button.disabled,
            buttonTag: button.tagName,
            buttonTabIndex: button.tabIndex,
            left: rect.left,
            right: rect.right,
            viewport: innerWidth
          };
        })()
        """
    )
    failure_message = str(failure_before["message"]).lower()
    require(
        failure_before["role"] == "alert" and failure_before["live"] == "assertive",
        f"{name} startup failure is not an assertive alert: {failure_before}",
    )
    require(
        "server" in failure_message and "retry" in failure_message,
        f"{name} startup failure is not actionable: {failure_before}",
    )
    require(
        failure_before["buttonTag"] == "BUTTON"
        and failure_before["buttonText"] == "Retry"
        and not failure_before["buttonDisabled"]
        and failure_before["buttonTabIndex"] >= 0,
        f"{name} startup Retry is not keyboard available: {failure_before}",
    )
    require(
        failure_before["left"] >= -0.5
        and failure_before["right"] <= failure_before["viewport"] + 0.5,
        f"{name} startup failure banner is horizontally clipped: {failure_before}",
    )
    persistence_ms = 3300
    time.sleep(persistence_ms / 1000)
    failure_after = cdp.evaluate(
        """
        (() => {
          const banner = document.getElementById("startupStatusBanner");
          return {
            visible: !banner.hidden,
            message:
              document.getElementById("startupStatusMessage").textContent
          };
        })()
        """
    )
    require(
        failure_after["visible"]
        and failure_after["message"] == failure_before["message"],
        f"{name} startup failure was transient: {failure_after}",
    )

    cdp.call("Network.setBlockedURLs", {"urls": []})
    unblocked_health = cdp.evaluate(
        """
        fetch("/api/health")
          .then((response) => ({ ok: response.ok, status: response.status }))
          .catch((error) => ({ ok: false, error: String(error) }))
        """
    )
    require(
        unblocked_health["ok"],
        f"{name} health API remained blocked before Retry: {unblocked_health}",
    )
    cdp.evaluate('document.getElementById("startupRetryButton").focus()')
    cdp.dispatch_key(" ", "Space", 32)
    cdp.wait_for(
        """
        document.getElementById("startupStatusBanner").hidden &&
        document.querySelectorAll('.node[role="button"]').length > 0
        """,
        f"{name} keyboard Retry recovery",
    )

    edit_run_results = None
    if bool(spec.get("exercise_run")):
        edit_run_results = probe_edit_run_results(cdp, name)

    return {
        "viewport": f"{width}x{height}",
        "top_controls": {
            key: {
                "label": value["label"],
                "left": value["left"],
                "right": value["right"],
            }
            for key, value in layout["controls"].items()
        },
        "no_horizontal_overflow": True,
        "brand_actions_separate": True,
        "execution_control_count": len(execution_controls),
        "execution_controls_in_viewport": True,
        "modal": {
            "initial_focus": initial_focus,
            "wrapped_forward": bool(wrapped_forward),
            "wrapped_backward": bool(wrapped_backward),
            "restored_to": "recipeSettingsButton",
        },
        "tabs": {
            "arrow_right": True,
            "history_back": True,
            "history_forward": True,
            "history_before": history_before,
            "history_after": history_after,
        },
        "topology": topology,
        "wheel_zoom": wheel_behavior,
        "keyboard_input_port": input_label,
        "startup_failure": {
            "role": failure_before["role"],
            "durable_after_ms": persistence_ms,
            "keyboard_retry_recovered": True,
        },
        "fresh_results": fresh_results,
        "edit_run_results": edit_run_results,
    }


def find_chrome(explicit: str) -> str:
    if explicit:
        resolved = shutil.which(explicit)
        if resolved:
            return resolved
        candidate = Path(explicit)
        if candidate.is_file():
            return str(candidate.resolve())
        raise ProbeFailure(f"Chrome executable not found: {explicit}")
    environment_candidate = os.environ.get("CHROME_BIN", "")
    candidates = [
        environment_candidate,
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise ProbeFailure(
        "No Chrome/Chromium executable found; pass --chrome or set CHROME_BIN"
    )


def run(chrome: str, timeout: float) -> Mapping[str, Any]:
    environment = ProbeEnvironment(chrome, timeout)
    cdp: Optional[CDP] = None
    target_id = ""
    try:
        environment.start()
        target = http_json(
            environment.opener,
            environment.devtools_base_url + "/json/new?about%3Ablank",
            method="PUT",
        )
        target_id = str(target["id"])
        cdp = CDP(str(target["webSocketDebuggerUrl"]), timeout)
        cdp.connect()
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call("Network.enable")
        cdp.call("Network.setCacheDisabled", {"cacheDisabled": True})
        results = [
            probe_viewport(
                cdp,
                environment.ui_base_url,
                {
                    "name": "desktop",
                    "width": 1440,
                    "height": 900,
                    "mobile": False,
                    "exercise_run": True,
                },
            ),
            probe_viewport(
                cdp,
                environment.ui_base_url,
                {
                    "name": "mobile-360",
                    "width": 360,
                    "height": 800,
                    "mobile": True,
                },
            ),
            probe_viewport(
                cdp,
                environment.ui_base_url,
                {
                    "name": "mobile-375",
                    "width": 375,
                    "height": 812,
                    "mobile": True,
                },
            ),
        ]
        return {
            "status": "passed",
            "chrome": chrome,
            "results": results,
        }
    except Exception:
        diagnostics = environment.diagnostics()
        if diagnostics:
            print(diagnostics, file=sys.stderr)
        raise
    finally:
        if cdp is not None:
            try:
                cdp.call("Network.setBlockedURLs", {"urls": []})
            except Exception:
                pass
            cdp.close()
        if target_id:
            try:
                http_bytes(
                    environment.opener,
                    environment.devtools_base_url + f"/json/close/{target_id}",
                    method="PUT",
                )
            except Exception:
                pass
        environment.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chrome",
        default="",
        help="Chrome/Chromium executable; otherwise CHROME_BIN and common names are tried.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-startup and browser assertion timeout in seconds.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        chrome = find_chrome(args.chrome)
        payload = run(chrome, max(5.0, float(args.timeout)))
    except Exception as exc:
        print(f"UI browser probe failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
