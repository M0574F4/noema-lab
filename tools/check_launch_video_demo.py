#!/usr/bin/env python3
"""Preflight the source checkout before recording Noema's flagship workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import socket
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = Path(".noema/training_exports/qpsk_iq_calibration_video")
DEFAULT_VENV = Path(".venv")
MINIMUM_FREE_BYTES = 8 * 1024**3
REQUIRED_PATHS = (
    Path("pyproject.toml"),
    Path("recipes/neural_receiver_qpsk_iq_calibration.yaml"),
    Path("demo_trainings/prepare_example.py"),
    Path("demo_trainings/neural_receiver_supervised_qpsk/training_plan.yaml"),
    Path("demo_trainings/neural_receiver_supervised_qpsk/requirements.txt"),
    Path("docs/break_the_comparison.md"),
)


def _available_command(candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return ""


def _port_availability(port: int) -> tuple[bool | None, str]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", int(port)))
    except PermissionError as exc:
        return None, str(exc)
    except OSError as exc:
        return False, str(exc)
    return True, ""


def _format_bytes(value: int) -> str:
    return "%.1f GiB" % (float(value) / 1024**3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        default=str(DEFAULT_BUNDLE),
        help="Bundle directory that must be absent at the start of the take.",
    )
    parser.add_argument(
        "--venv",
        default=str(DEFAULT_VENV),
        help="Virtual environment directory that must be absent at the start of the take.",
    )
    parser.add_argument("--port", type=int, default=8766, help="Noema UI port.")
    parser.add_argument(
        "--allow-existing-bundle",
        action="store_true",
        help="Report an existing bundle without failing (useful after a rehearsal).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    failures: list[str] = []
    warnings: list[str] = []
    passes: list[str] = []

    if (3, 11) <= sys.version_info[:2] < (3, 14):
        passes.append("Python %d.%d is supported" % sys.version_info[:2])
    else:
        failures.append("use Python 3.11, 3.12, or 3.13")

    uv = _available_command(("uv",))
    if uv:
        passes.append("uv found at %s" % uv)
    else:
        failures.append("uv is not installed or is not on PATH")

    missing = [str(path) for path in REQUIRED_PATHS if not (ROOT / path).is_file()]
    if missing:
        failures.append("required source files are missing: %s" % ", ".join(missing))
    else:
        passes.append("flagship recipe, trainer, requirements, and web demo are present")

    bundle = Path(args.bundle).expanduser()
    if not bundle.is_absolute():
        bundle = ROOT / bundle
    if bundle.exists():
        message = "recording bundle already exists: %s" % bundle
        if args.allow_existing_bundle:
            warnings.append(message)
        else:
            failures.append(message + " (choose a new bundle name or move it aside)")
    else:
        passes.append("recording bundle path is unused")

    venv = Path(args.venv).expanduser()
    if not venv.is_absolute():
        venv = ROOT / venv
    if venv.exists():
        failures.append(
            "recording virtual environment already exists: %s "
            "(choose a new environment name or move it aside)" % venv.resolve()
        )
    else:
        passes.append("recording virtual-environment path is unused")

    if not 1 <= int(args.port) <= 65535:
        failures.append("UI port must be between 1 and 65535")
    else:
        port_available, port_issue = _port_availability(args.port)
        if port_available:
            passes.append("127.0.0.1:%d is available" % args.port)
        elif port_available is None:
            warnings.append(
                "could not probe 127.0.0.1:%d in this restricted shell: %s"
                % (args.port, port_issue)
            )
        else:
            failures.append(
                "127.0.0.1:%d is unavailable: %s" % (args.port, port_issue)
            )

    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes >= MINIMUM_FREE_BYTES:
        passes.append("free disk space: %s" % _format_bytes(free_bytes))
    else:
        failures.append(
            "only %s is free; reserve at least %s for PyTorch, captures, and video"
            % (_format_bytes(free_bytes), _format_bytes(MINIMUM_FREE_BYTES))
        )

    browser = _available_command(
        ("google-chrome", "chromium", "chromium-browser", "firefox", "open", "xdg-open")
    )
    if browser:
        passes.append("browser/open command found at %s" % browser)
    else:
        failures.append("no browser or operating-system open command was found")

    recorder = _available_command(("obs", "obs-studio"))
    if recorder:
        passes.append("OBS Studio found at %s" % recorder)
    else:
        warnings.append("OBS Studio was not found; install it before the recording")

    for message in passes:
        print("PASS  %s" % message)
    for message in warnings:
        print("WARN  %s" % message)
    for message in failures:
        print("FAIL  %s" % message)
    if failures:
        print("\nLaunch-video preflight failed with %d blocking issue(s)." % len(failures))
        return 1
    print("\nLaunch-video source preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
