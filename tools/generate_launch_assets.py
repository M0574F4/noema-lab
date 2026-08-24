#!/usr/bin/env python3
"""Generate Noema's F0–F5 launch figures and canonical launch tables."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import html
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_RELATIVE = Path("launch_evidence.json")
GENERATOR_RELATIVE = Path("tools/generate_launch_assets.py")
SCHEMA_RELATIVE = Path("schemas/launch_asset_manifest.schema.json")
OUTPUT_RELATIVE = Path("docs/_static/launch")

FIGURES = (
    ("F0", "f0-launch-hero.svg", "Make the comparison traceable"),
    ("F1", "f1-comparison-anatomy.svg", "The anatomy of a defensible comparison"),
    ("F2", "f2-contract-to-evidence.svg", "From experiment contract to retained evidence"),
    ("F3", "f3-receiver-ber-vs-snr.svg", "Receiver BER versus SNR"),
    ("F4", "f4-break-the-comparison.svg", "Four ways to break the comparison"),
    ("F5", "f5-one-source-many-surfaces.svg", "One evidence source, many public outputs"),
)
TABLES = (
    ("T0", "tables/t0-result-summary.csv", "Paired BER result summary"),
    ("T0", "tables/t0-result-summary.md", "Paired BER result summary"),
    ("T1", "tables/t1-experiment-contract.csv", "Experiment contract summary"),
    ("T1", "tables/t1-experiment-contract.md", "Experiment contract summary"),
    ("T2", "tables/t2-claim-guardrails.csv", "Launch claim guardrails"),
    ("T2", "tables/t2-claim-guardrails.md", "Launch claim guardrails"),
)

BACKGROUND = "#07110f"
PANEL = "#0d211c"
PANEL_ALT = "#102a23"
INK = "#effbf6"
MUTED = "#a9c1b8"
GREEN = "#5de2a5"
GREEN_DEEP = "#22966a"
TEAL = "#4ec5c1"
ORANGE = "#f08345"
AMBER = "#ffc857"
RED = "#ff6b6b"
GRAY = "#8fa39b"
LINE = "#27443a"


class LaunchAssetError(ValueError):
    """Raised when source evidence or a generated launch asset is invalid."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise LaunchAssetError(message)


def _digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _file_digest(path: Path) -> str:
    return _digest(path.read_bytes())


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _text(
    x: float,
    y: float,
    value: Any,
    *,
    size: int = 28,
    weight: int = 500,
    fill: str = INK,
    anchor: str = "start",
    family: str = "Inter,Segoe UI,Arial,sans-serif",
    tracking: float | None = None,
) -> str:
    spacing = "" if tracking is None else f' letter-spacing="{tracking:g}"'
    return (
        f'<text x="{x:g}" y="{y:g}" fill="{fill}" font-family="{family}" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"'
        f'{spacing}>{_escape(value)}</text>'
    )


def _multiline(
    x: float,
    y: float,
    lines: Iterable[str],
    *,
    size: int = 28,
    weight: int = 500,
    fill: str = INK,
    line_height: float = 1.25,
    anchor: str = "start",
) -> str:
    tspans = []
    for index, line in enumerate(lines):
        dy = "0" if index == 0 else f"{size * line_height:g}"
        tspans.append(f'<tspan x="{x:g}" dy="{dy}">{_escape(line)}</tspan>')
    return (
        f'<text x="{x:g}" y="{y:g}" fill="{fill}" '
        f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}">'
        + "".join(tspans)
        + "</text>"
    )


def _rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str = PANEL,
    stroke: str = LINE,
    radius: float = 24,
    stroke_width: float = 2,
) -> str:
    return (
        f'<rect x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" '
        f'rx="{radius:g}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width:g}"/>'
    )


def _pill(x: float, y: float, width: float, label: str, *, color: str = GREEN) -> str:
    return "".join(
        (
            _rect(x, y, width, 46, fill="#0a1b17", stroke=color, radius=23),
            _text(x + width / 2, y + 30, label, size=18, weight=700, fill=color, anchor="middle"),
        )
    )


def _arrow(x1: float, y1: float, x2: float, y2: float, *, color: str = GREEN) -> str:
    marker = {
        RED: "arrow-red",
        AMBER: "arrow-amber",
        TEAL: "arrow-teal",
    }.get(color, "arrow")
    return (
        f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
        f'stroke="{color}" stroke-width="4" marker-end="url(#{marker})"/>'
    )


def _svg(title: str, description: str, body: str, *, width: int = 1600, height: int = 900) -> bytes:
    payload = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{_escape(title)}</title>
  <desc id="desc">{_escape(description)}</desc>
  <defs>
    <linearGradient id="hero" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#0b211b"/>
      <stop offset="1" stop-color="#06100e"/>
    </linearGradient>
    <radialGradient id="glow" cx="82%" cy="8%" r="72%">
      <stop offset="0" stop-color="#2fb47c" stop-opacity=".34"/>
      <stop offset="1" stop-color="#07110f" stop-opacity="0"/>
    </radialGradient>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{GREEN}"/>
    </marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{RED}"/>
    </marker>
    <marker id="arrow-amber" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{AMBER}"/>
    </marker>
    <marker id="arrow-teal" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{TEAL}"/>
    </marker>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="16" stdDeviation="22" flood-color="#000" flood-opacity=".28"/>
    </filter>
  </defs>
  <rect width="{width}" height="{height}" fill="{BACKGROUND}"/>
  <rect width="{width}" height="{height}" fill="url(#glow)"/>
  {body}
</svg>
'''
    return payload.encode("utf-8")


def _series_by_id(evidence: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["id"]): item for item in evidence["series"]}


def _figure_f0(evidence: Mapping[str, Any]) -> bytes:
    headline = evidence["headline"]
    design = evidence["design"]
    percent = f'{headline["primary_learned_reduction_percent"]:.2f}%'
    snr = f'{headline["primary_snr_db"]:g} dB'
    body = "".join(
        (
            _text(92, 92, "NOEMA · EXECUTABLE EXPERIMENT CONTRACTS", size=20, weight=750, fill=GREEN, tracking=2.8),
            _multiline(92, 220, ("Make the comparison", "traceable."), size=88, weight=760, line_height=1.02),
            _multiline(
                96,
                455,
                (
                    "Bind the protocol, execution, resource tracking, returned model,",
                    "terminal outcome, and retained evidence into one verifiable object.",
                ),
                size=27,
                fill=MUTED,
                line_height=1.45,
            ),
            _pill(96, 590, 230, f'{design["run_count"]} retained runs'),
            _pill(344, 590, 210, f'{len(design["snr_db"])} SNR cells'),
            _pill(572, 590, 220, f'{len(design["paired_seeds"])} paired seeds'),
            _rect(1010, 145, 480, 510, fill="#0b201a", stroke=GREEN_DEEP, radius=34, stroke_width=3),
            _text(1065, 210, "COMPLETED EXPERIMENTAL EVIDENCE", size=17, weight=750, fill=GREEN, tracking=1.8),
            _text(1065, 350, percent, size=104, weight=780, fill=INK),
            _text(1068, 397, f"BER reduction vs uncompensated · {snr}", size=20, weight=650, fill=MUTED),
            '<line x1="1065" y1="445" x2="1430" y2="445" stroke="#27443a" stroke-width="2"/>',
            _multiline(
                1065,
                500,
                (
                    "One synthetic QPSK demonstration.",
                    "Observed min–max, not confidence intervals.",
                    "Diagnostic oracle ≠ deployable competitor.",
                ),
                size=20,
                fill=MUTED,
                line_height=1.65,
            ),
        )
    )
    return _svg(
        "Noema: make the comparison traceable",
        "Launch hero showing Noema's contract-first claim and a bounded experimental QPSK receiver result.",
        body,
        height=700,
    )


def _figure_f1(evidence: Mapping[str, Any]) -> bytes:
    design = evidence["design"]
    cards = (
        ("01", "Condition", f'{len(design["snr_db"])} predeclared SNR cells', TEAL),
        ("02", "Pairing", f'{len(design["paired_seeds"])} held-out seeds per cell', GREEN),
        ("03", "Aggregation", "Same arithmetic mean", AMBER),
        ("04", "Metric", "Pre-decoder BER / same bits", TEAL),
        ("05", "Role", "Baseline · reference · candidate", GREEN),
    )
    body = [
        _text(80, 82, "COMPARISON ANATOMY", size=19, weight=750, fill=GREEN, tracking=2.5),
        _text(80, 155, "A percentage is the end of the contract—not the start.", size=46, weight=740),
        _text(80, 205, "A defensible claim survives only while every comparison boundary remains aligned.", size=24, fill=MUTED),
    ]
    x_positions = (80, 382, 684, 986, 1288)
    for (number, title, detail, color), x in zip(cards, x_positions):
        body.extend(
            (
                _rect(x, 310, 232, 250, fill=PANEL, stroke=color, radius=24),
                _text(x + 24, 360, number, size=18, weight=750, fill=color),
                _text(x + 24, 425, title, size=29, weight=730),
                _multiline(x + 24, 475, _wrap(detail, 22), size=19, fill=MUTED, line_height=1.35),
                _arrow(x + 116, 565, 800, 660, color=color),
            )
        )
    body.extend(
        (
            _rect(430, 660, 740, 150, fill="#102a23", stroke=GREEN, radius=30, stroke_width=3),
            _text(800, 713, "CLAIM THAT SURVIVES THE AUDIT", size=18, weight=750, fill=GREEN, anchor="middle", tracking=1.8),
            _text(800, 772, evidence["headline"]["text"], size=25, weight=650, anchor="middle"),
            _text(800, 860, evidence["scientific_status"]["disclosure"], size=17, fill=AMBER, anchor="middle"),
        )
    )
    return _svg(
        "The anatomy of a defensible comparison",
        "Five aligned boundaries—condition, pairing, aggregation, metric, and role—feed one bounded quantitative claim.",
        "".join(body),
    )


def _figure_f2(evidence: Mapping[str, Any]) -> bytes:
    design = evidence["design"]
    stages = (
        ("01", "Protocol", "Typed methods and SNR grid"),
        ("02", "Execution", f'{design["run_count"]} concrete runs'),
        ("03", "Metrics", "Bit and block error counts"),
        ("04", "Evidence", "Runs, identities, observed ranges"),
        ("05", "Projection", "launch_evidence.json"),
    )
    body = [
        _text(80, 82, "CONTRACT → EVIDENCE", size=19, weight=750, fill=GREEN, tracking=2.5),
        _text(80, 155, "One chain from declared question to retained result.", size=48, weight=740),
        _text(80, 205, "Noema links the declared comparison settings to every run and result.", size=24, fill=MUTED),
    ]
    x_positions = (70, 380, 690, 1000, 1310)
    for index, ((number, title, detail), x) in enumerate(zip(stages, x_positions)):
        body.extend(
            (
                _rect(x, 335, 220, 250, fill=PANEL, stroke=GREEN_DEEP if index < 4 else GREEN, radius=25),
                _text(x + 24, 382, number, size=17, weight=750, fill=GREEN),
                _text(x + 24, 445, title, size=29, weight=730),
                _multiline(x + 24, 495, _wrap(detail, 19), size=18, fill=MUTED, line_height=1.4),
            )
        )
        if index < len(stages) - 1:
            body.append(_arrow(x + 230, 460, x_positions[index + 1] - 18, 460))
    body.extend(
        (
            _rect(210, 690, 1180, 118, fill="#0a1b17", stroke=LINE, radius=22),
            _text(250, 738, "LOCAL VERIFICATION", size=16, weight=750, fill=GREEN, tracking=1.5),
            _text(250, 780, "Checks declared settings, run identities, and result links.", size=22, fill=MUTED),
        )
    )
    return _svg(
        "From experiment contract to retained evidence",
        "Five-stage Noema evidence chain from a typed protocol through execution and metrics to a generated launch projection.",
        "".join(body),
    )


def _plot_y(value: float, top: float, bottom: float) -> float:
    lower = math.log10(5e-4)
    upper = math.log10(3e-1)
    position = (upper - math.log10(value)) / (upper - lower)
    return top + position * (bottom - top)


def _figure_f3(evidence: Mapping[str, Any]) -> bytes:
    series = _series_by_id(evidence)
    snrs = [float(value) for value in evidence["design"]["snr_db"]]
    left, right, top, bottom = 155.0, 1460.0, 160.0, 690.0

    def x_for(snr: float) -> float:
        return left + (snr - snrs[0]) / (snrs[-1] - snrs[0]) * (right - left)

    body = [
        _text(70, 66, "CANONICAL EXPERIMENT FIGURE", size=18, weight=750, fill=GREEN, tracking=2.2),
        _text(70, 124, "Learned receiver tracks the calibrated diagnostic reference", size=42, weight=740),
    ]
    ticks = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
    for tick in ticks:
        y = _plot_y(tick, top, bottom)
        body.extend(
            (
                f'<line x1="{left:g}" y1="{y:g}" x2="{right:g}" y2="{y:g}" stroke="{LINE}" stroke-width="1.5"/>',
                _text(left - 20, y + 7, f"{tick:g}", size=17, fill=MUTED, anchor="end", family="ui-monospace,SFMono-Regular,Consolas,monospace"),
            )
        )
    for snr in snrs:
        x = x_for(snr)
        body.extend(
            (
                f'<line x1="{x:g}" y1="{top:g}" x2="{x:g}" y2="{bottom:g}" stroke="{LINE}" stroke-width="1" stroke-opacity=".55"/>',
                _text(x, bottom + 36, f"{snr:g}", size=18, fill=MUTED, anchor="middle"),
            )
        )
    body.extend(
        (
            f'<line x1="{left:g}" y1="{bottom:g}" x2="{right:g}" y2="{bottom:g}" stroke="{GRAY}" stroke-width="2"/>',
            f'<line x1="{left:g}" y1="{top:g}" x2="{left:g}" y2="{bottom:g}" stroke="{GRAY}" stroke-width="2"/>',
            _text((left + right) / 2, 778, "SNR (dB)", size=22, weight=650, anchor="middle"),
            f'<text transform="translate(48 450) rotate(-90)" fill="{INK}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="22" font-weight="650" text-anchor="middle">Pre-decoder bit error rate · log scale</text>',
        )
    )

    learned_points = series["learned_receiver"]["points"]
    upper_band = [
        (x_for(float(point["snr_db"])), _plot_y(float(point["summary"]["maximum_ber"]), top, bottom))
        for point in learned_points
    ]
    lower_band = [
        (x_for(float(point["snr_db"])), _plot_y(float(point["summary"]["minimum_ber"]), top, bottom))
        for point in reversed(learned_points)
    ]
    band = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper_band + lower_band)
    body.append(f'<polygon points="{band}" fill="{GREEN}" fill-opacity=".18" stroke="none"/>')

    styles = {
        "uncompensated_qpsk": (GRAY, "8 7", "square"),
        "calibrated_iq_oracle": (ORANGE, "12 9", "triangle"),
        "learned_receiver": (GREEN, "", "circle"),
    }
    # Draw the dashed oracle last so the nearly coincident learned curve remains
    # distinguishable instead of covering the diagnostic reference.
    for method in ("uncompensated_qpsk", "learned_receiver", "calibrated_iq_oracle"):
        color, dash, marker = styles[method]
        points = [
            (
                x_for(float(point["snr_db"])),
                _plot_y(float(point["summary"]["mean_ber"]), top, bottom),
            )
            for point in series[method]["points"]
        ]
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dash_attr = "" if not dash else f' stroke-dasharray="{dash}"'
        body.append(
            f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="5" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash_attr}/>'
        )
        for x, y in points:
            if marker == "square":
                body.append(f'<rect x="{x - 6:g}" y="{y - 6:g}" width="12" height="12" fill="{BACKGROUND}" stroke="{color}" stroke-width="4"/>')
            elif marker == "triangle":
                body.append(f'<path d="M {x:g} {y - 8:g} L {x + 8:g} {y + 7:g} L {x - 8:g} {y + 7:g} Z" fill="none" stroke="{color}" stroke-width="4"/>')
            else:
                body.append(f'<circle cx="{x:g}" cy="{y:g}" r="6" fill="{BACKGROUND}" stroke="{color}" stroke-width="4"/>')

    legend_items = (
        (GRAY, "Uncompensated QPSK", "dashed"),
        (ORANGE, "Calibrated I/Q oracle · diagnostic", "dashed"),
        (GREEN, "Learned I/Q receiver · observed min–max band", "solid"),
    )
    for index, (color, label, style) in enumerate(legend_items):
        x = 230 + index * 430
        dash = ' stroke-dasharray="8 7"' if style == "dashed" else ""
        body.extend(
            (
                f'<line x1="{x:g}" y1="834" x2="{x + 55:g}" y2="834" stroke="{color}" stroke-width="5"{dash}/>',
                _text(x + 70, 841, label, size=17, fill=MUTED),
            )
        )
    body.append(
        _text(
            800,
            884,
            "Completed experimental benchmark · three paired seeds per cell · "
            "bands are observed min/max, not confidence intervals",
            size=16,
            fill=AMBER,
            anchor="middle",
        )
    )
    return _svg(
        "Receiver BER versus SNR",
        evidence["headline"]["accessible_summary"] + " Bands show observed minima and maxima across three paired seeds, not confidence intervals.",
        "".join(body),
    )


def _figure_f4(evidence: Mapping[str, Any]) -> bytes:
    faults = (
        (90, 280, "Condition mismatch", "Candidate moved to another SNR cell", TEAL),
        (1130, 280, "Seed cherry-pick", "Best candidate seed versus baseline mean", AMBER),
        (90, 610, "Metric mismatch", "Candidate BLER placed beside baseline BER", RED),
        (1130, 610, "Hidden information", "Diagnostic oracle presented as a peer", GREEN),
    )
    body = [
        _text(80, 82, "BREAK THE COMPARISON", size=19, weight=750, fill=GREEN, tracking=2.5),
        _text(80, 155, "The arithmetic can be exact while the comparison is wrong.", size=48, weight=740),
        _rect(585, 310, 430, 270, fill="#241417", stroke=RED, radius=32, stroke_width=3),
        _text(800, 372, "CLAIM", size=18, weight=750, fill=RED, anchor="middle", tracking=2),
        _text(800, 458, f'{evidence["headline"]["primary_learned_reduction_percent"]:.2f}%', size=78, weight=780, anchor="middle"),
        _text(800, 510, "precise-looking ≠ defensible", size=24, weight=650, fill=MUTED, anchor="middle"),
        _text(800, 548, "Restore every contract boundary before quoting.", size=18, fill=AMBER, anchor="middle"),
    ]
    for x, y, title, detail, color in faults:
        body.extend(
            (
                _rect(x, y, 380, 180, fill=PANEL, stroke=color, radius=24),
                _text(x + 28, y + 58, title, size=28, weight=730, fill=color),
                _multiline(x + 28, y + 104, _wrap(detail, 33), size=18, fill=MUTED, line_height=1.35),
            )
        )
        start_x = x + 380 if x < 500 else x
        end_x = 575 if x < 500 else 1025
        body.append(_arrow(start_x, y + 90, end_x, 445, color=RED))
    body.append(_text(800, 858, evidence["scientific_status"]["disclosure"], size=18, fill=AMBER, anchor="middle"))
    return _svg(
        "Four ways to break the comparison",
        "Condition mismatch, seed cherry-picking, metric mismatch, and hidden oracle information all invalidate an otherwise precise-looking claim.",
        "".join(body),
    )


def _figure_f5(evidence: Mapping[str, Any]) -> bytes:
    consumers = (
        (90, 230, "README", "Launch headline"),
        (90, 470, "Documentation", "Methods + disclosure"),
        (90, 710, "Web demo", "Interactive contract audit"),
        (1160, 230, "Figure F3", "Series + observed range"),
        (1160, 470, "Tables T0–T2", "Comparisons + guardrails"),
        (1160, 710, "Launch video", "Recorded external walkthrough"),
    )
    body = [
        _text(80, 82, "ONE SOURCE, MANY SURFACES", size=19, weight=750, fill=GREEN, tracking=2.5),
        _text(80, 155, "Every launch number has one place to change.", size=48, weight=740),
        _rect(565, 315, 470, 300, fill="#102a23", stroke=GREEN, radius=36, stroke_width=3),
        _text(800, 380, "CANONICAL GENERATED SOURCE", size=17, weight=750, fill=GREEN, anchor="middle", tracking=1.8),
        _text(800, 455, "launch_evidence.json", size=37, weight=750, anchor="middle", family="ui-monospace,SFMono-Regular,Consolas,monospace"),
        _text(800, 506, f'{evidence["design"]["run_count"]} runs · {len(evidence["design"]["snr_db"])} cells · {len(evidence["design"]["paired_seeds"])} seeds', size=21, fill=MUTED, anchor="middle"),
        _text(800, 552, "Evidence " + str(evidence["sha256"])[:12] + "…", size=18, fill=TEAL, anchor="middle", family="ui-monospace,SFMono-Regular,Consolas,monospace"),
    ]
    for x, y, title, detail in consumers:
        body.extend(
            (
                _rect(x, y, 350, 140, fill=PANEL, stroke=LINE, radius=22),
                _text(x + 26, y + 54, title, size=27, weight=730),
                _text(x + 26, y + 96, detail, size=18, fill=MUTED),
            )
        )
        if x < 500:
            body.append(_arrow(450, y + 70, 555, 465))
        else:
            body.append(_arrow(1045, 465, 1150, y + 70))
    body.extend(
        (
            _rect(475, 730, 650, 90, fill="#251f0d", stroke="#705c23", radius=18),
            _text(800, 768, "The warning travels with the number", size=17, weight=750, fill=AMBER, anchor="middle", tracking=1.2),
            _text(800, 800, evidence["scientific_status"]["disclosure"], size=15, fill=MUTED, anchor="middle"),
        )
    )
    return _svg(
        "One evidence source, many public outputs",
        "The canonical launch evidence supplies README, documentation, webpage, experiment figure, result tables, and the recorded launch video.",
        "".join(body),
    )


def _wrap(value: str, width: int) -> tuple[str, ...]:
    words = value.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and len(candidate) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return tuple(lines)


def _csv_bytes(rows: Iterable[Iterable[Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for row in rows:
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _markdown_bytes(title: str, headers: list[str], rows: list[list[str]], note: str) -> bytes:
    def cell(value: str) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        f"### {title}",
        "",
        "| " + " | ".join(cell(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(item) for item in row) + " |" for row in rows)
    lines.extend(("", note, ""))
    return "\n".join(lines).encode("utf-8")


def _rate(value: Any) -> str:
    return f"{float(value):.6e}"


def _percent(value: Any) -> str:
    return f"{float(value):.2f}%"


def _tables(evidence: Mapping[str, Any]) -> dict[str, bytes]:
    summary_headers = [
        "SNR (dB)",
        "Uncompensated mean BER",
        "Calibrated-reference mean BER",
        "Learned mean BER",
        "Learned observed min BER",
        "Learned observed max BER",
        "Reduction vs uncompensated",
        "Relative gap to reference",
    ]
    summary_rows = [
        [
            f'{float(row["snr_db"]):g}',
            _rate(row["uncompensated_mean_ber"]),
            _rate(row["calibrated_reference_mean_ber"]),
            _rate(row["learned_mean_ber"]),
            _rate(row["learned_minimum_ber"]),
            _rate(row["learned_maximum_ber"]),
            _percent(row["learned_reduction_percent"]),
            _percent(row["learned_relative_gap_to_reference_percent"]),
        ]
        for row in evidence["comparisons"]
    ]
    summary_note = (
        "Observed minima and maxima span three paired held-out seeds; they are not confidence "
        "intervals. The calibrated reference has diagnostic calibration knowledge. "
        + evidence["scientific_status"]["disclosure"]
    )

    design = evidence["design"]
    status = evidence["scientific_status"]
    clearance = evidence["distribution_clearance"]
    method_roles = "; ".join(
        f'{item["label"]} ({item["role"]})' for item in design["methods"]
    )
    contract_rows = [
        ["Experiment", evidence["identity"]["title"], "identity.title"],
        ["Result status", evidence["identity"]["result_status"], "identity.result_status"],
        ["Methods and roles", method_roles, "design.methods"],
        ["SNR cells (dB)", ", ".join(f"{float(value):g}" for value in design["snr_db"]), "design.snr_db"],
        ["Paired seeds", ", ".join(str(value) for value in design["paired_seeds"]), "design.paired_seeds"],
        ["Retained runs", str(design["run_count"]), "design.run_count"],
        ["Compared bits per run", f'{design["bits_per_run"]:,}', "design.bits_per_run"],
        ["Aggregation", status["aggregation"], "scientific_status.aggregation"],
        ["Statistical unit", design["statistical_unit"], "design.statistical_unit"],
        ["Uncertainty display", status["uncertainty_display"], "scientific_status.uncertainty_display"],
        ["Evidence tier", status["evidence_level"], "scientific_status.evidence_level"],
        ["Publication ready", str(status["publication_ready"]).lower(), "scientific_status.publication_ready"],
        ["Distribution clearance", clearance["status"], "distribution_clearance.status"],
        ["Reference role", status["reference_role"], "scientific_status.reference_role"],
    ]
    contract_headers = ["Contract field", "Declared value", "Evidence path"]
    contract_note = "Distribution clearance and scientific status are independent."

    guardrail_headers = ["Guardrail", "Required handling", "Evidence path"]
    guardrail_rows = [
        ["Primary quantitative wording", evidence["headline"]["text"], "headline.text"],
        ["Adjacent disclosure", status["disclosure"], "scientific_status.disclosure"],
        ["Uncertainty language", "Call bands observed minima/maxima; never confidence intervals.", "scientific_status.uncertainty_display"],
        ["Reference role", "Describe the calibrated oracle as a diagnostic reference with privileged calibration knowledge.", "scientific_status.reference_role"],
        ["Scientific tier", "Do not call the selected benchmark publication-ready.", "scientific_status.publication_ready"],
        ["Final release state", "The development repository is public; stable v0.2.0 and the paper bundle remain pending final review.", "distribution_clearance.status"],
        ["Numeric source", "Derive displayed values from series, comparisons, or headline.", "consumer_contract.rules"],
    ]
    guardrail_note = "These guardrails apply to README, documentation, webpage, figures, tables, and video overlays."

    return {
        "tables/t0-result-summary.csv": _csv_bytes([summary_headers, *summary_rows]),
        "tables/t0-result-summary.md": _markdown_bytes("T0 · Paired BER result summary", summary_headers, summary_rows, summary_note),
        "tables/t1-experiment-contract.csv": _csv_bytes([contract_headers, *contract_rows]),
        "tables/t1-experiment-contract.md": _markdown_bytes("T1 · Experiment contract summary", contract_headers, contract_rows, contract_note),
        "tables/t2-claim-guardrails.csv": _csv_bytes([guardrail_headers, *guardrail_rows]),
        "tables/t2-claim-guardrails.md": _markdown_bytes("T2 · Launch claim guardrails", guardrail_headers, guardrail_rows, guardrail_note),
    }


def _load_evidence(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LaunchAssetError(f"cannot load launch evidence: {exc}") from exc
    _require(isinstance(payload, dict), "launch evidence must be an object")
    spec = importlib.util.spec_from_file_location(
        "noema_launch_evidence_asset_verifier",
        ROOT / "tools" / "generate_launch_evidence.py",
    )
    _require(spec is not None and spec.loader is not None, "cannot load launch verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.verify_projection(payload, root=ROOT)


def build_assets(evidence: Mapping[str, Any]) -> dict[str, bytes]:
    assets = {
        "f0-launch-hero.svg": _figure_f0(evidence),
        "f1-comparison-anatomy.svg": _figure_f1(evidence),
        "f2-contract-to-evidence.svg": _figure_f2(evidence),
        "f3-receiver-ber-vs-snr.svg": _figure_f3(evidence),
        "f4-break-the-comparison.svg": _figure_f4(evidence),
        "f5-one-source-many-surfaces.svg": _figure_f5(evidence),
        **_tables(evidence),
    }
    figure_records = [
        {
            "id": identifier,
            "title": title,
            "path": (OUTPUT_RELATIVE / filename).as_posix(),
            "sha256": _digest(assets[filename]),
            "size_bytes": len(assets[filename]),
        }
        for identifier, filename, title in FIGURES
    ]
    table_records = [
        {
            "id": identifier,
            "title": title,
            "path": (OUTPUT_RELATIVE / filename).as_posix(),
            "sha256": _digest(assets[filename]),
            "size_bytes": len(assets[filename]),
        }
        for identifier, filename, title in TABLES
    ]
    evidence_bytes = (ROOT / EVIDENCE_RELATIVE).read_bytes()
    manifest = {
        "schema_version": 1,
        "kind": "noema.launch_asset_manifest",
        "evidence": {
            "path": EVIDENCE_RELATIVE.as_posix(),
            "file_sha256": _digest(evidence_bytes),
            "document_sha256": evidence["sha256"],
        },
        "generator": {
            "path": GENERATOR_RELATIVE.as_posix(),
            "sha256": _file_digest(ROOT / GENERATOR_RELATIVE),
        },
        "schema": {
            "path": SCHEMA_RELATIVE.as_posix(),
            "sha256": _file_digest(ROOT / SCHEMA_RELATIVE),
        },
        "figures": figure_records,
        "tables": table_records,
        "scientific_disclosure": evidence["scientific_status"]["disclosure"],
        "paper_binding": "not_selected_for_paper_during_review",
    }
    schema = json.loads((ROOT / SCHEMA_RELATIVE).read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(manifest),
        key=lambda item: list(item.absolute_path),
    )
    _require(
        not errors,
        "launch asset manifest schema errors: "
        + "; ".join(error.message for error in errors),
    )
    assets["manifest.json"] = (
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return assets


def _write_assets(output_dir: Path, assets: Mapping[str, bytes]) -> None:
    for relative, payload in assets.items():
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def _check_assets(output_dir: Path, assets: Mapping[str, bytes]) -> None:
    stale = []
    expected = set(assets)
    for relative, payload in assets.items():
        target = output_dir / relative
        if not target.is_file() or target.read_bytes() != payload:
            stale.append(relative)
    if output_dir.is_dir():
        actual = {
            path.relative_to(output_dir).as_posix()
            for path in output_dir.rglob("*")
            if path.is_file()
        }
        stale.extend("unexpected:" + item for item in sorted(actual - expected))
    _require(not stale, "launch assets are stale: " + ", ".join(sorted(stale)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / OUTPUT_RELATIVE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        evidence = _load_evidence(ROOT / EVIDENCE_RELATIVE)
        assets = build_assets(evidence)
        if args.check:
            _check_assets(args.output_dir, assets)
        else:
            _write_assets(args.output_dir, assets)
    except (LaunchAssetError, OSError, ValueError) as exc:
        parser.error(str(exc))
    if not args.check:
        print(
            "generated %d launch assets in %s"
            % (len(assets), args.output_dir)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
