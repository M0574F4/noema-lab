#!/usr/bin/env python3
"""Finalize an orphaned local benchmark result so it can be explicitly resumed.

This recovery command is intentionally narrow. It accepts only a result whose
result.json is still ``running``, whose ledger attempt is nonterminal, whose
completed recipe evidence validates, and whose job-control files have been
inactive for a caller-specified grace period. It then records a failed terminal
result and appends the corresponding immutable ledger event.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Mapping

from noema_lab.core.attempt_ledger import BenchmarkAttemptLedger
from noema_lab.core.benchmark_run_evidence import (
    validate_benchmark_run_evidence_snapshots,
)
from noema_lab.core.benchmarks import (
    _benchmark_attempt_details,
    write_benchmark_reports,
)
from noema_lab.core.reproducibility import utc_now_iso
from noema_lab.core.storage import LocalStore


def _attempt(snapshot: Mapping[str, Any], attempt_id: str) -> Dict[str, Any]:
    rows = [
        dict(row)
        for row in snapshot.get("attempts") or []
        if isinstance(row, Mapping) and row.get("attempt_id") == attempt_id
    ]
    if len(rows) != 1:
        raise ValueError("Result attempt is absent or duplicated in the ledger")
    return rows[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(".noema"))
    parser.add_argument("--result-id", required=True)
    parser.add_argument("--job-control", required=True, type=Path)
    parser.add_argument("--minimum-inactive-seconds", type=float, default=120.0)
    parser.add_argument(
        "--reason",
        default="Worker process was interrupted before terminal finalization.",
    )
    args = parser.parse_args()

    store = LocalStore(args.workspace.resolve())
    result_dir = store.get_benchmark_result_dir(args.result_id)
    if (
        result_dir.is_symlink()
        or not result_dir.is_dir()
        or result_dir.resolve().parent != store.benchmarks_dir.resolve()
    ):
        raise ValueError("Result ID does not resolve to a safe benchmark directory")
    result_path = result_dir / "result.json"
    result = store.read_json(result_path)
    if result.get("status") != "running":
        raise ValueError("Recovery accepts only a result with status=running")

    job_dir = args.job_control.resolve()
    if not job_dir.is_dir():
        raise ValueError("Job-control directory is absent")
    activity_files = [
        path
        for path in (
            job_dir / "status.json",
            job_dir / "events.jsonl",
            job_dir / "worker.log",
        )
        if path.is_file()
    ]
    if not activity_files:
        raise ValueError("Job-control directory contains no activity files")
    inactive_seconds = time.time() - max(path.stat().st_mtime for path in activity_files)
    if inactive_seconds < args.minimum_inactive_seconds:
        raise ValueError(
            "Job-control files are too recent for orphan recovery (%.1f seconds)"
            % inactive_seconds
        )

    ledger = BenchmarkAttemptLedger(store.benchmarks_dir)
    snapshot = ledger.verify()
    if snapshot.get("status") != "valid":
        raise ValueError(
            "Attempt ledger is invalid: %s"
            % "; ".join(str(item) for item in snapshot.get("errors") or [])
        )
    attempt_id = str(result.get("attempt_id") or "")
    attempt = _attempt(snapshot, attempt_id)
    if attempt.get("terminal_status") is not None:
        raise ValueError("Attempt is already terminal")

    validated = validate_benchmark_run_evidence_snapshots(result_dir, result)
    completed_indices = {
        index
        for index, row in enumerate(result.get("recipes") or [])
        if isinstance(row, Mapping)
        and row.get("status") in {"completed", "rejected_resource_budget"}
    }
    validated_indices = {
        int(row["entry_index"])
        for row in validated.get("entries") or []
        if isinstance(row, Mapping)
    }
    if completed_indices - validated_indices:
        raise ValueError("Completed recipe rows lack validated result-local evidence")

    result["status"] = "failed"
    result["error"] = str(args.reason)
    result["completed_at_utc"] = utc_now_iso()
    descriptor = result.get("attempt_ledger")
    if not isinstance(descriptor, dict):
        raise ValueError("Result has no attempt-ledger descriptor")
    descriptor["expected_terminal_status"] = "failed"
    write_benchmark_reports(result_dir, result)
    store.write_json(result_path, result)
    ledger.finalize(
        attempt_id,
        status="failed",
        result_dir=result_dir,
        recipe_outcomes=[
            {
                "recipe_id": str(row.get("id") or ""),
                "status": str(row.get("status") or ""),
            }
            for row in result.get("recipes") or []
            if isinstance(row, Mapping)
        ],
        error=str(args.reason),
        details={
            **_benchmark_attempt_details(result),
            "orphan_recovery": {
                "job_control": str(job_dir),
                "inactive_seconds": inactive_seconds,
            },
        },
    )
    print("finalized interrupted benchmark: %s" % args.result_id)
    print("validated completed rows: %d" % len(completed_indices))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
