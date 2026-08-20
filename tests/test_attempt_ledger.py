from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import noema_lab.core.attempt_ledger as attempt_ledger_module
from noema_lab.core.attempt_ledger import (
    AttemptLedgerError,
    BenchmarkAttemptLedger,
)


class BenchmarkAttemptLedgerTests(unittest.TestCase):
    def test_windows_lock_backend_locks_and_unlocks_one_byte(self):
        class FakeMsvcrt:
            LK_LOCK = 1
            LK_UNLCK = 2

            def __init__(self) -> None:
                self.calls = []

            def locking(self, fd: int, mode: int, size: int) -> None:
                self.calls.append((fd, mode, size))

        backend = FakeMsvcrt()
        with tempfile.TemporaryFile(mode="w+b") as handle:
            with (
                mock.patch.object(attempt_ledger_module, "_fcntl", None),
                mock.patch.object(attempt_ledger_module, "_msvcrt", backend),
                attempt_ledger_module._exclusive_file_lock(handle),
            ):
                self.assertEqual(handle.read(1), b"\0")

        self.assertEqual(
            [(mode, size) for _, mode, size in backend.calls],
            [(backend.LK_LOCK, 1), (backend.LK_UNLCK, 1)],
        )

    def test_windows_directory_fsync_is_a_safe_noop(self):
        unused = Path("unused")
        with (
            mock.patch.object(attempt_ledger_module.os, "name", "nt"),
            mock.patch.object(attempt_ledger_module.os, "open") as open_mock,
        ):
            attempt_ledger_module._fsync_directory(unused)

        open_mock.assert_not_called()

    def test_all_attempt_outcomes_remain_in_registration_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            ledger = BenchmarkAttemptLedger(root)
            ids = {}
            for status in ("completed", "failed", "cancelled", "resource_rejected"):
                started = ledger.begin(
                    benchmark_id="benchmark.test",
                    benchmark_version="1",
                    protocol_sha256="a" * 64,
                    attempt_id="attempt-%s" % status,
                )
                ids[status] = started["attempt_id"]
                ledger.finalize(
                    started["attempt_id"],
                    status=status,
                    error="fixture" if status in {"failed", "cancelled"} else None,
                    recipe_outcomes=[
                        {
                            "recipe_id": "method",
                            "status": "rejected_resource_budget"
                            if status == "resource_rejected"
                            else status,
                        }
                    ],
                )
            interrupted = ledger.begin(
                benchmark_id="benchmark.test",
                benchmark_version="1",
                protocol_sha256="a" * 64,
                attempt_id="attempt-interrupted",
            )
            replacement = ledger.begin(
                benchmark_id="benchmark.test",
                benchmark_version="1",
                protocol_sha256="a" * 64,
                attempt_id="attempt-replacement",
            )
            ledger.finalize(replacement["attempt_id"], status="completed")
            ledger.supersede(
                ids["completed"],
                superseded_by_attempt_id=replacement["attempt_id"],
                reason="preregistered corrected rerun",
            )

            snapshot_path = Path(tmp) / "paper-attempt-ledger.json"
            snapshot = ledger.registration_snapshot(snapshot_path)
            snapshot_written = snapshot_path.is_file()

        self.assertEqual(snapshot["attempt_count"], 6)
        self.assertEqual(snapshot["status_counts"]["superseded"], 1)
        self.assertEqual(snapshot["status_counts"]["failed"], 1)
        self.assertEqual(snapshot["status_counts"]["cancelled"], 1)
        self.assertEqual(snapshot["status_counts"]["resource_rejected"], 1)
        self.assertEqual(snapshot["status_counts"]["started_without_terminal_event"], 1)
        self.assertEqual(interrupted["attempt_id"], "attempt-interrupted")
        self.assertTrue(snapshot_written)
        self.assertEqual(len(snapshot["sha256"]), 64)

    def test_event_tampering_breaks_chain_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BenchmarkAttemptLedger(Path(tmp) / "benchmarks")
            event = ledger.begin(
                benchmark_id="benchmark.test",
                benchmark_version="1",
                protocol_sha256="b" * 64,
                attempt_id="tamper-me",
            )
            event_path = next(ledger.events_dir.glob("*.json"))
            payload = json.loads(event_path.read_text(encoding="utf-8"))
            payload["payload"]["benchmark_id"] = "forged"
            event_path.write_text(json.dumps(payload), encoding="utf-8")
            report = ledger.verify()

        self.assertEqual(event["event_type"], "started")
        self.assertEqual(report["status"], "invalid")
        self.assertTrue(any("digest mismatch" in error for error in report["errors"]))

    def test_duplicate_event_keys_are_rejected_before_hash_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BenchmarkAttemptLedger(Path(tmp) / "benchmarks")
            ledger.begin(
                benchmark_id="benchmark.test",
                benchmark_version="1",
                protocol_sha256="b" * 64,
                attempt_id="duplicate-key",
            )
            event_path = next(ledger.events_dir.glob("*.json"))
            content = event_path.read_text(encoding="utf-8")
            content = content.replace(
                '"kind": "noema.benchmark_attempt_event",',
                '"kind": "forged", "kind": "noema.benchmark_attempt_event",',
                1,
            )
            event_path.write_text(content, encoding="utf-8")
            report = ledger.verify()

        self.assertEqual(report["status"], "invalid")
        self.assertTrue(
            any("Duplicate JSON object key" in error for error in report["errors"])
        )

    def test_tail_truncation_below_recorded_head_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BenchmarkAttemptLedger(Path(tmp) / "benchmarks")
            for attempt_id in ("first", "second"):
                ledger.begin(
                    benchmark_id="benchmark.test",
                    benchmark_version="1",
                    protocol_sha256="e" * 64,
                    attempt_id=attempt_id,
                )
            sorted(ledger.events_dir.glob("*.json"))[-1].unlink()
            report = ledger.verify()

        self.assertEqual(report["status"], "invalid")
        self.assertTrue(any("truncated" in error for error in report["errors"]))

    def test_concurrent_appends_form_one_contiguous_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BenchmarkAttemptLedger(Path(tmp) / "benchmarks")
            errors = []

            def record(index: int) -> None:
                try:
                    ledger.begin(
                        benchmark_id="benchmark.concurrent",
                        benchmark_version="1",
                        protocol_sha256="c" * 64,
                        attempt_id="attempt-%02d" % index,
                    )
                except Exception as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            threads = [threading.Thread(target=record, args=(index,)) for index in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
            report = ledger.verify()

        self.assertEqual(errors, [])
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["event_count"], 12)
        self.assertEqual(
            [event["sequence"] for event in report["events"]],
            list(range(1, 13)),
        )

    def test_concurrent_sealed_test_reservations_are_unique_and_denials_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BenchmarkAttemptLedger(Path(tmp) / "benchmarks")
            events = []
            errors = []
            result_lock = threading.Lock()

            def reserve(index: int) -> None:
                try:
                    event = ledger.begin(
                        benchmark_id="benchmark.sealed",
                        benchmark_version="1",
                        protocol_sha256="f" * 64,
                        attempt_id="sealed-%02d" % index,
                        invocation={
                            "test_access": True,
                            "access_token_id": "1" * 64,
                            "access_budget": 2,
                            # This untrusted pre-count is deliberately wrong;
                            # the locked append must overwrite it.
                            "access_sequence": 999,
                        },
                    )
                    with result_lock:
                        events.append(event)
                except Exception as exc:  # pragma: no cover - asserted below
                    with result_lock:
                        errors.append(exc)

            threads = [threading.Thread(target=reserve, args=(index,)) for index in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
            report = ledger.verify()

        self.assertEqual(errors, [])
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["attempt_count"], 8)
        invocations = [event["payload"]["invocation"] for event in events]
        self.assertEqual(
            sorted(item["access_sequence"] for item in invocations),
            list(range(1, 9)),
        )
        self.assertEqual(sum(bool(item["access_granted"]) for item in invocations), 2)
        self.assertEqual(
            report["status_counts"],
            {"started_without_terminal_event": 8},
        )

    def test_sealed_test_resume_continues_failed_granted_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            ledger = BenchmarkAttemptLedger(root)
            token = "2" * 64
            protocol = "f" * 64
            started = ledger.begin(
                benchmark_id="benchmark.sealed",
                benchmark_version="1",
                protocol_sha256=protocol,
                attempt_id="sealed-original",
                invocation={
                    "test_access": True,
                    "access_token_id": token,
                    "access_budget": 1,
                },
            )
            failed_result = root / "failed-result"
            failed_result.mkdir(parents=True)
            (failed_result / "result.json").write_text("{}\n", encoding="utf-8")
            ledger.finalize(
                started["attempt_id"],
                status="failed",
                result_dir=failed_result,
                error="synthetic interruption",
            )

            resumed = ledger.begin(
                benchmark_id="benchmark.sealed",
                benchmark_version="1",
                protocol_sha256=protocol,
                attempt_id="sealed-resume",
                invocation={
                    "test_access": True,
                    "access_token_id": token,
                    "access_budget": 1,
                    "resume": {
                        "explicit": True,
                        "source_result_id": failed_result.name,
                    },
                },
            )
            fresh = ledger.begin(
                benchmark_id="benchmark.sealed",
                benchmark_version="1",
                protocol_sha256=protocol,
                attempt_id="sealed-fresh",
                invocation={
                    "test_access": True,
                    "access_token_id": token,
                    "access_budget": 1,
                },
            )

        resumed_invocation = resumed["payload"]["invocation"]
        self.assertTrue(resumed_invocation["access_granted"])
        self.assertTrue(resumed_invocation["access_continuation"])
        self.assertEqual(resumed_invocation["access_sequence"], 1)
        self.assertEqual(
            resumed_invocation["continued_attempt_id"],
            "sealed-original",
        )
        self.assertFalse(fresh["payload"]["invocation"]["access_granted"])

    def test_invalid_transitions_and_out_of_store_result_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BenchmarkAttemptLedger(Path(tmp) / "benchmarks")
            started = ledger.begin(
                benchmark_id="benchmark.test",
                benchmark_version="1",
                protocol_sha256="d" * 64,
                attempt_id="one",
            )
            outside = Path(tmp) / "outside"
            outside.mkdir()
            (outside / "result.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(AttemptLedgerError, "outside"):
                ledger.finalize(
                    started["attempt_id"],
                    status="completed",
                    result_dir=outside,
                )
            ledger.finalize(started["attempt_id"], status="failed")
            with self.assertRaisesRegex(AttemptLedgerError, "already finalized"):
                ledger.finalize(started["attempt_id"], status="completed")


if __name__ == "__main__":
    unittest.main()
