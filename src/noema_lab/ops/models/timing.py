from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, TypeVar

JsonDict = Dict[str, Any]
T = TypeVar("T")


def timed_call(
    measurements: List[JsonDict],
    example_index: Optional[int],
    stage: str,
    fn: Callable[[], T],
    **fields: Any,
) -> T:
    start = time.perf_counter()
    result = fn()
    append_measurement(measurements, example_index, stage, time.perf_counter() - start, **fields)
    return result


def append_measurement(
    measurements: List[JsonDict],
    example_index: Optional[int],
    stage: str,
    duration_s: float,
    **fields: Any,
) -> None:
    record: JsonDict = {
        "stage": str(stage),
        "duration_s": float(duration_s),
        "unit": "seconds",
    }
    if example_index is not None:
        record["example_index"] = int(example_index)
    record.update({key: value for key, value in fields.items() if value is not None})
    measurements.append(record)


def codec_timing_metadata(
    role: str,
    measurements: List[JsonDict],
    runner: str = "operation",
    notes: Optional[JsonDict] = None,
) -> JsonDict:
    metadata: JsonDict = {
        "schema_version": 2,
        "enabled": True,
        "scope": "per_example_stage",
        "measurement_protocol": "single_execution_per_example",
        "warmup_runs": 0,
        "timed_runs": 1,
        "clock": "perf_counter",
        "runner": runner,
        "role": role,
        "measurements": list(measurements),
    }
    if notes:
        metadata["notes"] = dict(notes)
    return metadata
