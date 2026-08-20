from __future__ import annotations

"""Validation helpers for flat arrays that carry explicit capture records.

Some channel operations store a logically batched tensor as one flat NPZ
array.  ``capture_record_count`` and ``capture_record_shape`` are therefore a
contract about the stored array, not descriptive hints.  Representation
transforms must validate the input contract and either rewrite it for their
output representation or remove it deliberately.
"""

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping, MutableMapping, Sequence


class CaptureRecordLayoutError(ValueError):
    """An explicit capture-record declaration is incomplete or inconsistent."""


@dataclass(frozen=True)
class CaptureRecordLayout:
    count: int
    shape: tuple[int, ...]

    @property
    def elements_per_record(self) -> int:
        return int(prod(self.shape))

    @property
    def element_count(self) -> int:
        return int(self.count * self.elements_per_record)


def explicit_capture_record_layout(
    metadata: Mapping[str, Any],
    element_count: int,
    *,
    label: str,
) -> CaptureRecordLayout | None:
    """Return and validate the paired explicit record declaration, if present."""

    declared_count = metadata.get("capture_record_count")
    declared_shape = metadata.get("capture_record_shape")
    if declared_count is None and declared_shape is None:
        return None
    if (
        not isinstance(declared_count, int)
        or isinstance(declared_count, bool)
        or declared_count <= 0
    ):
        raise CaptureRecordLayoutError(
            "%s has invalid capture_record_count" % label
        )
    if (
        not isinstance(declared_shape, (list, tuple))
        or not declared_shape
        or any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or item <= 0
            for item in declared_shape
        )
    ):
        raise CaptureRecordLayoutError(
            "%s has invalid capture_record_shape" % label
        )
    layout = CaptureRecordLayout(
        count=int(declared_count),
        shape=tuple(int(item) for item in declared_shape),
    )
    if layout.element_count != int(element_count):
        raise CaptureRecordLayoutError(
            "%s capture-record layout declares %d elements but stores %d"
            % (label, layout.element_count, int(element_count))
        )
    return layout


def set_uniform_capture_record_layout(
    metadata: MutableMapping[str, Any],
    layout: CaptureRecordLayout | None,
    output_element_count: int,
    *,
    label: str,
) -> None:
    """Rewrite a validated record layout for an independent flat transform."""

    metadata.pop("capture_record_count", None)
    metadata.pop("capture_record_shape", None)
    if layout is None:
        return
    if int(output_element_count) <= 0 or int(output_element_count) % layout.count:
        raise CaptureRecordLayoutError(
            "%s cannot represent %d output elements as %d equal capture records"
            % (label, int(output_element_count), layout.count)
        )
    metadata["capture_record_count"] = int(layout.count)
    metadata["capture_record_shape"] = [
        int(output_element_count) // int(layout.count)
    ]


def set_capture_record_shape(
    metadata: MutableMapping[str, Any],
    layout: CaptureRecordLayout | None,
    output_shape: Sequence[int],
    output_element_count: int,
    *,
    label: str,
) -> None:
    """Rewrite a layout to an exact caller-provided per-record shape."""

    metadata.pop("capture_record_count", None)
    metadata.pop("capture_record_shape", None)
    if layout is None:
        return
    shape = tuple(int(item) for item in output_shape)
    if not shape or any(item <= 0 for item in shape):
        raise CaptureRecordLayoutError(
            "%s produced an invalid capture-record shape" % label
        )
    expected = int(layout.count * prod(shape))
    if expected != int(output_element_count):
        raise CaptureRecordLayoutError(
            "%s output capture-record layout declares %d elements but stores %d"
            % (label, expected, int(output_element_count))
        )
    metadata["capture_record_count"] = int(layout.count)
    metadata["capture_record_shape"] = [int(item) for item in shape]


def remove_explicit_capture_record_layout(
    metadata: MutableMapping[str, Any],
) -> None:
    metadata.pop("capture_record_count", None)
    metadata.pop("capture_record_shape", None)


def remove_capture_record_metadata(
    metadata: MutableMapping[str, Any],
) -> None:
    """Remove every array-layout key before assigning new record semantics."""

    remove_explicit_capture_record_layout(metadata)
    for key in (
        "capture_record_axis",
        "record_axis",
        "sample_axis",
        "capture_record_unit",
    ):
        metadata.pop(key, None)
