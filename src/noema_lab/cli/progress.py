"""Small, dependency-free progress rendering for command-line workflows."""

from __future__ import annotations

import math
import sys
import time
from typing import Any, Callable, Mapping, Optional, TextIO, Tuple


class CliProgressRenderer:
    """Render structured progress payloads to a text stream.

    Progress is enabled automatically for an interactive stream.  Callers may
    explicitly enable it for redirected streams, where updates are emitted as
    newline-delimited records instead of using carriage-return redraws.
    """

    def __init__(
        self,
        *,
        stream: Optional[TextIO] = None,
        enabled: Optional[bool] = None,
        label: str = "Progress",
        bucket_size: int = 5,
        bar_width: int = 24,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.interactive = _is_interactive(self.stream)
        self.enabled = self.interactive if enabled is None else bool(enabled)
        self.label = str(label).strip() or "Progress"
        self.bucket_size = max(1, min(100, int(bucket_size)))
        self.bar_width = max(8, min(60, int(bar_width)))
        self._clock = clock
        self._started_at: Optional[float] = None
        self._last_width = 0
        self._line_active = False
        self._closed = False
        self._last_record_signature: Optional[Tuple[str, int, bool]] = None

    def __enter__(self) -> "CliProgressRenderer":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> bool:
        self.close()
        return False

    def update(self, payload: Mapping[str, Any]) -> None:
        """Render one structured progress update without propagating I/O errors."""

        if not self.enabled or self._closed:
            return
        progress = dict(payload or {})
        now = self._clock()
        if self._started_at is None:
            self._started_at = now
        elapsed = max(0.0, now - self._started_at)
        percent = _percent(progress.get("percent"))
        phase = _clean(progress.get("phase"))
        terminal = phase.lower() in {"completed", "complete", "failed", "cancelled", "canceled"}
        if percent is not None and percent >= 100.0:
            terminal = True
        line = self._format(
            progress,
            percent=percent,
            phase=phase,
            elapsed=elapsed,
        )

        if self.interactive:
            padding = " " * max(0, self._last_width - len(line))
            if not self._write("\r" + line + padding):
                return
            self._last_width = len(line)
            self._line_active = True
            if terminal:
                self._write("\n")
                self._line_active = False
                self._last_width = 0
            return

        # Redirected progress is intentionally sparse so --progress remains
        # useful in logs without producing one record per captured sample.
        bucket = int((percent or 0.0) // self.bucket_size)
        signature = (phase, bucket, terminal)
        if signature == self._last_record_signature:
            return
        self._last_record_signature = signature
        self._write(line + "\n")

    def close(self) -> None:
        """Finish an active terminal line; safe to call more than once."""

        if self._closed:
            return
        self._closed = True
        if self.enabled and self.interactive and self._line_active:
            self._write("\n")
        self._line_active = False
        self._last_width = 0

    def _format(
        self,
        progress: Mapping[str, Any],
        *,
        percent: Optional[float],
        phase: str,
        elapsed: float,
    ) -> str:
        parts = [self.label, _bar(percent, self.bar_width)]
        if percent is not None:
            parts.append("%5.1f%%" % percent)

        completed = _integer(progress.get("completed_samples"))
        total = _integer(progress.get("total_samples"))
        if completed is not None and total is not None:
            parts.append("%d/%d samples" % (completed, total))
        elif completed is not None:
            parts.append("%d samples" % completed)

        parts.append("elapsed %s" % _duration(elapsed))
        rate: Optional[float] = None
        if elapsed >= 0.1 and completed is not None and completed > 0:
            rate = completed / elapsed
            parts.append("%.1f samples/s" % rate)
        if (
            rate is not None
            and total is not None
            and completed is not None
            and total > completed
        ):
            eta = (total - completed) / rate
            parts.append("ETA %s" % _duration(eta))

        if phase:
            parts.append(phase)
        message = _clean(progress.get("message"))
        if message and message.casefold() != phase.casefold():
            parts.append(message)
        return " | ".join(parts)

    def _write(self, value: str) -> bool:
        try:
            self.stream.write(value)
            self.stream.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            self.enabled = False
            self._line_active = False
            return False


def _is_interactive(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError, ValueError):
        return False


def _percent(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return max(0.0, min(100.0, parsed))


def _integer(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _bar(percent: Optional[float], width: int) -> str:
    if percent is None:
        return "[" + ("?" * width) + "]"
    completed = int(round(width * percent / 100.0))
    completed = max(0, min(width, completed))
    return "[" + ("#" * completed) + ("-" * (width - completed)) + "]"


def _duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 10.0:
        return "%.1fs" % seconds
    rounded = int(round(seconds))
    if rounded < 60:
        return "%ds" % rounded
    minutes, remaining = divmod(rounded, 60)
    if minutes < 60:
        return "%dm%02ds" % (minutes, remaining)
    hours, minutes = divmod(minutes, 60)
    return "%dh%02dm" % (hours, minutes)
