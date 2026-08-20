from __future__ import annotations

from pathlib import Path

from noema_lab.core import structured_input


STANDALONE_STRUCTURED_INPUT_FILENAME = "structured_input.py"


def standalone_structured_input_source() -> str:
    """Return the dependency-free decoder source shipped with Noema."""

    source = Path(structured_input.__file__).resolve()
    if source.suffix == ".pyc":
        source = source.with_suffix(".py")
    if not source.is_file():
        raise RuntimeError(
            "Noema's standalone structured-input decoder source is unavailable: %s"
            % source
        )
    return source.read_text(encoding="utf-8")


def write_standalone_structured_input(directory: Path) -> Path:
    """Copy the dependency-free strict decoder into an exported project."""

    destination = Path(directory) / STANDALONE_STRUCTURED_INPUT_FILENAME
    destination.write_text(standalone_structured_input_source(), encoding="utf-8")
    return destination
