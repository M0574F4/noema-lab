from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional


def demo_training_candidates(relative: Path) -> Iterable[Path]:
    """Yield checkout, target-install, and normal-prefix demo locations."""

    relative = Path(relative)
    module_path = Path(__file__).resolve()
    yield Path.cwd() / relative
    yield module_path.parents[3] / relative
    yield module_path.parents[2] / "share" / "noema-lab" / relative
    yield Path(sys.prefix) / "share" / "noema-lab" / relative


def find_demo_training_dir(
    relative: Path,
    *,
    marker: str = "template.yaml",
) -> Optional[Path]:
    for candidate in demo_training_candidates(relative):
        if (candidate / marker).is_file():
            return candidate
    return None
