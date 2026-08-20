from __future__ import annotations

"""Safe directory preparation for optional checked-in training examples.

Demo starters are copied into a researcher's training bundle.  Once attached,
that directory is also where the external trainer writes histories, evaluation
reports, benchmark packs, and sometimes additional researcher-owned files.
Refreshing the checked-in example must therefore update its known scaffold
files in place instead of replacing the directory.
"""

from pathlib import Path
from typing import Type

from noema_lab.core.structured_input import load_strict_yaml_or_json


_GENERATED_PROJECT_KIND = "noema.standalone_training_project"
_LEGACY_GENERATED_FILES = (
    "training_template.yaml",
    "train_config.yaml",
    "noema_recipe.yaml",
)


def prepare_demo_starter_directory(
    destination: Path,
    *,
    force: bool,
    error_type: Type[Exception] = ValueError,
) -> Path:
    """Validate a starter destination and prepare it for an in-place refresh.

    With ``force=True``, only a recognized generated starter may be refreshed.
    The directory itself is deliberately retained: each exporter overwrites
    the scaffold files it owns, while researcher-created evidence and artifacts
    that are not part of the scaffold survive the refresh.
    """

    path = Path(destination)
    if path.exists() and not path.is_dir():
        raise error_type("Export path exists and is not a directory: %s" % path)

    if path.is_dir() and any(path.iterdir()):
        if not force:
            raise error_type(
                "Export directory already exists and is not empty: %s. "
                "Use --force to refresh the generated project." % path
            )
        if not _is_generated_demo_starter(path, error_type=error_type):
            raise error_type(
                "Refusing to overwrite a nonempty directory that is not a "
                "recognized Noema demonstration starter: %s" % path
            )

    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_generated_demo_starter(
    path: Path,
    *,
    error_type: Type[Exception] = ValueError,
) -> bool:
    manifest_path = path / "project_manifest.yaml"
    if manifest_path.is_file():
        try:
            manifest = load_strict_yaml_or_json(manifest_path)
        except (OSError, UnicodeError, ValueError) as exc:
            raise error_type(
                "Cannot verify ownership of existing demonstration starter %s: %s"
                % (path, exc)
            ) from exc
        if (
            isinstance(manifest, dict)
            and str(manifest.get("kind") or "") == _GENERATED_PROJECT_KIND
        ):
            return True

    # Older released starters did not rely on the manifest kind for ownership.
    # Require the complete legacy marker set before allowing an in-place update.
    return all((path / filename).is_file() for filename in _LEGACY_GENERATED_FILES)
