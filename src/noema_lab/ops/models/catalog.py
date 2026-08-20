from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from noema_lab.core.structured_input import load_strict_yaml_or_json

JsonDict = Dict[str, Any]


def load_model_catalog() -> JsonDict:
    catalog: JsonDict = {}
    for path in _catalog_paths():
        if path.is_file():
            _merge_catalog(catalog, _read_catalog(path))
    return catalog


def compressai_model_ids(catalog: JsonDict) -> List[str]:
    models = (((catalog.get("compressai") or {}).get("models")) or [])
    return [str(model["id"]) for model in models if isinstance(model, dict) and model.get("id")]


def compressai_model_config(catalog: JsonDict, model_id: str) -> JsonDict:
    models = (((catalog.get("compressai") or {}).get("models")) or [])
    for model in models:
        if isinstance(model, dict) and model.get("id") == model_id:
            return dict(model)
    raise KeyError(model_id)


def default_compressai_model(catalog: JsonDict) -> str:
    section = catalog.get("compressai") or {}
    return str(section.get("default_model") or (compressai_model_ids(catalog) or ["bmshj2018_hyperprior"])[0])


def default_compressai_quality(catalog: JsonDict) -> int:
    return int((catalog.get("compressai") or {}).get("default_quality") or 3)


def default_compressai_metric(catalog: JsonDict) -> str:
    return str((catalog.get("compressai") or {}).get("default_metric") or "mse")


def default_diffusers_model(catalog: JsonDict, family: str, fallback: str) -> str:
    return str((((catalog.get("diffusers") or {}).get(family) or {}).get("default_model_id")) or fallback)


def diffusers_model_config(catalog: JsonDict, family: str, model_id: str) -> JsonDict:
    section = ((catalog.get("diffusers") or {}).get(family) or {})
    for model in section.get("models") or []:
        if isinstance(model, dict) and model.get("id") == model_id:
            return dict(model)
    return {}


def diffusers_load_kwargs(catalog: JsonDict, family: str, model_id: str, params: JsonDict) -> JsonDict:
    config = diffusers_model_config(catalog, family, model_id)
    kwargs = dict(config.get("load_kwargs") or {})
    for key in ("subfolder", "revision", "variant"):
        if params.get(key):
            kwargs[key] = params[key]
    local_path = Path(str(model_id)).expanduser()
    if local_path.exists():
        # Hugging Face revisions do not identify local directories. Preserve
        # that distinction explicitly in result evidence at the call site.
        kwargs.pop("revision", None)
        return kwargs
    revision = str(kwargs.get("revision") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "Remote Diffusers model `%s` requires revision as a full immutable "
            "40-character commit" % model_id
        )
    kwargs["revision"] = revision
    return kwargs


def _catalog_paths() -> Iterable[Path]:
    yield Path(__file__).with_name("model_catalog.yaml")
    env_value = os.environ.get("NOEMA_MODEL_CATALOG") or ""
    for item in env_value.split(os.pathsep):
        if item.strip():
            yield Path(item.strip()).expanduser()
    cwd = Path.cwd()
    yield cwd / "noema_models.yaml"
    yield cwd / "models" / "noema_models.yaml"


def _read_catalog(path: Path) -> JsonDict:
    data = load_strict_yaml_or_json(path) or {}
    if not isinstance(data, dict):
        raise ValueError("Model catalog must be a mapping: %s" % path)
    _validate_unique_model_ids(data, path=path, location="$")
    return dict(data)


def _validate_unique_model_ids(value: Any, *, path: Path, location: str) -> None:
    """Reject ambiguous repeated model identities within one catalog source."""

    if isinstance(value, dict):
        for key, child in value.items():
            _validate_unique_model_ids(
                child,
                path=path,
                location="%s.%s" % (location, key),
            )
        return
    if not isinstance(value, list):
        return
    seen_ids: set[str] = set()
    for index, item in enumerate(value):
        item_location = "%s[%d]" % (location, index)
        if isinstance(item, dict) and "id" in item:
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                raise ValueError(
                    "Model catalog %s has an invalid model id at %s"
                    % (path, item_location)
                )
            if model_id in seen_ids:
                raise ValueError(
                    "Model catalog %s repeats model id %r at %s"
                    % (path, model_id, item_location)
                )
            seen_ids.add(model_id)
        _validate_unique_model_ids(item, path=path, location=item_location)


def _merge_catalog(target: JsonDict, incoming: JsonDict) -> None:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_catalog(target[key], value)
        elif isinstance(value, list) and isinstance(target.get(key), list):
            existing_ids = {
                item.get("id")
                for item in target[key]
                if isinstance(item, dict) and item.get("id") is not None
            }
            incoming_ids: set[Any] = set()
            for item in value:
                item_id = item.get("id") if isinstance(item, dict) else None
                if item_id is not None and item_id in incoming_ids:
                    raise ValueError(
                        "Incoming model catalog repeats model id %r in %s"
                        % (item_id, key)
                    )
                if item_id is not None:
                    incoming_ids.add(item_id)
                if item_id is not None and item_id in existing_ids:
                    target[key] = [
                        dict(item) if isinstance(old, dict) and old.get("id") == item_id else old
                        for old in target[key]
                    ]
                else:
                    target[key].append(item)
                    if item_id is not None:
                        existing_ids.add(item_id)
        else:
            target[key] = value
