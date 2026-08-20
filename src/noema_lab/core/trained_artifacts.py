from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from noema_lab.core.operations import OperationRegistry
from noema_lab.core.structured_input import (
    StructuredInputError,
    load_strict_yaml_or_json,
)


JsonDict = Dict[str, Any]
TRAINED_ARTIFACT_KIND = "noema.trained_block_artifact"
TRAINED_ARTIFACT_FILENAMES = (
    "trained_artifact.yaml",
    "trained_artifact.yml",
    "trained_artifact.json",
)
MAX_IMPORTED_CHECKPOINT_BYTES = 64 * 1024 * 1024
MAX_IMPORTED_CHECKPOINT_MEMBERS = 32
MAX_IMPORTED_CHECKPOINT_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_IMPORTED_ARTIFACT_PACKAGE_BYTES = 256 * 1024 * 1024
MAX_IMPORTED_ARTIFACT_MEMBERS = 128
MAX_IMPORTED_ARTIFACT_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_ARTIFACT_COMPONENT_BYTES = 256 * 1024 * 1024
MAX_ARTIFACT_CONTRACT_BYTES = 4 * 1024 * 1024
MAX_ARTIFACT_SUPPORT_FILE_BYTES = 32 * 1024 * 1024
TRAINABLE_SLOT_CONTRACT_KIND = "noema.trainable_slot_contract@1"
MODEL_SELECTION_HISTORY_KIND = "noema.model_selection_history"
PUBLICATION_SELECTION_ROLES = {"development_only", "adaptation_validation"}


class TrainedArtifactError(ValueError):
    pass


def validate_trained_artifact_publication_readiness(
    manifest_path: Path,
    *,
    project_root: Path,
    registry: Optional[OperationRegistry] = None,
) -> JsonDict:
    """Require the stronger publication boundary without changing runtime readiness.

    ``inspect_trained_artifact`` deliberately keeps ``ready`` as the compatibility
    and runtime verdict used by existing recipes.  This validator is for frozen
    publication protocols: it additionally requires a portable schema-v2 package
    and a hash-bound, validation-only history of every searched candidate.
    """

    inspected = inspect_trained_artifact(
        manifest_path,
        project_root=project_root,
        registry=registry,
    )
    if not inspected.get("publication_ready", False):
        raise TrainedArtifactError(
            "trained artifact is not publication-ready: %s"
            % "; ".join(
                str(item)
                for item in inspected.get("publication_issues") or []
            )
        )
    return inspected


def trained_artifact_source_class(
    source: Mapping[str, Any], manifest_path: Optional[Path] = None
) -> str:
    """Classify artifact provenance for presentation and benchmark semantics."""

    if manifest_path is not None:
        normalized = str(Path(manifest_path)).replace("\\", "/")
        if "/.noema/trained_artifacts/imported/" in "/%s" % normalized.lstrip("/"):
            return "imported"
    origin = str((source or {}).get("origin") or "").strip().lower()
    if origin in {
        "external_checkpoint_import",
        "external_artifact_import",
        "imported_artifact",
    }:
        return "imported"
    if origin in {
        "published_reference",
        "published_reference_checkpoint",
        "published_pretrained_reference",
    }:
        return "published_reference"
    if origin in {
        "noema_reference_baseline",
        "reference_baseline",
        "classical_reference_baseline",
    }:
        return "reference_baseline"
    return "project_trained"


def trained_artifact_recipe_compatibility_issues(
    artifact: Mapping[str, Any],
    recipe: Any,
) -> List[str]:
    """Return fixed-shape/protocol mismatches for a CSI-feedback artifact."""

    bindings = list(artifact.get("compatible_operations") or [])
    operations = {str(item.get("operation") or "") for item in bindings}
    if not {
        "model.csi_feedback_encoder",
        "model.csi_feedback_decoder",
    }.issubset(operations):
        return []

    raw_steps = getattr(recipe, "steps", None)
    if raw_steps is None and isinstance(recipe, Mapping):
        raw_steps = recipe.get("steps")
    steps = list(raw_steps or [])

    def operation_id(step: Any) -> str:
        return str(
            getattr(step, "op", None)
            or (step.get("op") if isinstance(step, Mapping) else "")
            or ""
        )

    def params(step: Any) -> Mapping[str, Any]:
        value = getattr(step, "params", None)
        if value is None and isinstance(step, Mapping):
            value = step.get("params")
        return value if isinstance(value, Mapping) else {}

    def find_step(operation: str) -> Any:
        return next((step for step in steps if operation_id(step) == operation), None)

    channel = find_step("wireless.miso_ofdm_csi")
    encoder_step = find_step("model.csi_feedback_encoder")
    decoder_step = find_step("model.csi_feedback_decoder")
    feedback_link = find_step("channel.csi_feedback_link")
    if not all((channel, encoder_step, decoder_step, feedback_link)):
        return [
            "CSI feedback artifact requires channel, encoder, feedback-link, and decoder blocks"
        ]

    encoder_binding = next(
        item
        for item in bindings
        if str(item.get("operation") or "") == "model.csi_feedback_encoder"
    )
    encoder_abi = encoder_binding.get("tensor_abi") or {}
    csi_tensor = _effective_entrypoint_tensor(encoder_abi, "inputs", "csi_ri")
    feedback_tensor = _effective_entrypoint_tensor(
        encoder_abi, "outputs", "feedback_code"
    )
    csi_shape = list(csi_tensor.get("shape") or [])
    feedback_shape = list(feedback_tensor.get("shape") or [])
    fixed_tx_antennas = csi_shape[2] if len(csi_shape) == 4 and isinstance(csi_shape[2], int) else None
    fixed_subcarriers = csi_shape[3] if len(csi_shape) == 4 and isinstance(csi_shape[3], int) else None
    fixed_feedback_dimension = (
        feedback_shape[-1]
        if feedback_shape and isinstance(feedback_shape[-1], int)
        else None
    )

    result: List[str] = []
    channel_params = params(channel)
    actual_tx_antennas = _optional_int(channel_params.get("tx_antennas"))
    actual_subcarriers = _optional_int(channel_params.get("ofdm_fft_size"))
    for fixed, actual, label in (
        (fixed_tx_antennas, actual_tx_antennas, "transmit antennas"),
        (fixed_subcarriers, actual_subcarriers, "OFDM subcarriers"),
    ):
        if fixed is None:
            continue
        if actual is None:
            result.append(
                "artifact requires %d %s; recipe has no valid integer value"
                % (fixed, label)
            )
        elif fixed != actual:
            result.append(
                "artifact requires %d %s; recipe has %d" % (fixed, label, actual)
            )

    for step, role in ((encoder_step, "encoder"), (decoder_step, "decoder")):
        configured_dimension = _optional_int(params(step).get("feedback_dimension"))
        if fixed_feedback_dimension is not None:
            if configured_dimension is None:
                result.append(
                    "artifact requires feedback dimension %d; recipe %s has no valid integer value"
                    % (fixed_feedback_dimension, role)
                )
            elif fixed_feedback_dimension != configured_dimension:
                result.append(
                    "artifact requires feedback dimension %d; recipe %s has %d"
                    % (fixed_feedback_dimension, role, configured_dimension)
                )

    feedback_constraint = dict((artifact.get("training") or {}).get("feedback_constraint") or {})
    expected_mode = str(feedback_constraint.get("mode") or "").strip()
    link_params = params(feedback_link)
    actual_mode = str(link_params.get("mode") or "ideal_noiseless").strip()
    if expected_mode and expected_mode != actual_mode:
        result.append(
            "artifact requires feedback transport %s; recipe uses %s"
            % (expected_mode, actual_mode)
        )
    bits_per_latent = _optional_int(link_params.get("bits_per_latent"))
    actual_bits = (
        fixed_feedback_dimension * bits_per_latent
        if fixed_feedback_dimension is not None
        and bits_per_latent is not None
        and actual_mode == "uniform_quantized"
        else None
    )
    raw_expected_bits = feedback_constraint.get("feedback_bits_per_sample")
    expected_bits = _optional_int(raw_expected_bits)
    if raw_expected_bits is not None and expected_bits is None:
        result.append(
            "artifact feedback_bits_per_sample constraint is not a valid integer"
        )
    if expected_bits is not None and actual_bits != expected_bits:
        result.append(
            "artifact requires %d feedback bits per CSI sample; recipe provides %s"
            % (expected_bits, str(actual_bits) if actual_bits is not None else "no finite bit budget")
        )
    return result


def _optional_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result == value else None


def import_external_trained_artifact(
    project_root: Path,
    source_path: Path,
    *,
    operation: str = "",
    original_filename: str = "",
    label: str = "",
    registry: Optional[OperationRegistry] = None,
) -> JsonDict:
    """Validate and register an externally trained checkpoint in this project.

    The caller supplies a temporary, server-side file. No path supplied by a web
    client is trusted or retained: the validated checkpoint is copied into the
    project-owned trained-artifact store and represented by the same portable,
    hash-pinned manifest used by exported training projects.

    Import adapters are intentionally operation-specific even though the resulting
    artifact contract and API are generic. This prevents a file extension alone from
    being treated as proof that a checkpoint is safe or compatible.
    """

    root = Path(project_root).resolve()
    source = Path(source_path).resolve()
    if (
        source.is_dir()
        or source.suffix.lower() in {".yaml", ".yml", ".json", ".zip", ".noema-artifact"}
    ):
        return import_external_trained_artifact_package(
            root,
            source,
            operation=operation,
            label=label,
            registry=registry,
        )
    operation_id = str(operation or "").strip()
    if operation_id in {
        "model.deepjscc_external_encode",
        "model.deepjscc_external_decode",
    }:
        return _import_external_deepjscc_artifact(
            root,
            source,
            original_filename=original_filename,
            label=label,
            registry=registry,
        )
    if operation_id != "model.symbol_power_allocator":
        raise TrainedArtifactError(
            "external checkpoint import is not available for operation: %s"
            % (operation_id or "<empty>")
        )
    if not source.is_file():
        raise TrainedArtifactError("external checkpoint upload is not a readable file")
    if source.suffix.lower() != ".npz":
        raise TrainedArtifactError("learned power-allocator checkpoints must use the safe .npz format")
    size_bytes = int(source.stat().st_size)
    if size_bytes <= 0 or size_bytes > MAX_IMPORTED_CHECKPOINT_BYTES:
        raise TrainedArtifactError(
            "external checkpoint size %d bytes is outside the allowed range (max %d)"
            % (size_bytes, MAX_IMPORTED_CHECKPOINT_BYTES)
        )
    _preflight_safe_npz(source)

    checkpoint_sha = _file_sha256(source)
    try:
        from noema_lab.ops.channel.power_allocator_checkpoint import (
            CHECKPOINT_FORMAT,
            load_csi_power_allocator_checkpoint,
        )

        checkpoint = load_csi_power_allocator_checkpoint(
            str(source),
            checkpoint_sha,
            strict=True,
            max_bytes=MAX_IMPORTED_CHECKPOINT_BYTES,
        )
    except Exception as exc:
        raise TrainedArtifactError("checkpoint validation failed: %s" % exc) from exc

    supplied_name = Path(str(original_filename or source.name)).name
    if not supplied_name.lower().endswith(".npz"):
        raise TrainedArtifactError("learned power-allocator checkpoints must use a .npz filename")
    name_stem = _safe_artifact_slug(Path(supplied_name).stem)
    artifact_dir = (
        root
        / ".noema"
        / "trained_artifacts"
        / "imported"
        / ("%s-%s" % (name_stem, checkpoint_sha[:12]))
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    managed_checkpoint = artifact_dir / "checkpoint.npz"
    if managed_checkpoint.is_file() and _file_sha256(managed_checkpoint) != checkpoint_sha:
        raise TrainedArtifactError(
            "managed artifact destination already exists with different contents: %s"
            % _project_path(managed_checkpoint, root)
        )
    if not managed_checkpoint.is_file():
        with tempfile.NamedTemporaryFile(
            prefix=".checkpoint-",
            suffix=".npz",
            dir=str(artifact_dir),
            delete=False,
        ) as handle:
            temporary_destination = Path(handle.name)
        try:
            shutil.copyfile(source, temporary_destination)
            if _file_sha256(temporary_destination) != checkpoint_sha:
                raise TrainedArtifactError("managed checkpoint copy failed its SHA-256 verification")
            os.replace(str(temporary_destination), str(managed_checkpoint))
        finally:
            if temporary_destination.exists():
                temporary_destination.unlink()

    display_name = _clean_display_text(label) or Path(supplied_name).stem or "Learned power allocator"
    artifact_id = "imported.symbol_power_allocator.%s.%s" % (
        name_stem.replace("-", "_"),
        checkpoint_sha[:12],
    )
    manifest = {
        "schema_version": 1,
        "kind": TRAINED_ARTIFACT_KIND,
        "id": artifact_id,
        "name": display_name,
        "label": display_name,
        "description": "Externally trained CSI-conditioned power allocator imported into this project.",
        "artifact": {
            "path": managed_checkpoint.name,
            "sha256": checkpoint_sha,
            "format": CHECKPOINT_FORMAT,
        },
        "compatible_operations": [
            {
                "operation": operation_id,
                "label": display_name,
                "description": "Frozen per-subcarrier learned power-allocation policy.",
                "required_inputs": ["channel_state"],
                "params": {
                    "policy": "learned_checkpoint",
                    "granularity": "per_subcarrier",
                    "budget_mode": "fixed_average",
                    "checkpoint_path": managed_checkpoint.name,
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_format": CHECKPOINT_FORMAT,
                    "checkpoint_strict": True,
                },
            }
        ],
        "source": {
            "origin": "external_checkpoint_import",
            "original_filename": supplied_name,
        },
        "training": dict(checkpoint.metadata.get("training") or {}),
    }
    manifest_path = artifact_dir / "trained_artifact.yaml"
    temporary_manifest: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=".trained-artifact-",
            suffix=".yaml",
            dir=str(artifact_dir),
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary_manifest = Path(handle.name)
            yaml.safe_dump(manifest, handle, sort_keys=False)
        os.replace(str(temporary_manifest), str(manifest_path))
    finally:
        if temporary_manifest is not None and temporary_manifest.exists():
            temporary_manifest.unlink()
    inspected = inspect_trained_artifact(
        manifest_path,
        project_root=root,
        registry=registry,
    )
    if not inspected.get("ready"):
        try:
            manifest_path.unlink()
        except OSError:
            pass
        raise TrainedArtifactError(
            "imported checkpoint could not be registered: %s"
            % "; ".join(str(item) for item in inspected.get("issues") or [])
        )
    return inspected


def _import_external_deepjscc_artifact(
    project_root: Path,
    source_path: Path,
    *,
    original_filename: str,
    label: str,
    registry: Optional[OperationRegistry],
) -> JsonDict:
    root = Path(project_root).resolve()
    source = Path(source_path).resolve()
    if not source.is_file():
        raise TrainedArtifactError("external checkpoint upload is not a readable file")
    if source.suffix.lower() != ".npz":
        raise TrainedArtifactError("learned DeepJSCC checkpoints must use the safe .npz format")
    size_bytes = int(source.stat().st_size)
    if size_bytes <= 0 or size_bytes > MAX_IMPORTED_CHECKPOINT_BYTES:
        raise TrainedArtifactError(
            "external checkpoint size %d bytes is outside the allowed range (max %d)"
            % (size_bytes, MAX_IMPORTED_CHECKPOINT_BYTES)
        )
    _preflight_safe_npz(source)

    checkpoint_sha = _file_sha256(source)
    try:
        from noema_lab.ops.models.deepjscc_checkpoint import (
            CHECKPOINT_FORMAT,
            load_deepjscc_reference_checkpoint,
        )

        checkpoint = load_deepjscc_reference_checkpoint(
            str(source),
            checkpoint_sha,
            checkpoint_format=CHECKPOINT_FORMAT,
            strict=True,
            max_bytes=MAX_IMPORTED_CHECKPOINT_BYTES,
        )
    except Exception as exc:
        raise TrainedArtifactError("checkpoint validation failed: %s" % exc) from exc

    supplied_name = Path(str(original_filename or source.name)).name
    if not supplied_name.lower().endswith(".npz"):
        raise TrainedArtifactError("learned DeepJSCC checkpoints must use a .npz filename")
    name_stem = _safe_artifact_slug(Path(supplied_name).stem)
    artifact_dir = (
        root
        / ".noema"
        / "trained_artifacts"
        / "imported"
        / ("deepjscc-%s-%s" % (name_stem, checkpoint_sha[:12]))
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    managed_checkpoint = artifact_dir / "checkpoint.npz"
    if managed_checkpoint.is_file() and _file_sha256(managed_checkpoint) != checkpoint_sha:
        raise TrainedArtifactError(
            "managed artifact destination already exists with different contents: %s"
            % _project_path(managed_checkpoint, root)
        )
    if not managed_checkpoint.is_file():
        with tempfile.NamedTemporaryFile(
            prefix=".checkpoint-",
            suffix=".npz",
            dir=str(artifact_dir),
            delete=False,
        ) as handle:
            temporary_destination = Path(handle.name)
        try:
            shutil.copyfile(source, temporary_destination)
            if _file_sha256(temporary_destination) != checkpoint_sha:
                raise TrainedArtifactError("managed checkpoint copy failed its SHA-256 verification")
            os.replace(str(temporary_destination), str(managed_checkpoint))
        finally:
            if temporary_destination.exists():
                temporary_destination.unlink()

    display_name = _clean_display_text(label) or Path(supplied_name).stem or "Learned DeepJSCC"
    artifact_id = "imported.deepjscc.%s.%s" % (
        name_stem.replace("-", "_"),
        checkpoint_sha[:12],
    )
    common_params = {
        "runtime": "learned_checkpoint",
        "checkpoint_path": managed_checkpoint.name,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "checkpoint_strict": True,
        "symbol_channels": int(checkpoint.symbol_channels),
    }
    binding_group = "deepjscc_sender_receiver"
    manifest = {
        "schema_version": 1,
        "kind": TRAINED_ARTIFACT_KIND,
        "id": artifact_id,
        "name": display_name,
        "label": display_name,
        "description": "Externally trained paired DeepJSCC sender/receiver imported into this project.",
        "application": {"mode": "all_group_bindings"},
        "artifact": {
            "path": managed_checkpoint.name,
            "sha256": checkpoint_sha,
            "format": CHECKPOINT_FORMAT,
        },
        "compatible_operations": [
            {
                "operation": "model.deepjscc_external_encode",
                "label": display_name,
                "description": "Frozen learned DeepJSCC image encoder.",
                "binding_group": binding_group,
                "role": "encoder",
                "preferred_step_id": "sender",
                "required_inputs": ["images"],
                "params": dict(common_params),
            },
            {
                "operation": "model.deepjscc_external_decode",
                "label": display_name,
                "description": "Frozen learned DeepJSCC image decoder.",
                "binding_group": binding_group,
                "role": "decoder",
                "preferred_step_id": "receiver",
                "required_inputs": ["symbols"],
                "params": dict(common_params),
            },
        ],
        "source": {
            "origin": "external_checkpoint_import",
            "original_filename": supplied_name,
        },
        "training": dict(checkpoint.metadata.get("training") or {}),
    }
    manifest_path = artifact_dir / "trained_artifact.yaml"
    temporary_manifest: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=".trained-artifact-",
            suffix=".yaml",
            dir=str(artifact_dir),
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary_manifest = Path(handle.name)
            yaml.safe_dump(manifest, handle, sort_keys=False)
        os.replace(str(temporary_manifest), str(manifest_path))
    finally:
        if temporary_manifest is not None and temporary_manifest.exists():
            temporary_manifest.unlink()
    inspected = inspect_trained_artifact(
        manifest_path,
        project_root=root,
        registry=registry,
    )
    if not inspected.get("ready"):
        try:
            manifest_path.unlink()
        except OSError:
            pass
        raise TrainedArtifactError(
            "imported checkpoint could not be registered: %s"
            % "; ".join(str(item) for item in inspected.get("issues") or [])
        )
    return inspected


def import_external_trained_artifact_package(
    project_root: Path,
    source_path: Path,
    *,
    operation: str = "",
    label: str = "",
    registry: Optional[OperationRegistry] = None,
) -> JsonDict:
    """Validate and atomically register a schema-v2 executable artifact package.

    ``source_path`` may be a package directory, its manifest, or a ZIP archive.
    ZIP members are bounded and extracted without trusting member paths.  Only the
    manifest, contract, and declared component files are copied into the managed
    store, so executable graph bytes cannot arrive without a corresponding hash.
    """

    root = Path(project_root).resolve()
    source = Path(source_path).resolve()
    if not source.exists():
        raise TrainedArtifactError("trained artifact package does not exist: %s" % source)
    active_registry = registry
    if active_registry is None:
        # Managed imports must prove operation-owned binding safety. A caller may
        # supply a plugin-aware registry; otherwise use Noema's standard registry
        # rather than treating unverified package parameters as directly applicable.
        from noema_lab.ops import build_registry

        active_registry = build_registry()

    temporary_root: Optional[tempfile.TemporaryDirectory] = None
    try:
        if source.is_dir():
            manifest_path = _locate_v2_package_manifest(source)
            _assert_path_within(
                manifest_path,
                source,
                "trained artifact manifest escapes the selected package directory",
            )
        elif source.suffix.lower() in {".yaml", ".yml", ".json"}:
            manifest_path = source
        else:
            if int(source.stat().st_size) > MAX_IMPORTED_ARTIFACT_PACKAGE_BYTES:
                raise TrainedArtifactError(
                    "trained artifact package exceeds the %d-byte limit"
                    % MAX_IMPORTED_ARTIFACT_PACKAGE_BYTES
                )
            temporary_root = tempfile.TemporaryDirectory(prefix="noema-trained-artifact-")
            extracted_root = Path(temporary_root.name)
            _extract_safe_artifact_archive(source, extracted_root)
            manifest_path = _locate_v2_package_manifest(extracted_root)
            _assert_path_within(
                manifest_path,
                extracted_root,
                "trained artifact manifest escapes the extracted package",
            )

        # Relative paths in a package are always relative to the manifest, whether
        # the caller selected that directory directly or selected an ancestor that
        # contains one nested package.
        manifest_path = Path(manifest_path).resolve()
        package_source = manifest_path.parent

        payload = _load_mapping(manifest_path)
        if int(payload.get("schema_version") or 0) != 2:
            raise TrainedArtifactError(
                "generic executable artifact packages require schema_version: 2"
            )
        inspected_source = inspect_trained_artifact(
            manifest_path,
            project_root=package_source,
            registry=active_registry,
        )
        if not inspected_source.get("valid", False):
            raise TrainedArtifactError(
                "trained artifact package is invalid: %s"
                % "; ".join(str(item) for item in inspected_source.get("issues") or [])
            )
        requested_operation = str(operation or "").strip()
        if requested_operation and not any(
            str(binding.get("operation") or "") == requested_operation
            for binding in inspected_source.get("compatible_operations") or []
        ):
            raise TrainedArtifactError(
                "trained artifact package is not compatible with operation: %s"
                % requested_operation
            )

        artifact_id = str(inspected_source.get("id") or "trained-artifact")
        identity_digest = _v2_package_identity(inspected_source)
        upstream_source = payload.get("source")
        upstream_source = (
            dict(upstream_source)
            if isinstance(upstream_source, Mapping)
            else {}
        )
        managed_payload = dict(payload)
        managed_payload.pop("publication", None)
        managed_payload["source"] = {
            "origin": "external_artifact_import",
            "imported_from_runtime_identity_sha256": identity_digest,
            "upstream_claimed_source": upstream_source,
        }
        managed_store = _managed_store_path(root, ".noema", "trained_artifacts")
        imported_parent = _managed_store_path(
            root, ".noema", "trained_artifacts", "imported"
        )
        staging_parent = _managed_store_path(
            root, ".noema", "trained_artifacts", ".staging"
        )
        artifact_dir = _managed_store_path(
            root,
            ".noema",
            "trained_artifacts",
            "imported",
            "%s-%s" % (_safe_artifact_slug(artifact_id), identity_digest[:12]),
        )
        if artifact_dir.is_dir():
            existing_manifest = _locate_v2_package_manifest(artifact_dir)
            existing = inspect_trained_artifact(
                existing_manifest,
                project_root=root,
                registry=active_registry,
            )
            existing_source = existing.get("source") or {}
            imported_from = (
                str(
                    existing_source.get(
                        "imported_from_runtime_identity_sha256"
                    )
                    or ""
                )
                if isinstance(existing_source, Mapping)
                else ""
            )
            if existing.get("valid") and (
                imported_from == identity_digest
                or _v2_package_identity(existing) == identity_digest
            ):
                return existing
            raise TrainedArtifactError(
                "managed artifact destination already exists with different contents: %s"
                % _project_path(artifact_dir, root)
            )

        staging_parent.mkdir(parents=True, exist_ok=True)
        # mkdir follows symlinks if an ancestor is exchanged concurrently. Recheck
        # the complete path before placing any package bytes below it.
        _assert_managed_store_path(root, managed_store)
        _assert_managed_store_path(root, imported_parent)
        _assert_managed_store_path(root, staging_parent)
        staged_dir = Path(tempfile.mkdtemp(prefix="artifact-", dir=str(staging_parent)))
        try:
            declared_files = _v2_declared_package_files(payload, package_source)
            for relative_path, source_file in declared_files:
                destination = (staged_dir / relative_path).resolve()
                try:
                    destination.relative_to(staged_dir.resolve())
                except ValueError as exc:  # Defensive; paths were already inspected.
                    raise TrainedArtifactError(
                        "declared artifact file escapes the managed package: %s"
                        % relative_path
                    ) from exc
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_file, destination)

            # Keep the package's descriptive identity and executable declarations,
            # but establish a project-owned trust boundary for provenance. An
            # uploaded package cannot promote itself to a published reference or
            # carry self-asserted publication readiness into the managed store.
            managed_manifest = staged_dir / "trained_artifact.yaml"
            managed_manifest.write_text(
                yaml.safe_dump(managed_payload, sort_keys=False),
                encoding="utf-8",
            )
            staged_inspection = inspect_trained_artifact(
                managed_manifest,
                project_root=root,
                registry=active_registry,
            )
            if not staged_inspection.get("valid", False):
                raise TrainedArtifactError(
                    "managed artifact validation failed: %s"
                    % "; ".join(str(item) for item in staged_inspection.get("issues") or [])
                )
            artifact_dir.parent.mkdir(parents=True, exist_ok=True)
            _assert_managed_store_path(root, artifact_dir.parent)
            _assert_managed_store_path(root, artifact_dir)
            os.replace(str(staged_dir), str(artifact_dir))
            final_manifest = artifact_dir / "trained_artifact.yaml"
            return inspect_trained_artifact(
                final_manifest,
                project_root=root,
                registry=active_registry,
            )
        finally:
            if staged_dir.exists():
                shutil.rmtree(staged_dir)
            try:
                staging_parent.rmdir()
            except OSError:
                pass
    finally:
        if temporary_root is not None:
            temporary_root.cleanup()


def validate_trained_artifact_package(
    source_path: Path,
    *,
    registry: Optional[OperationRegistry] = None,
) -> JsonDict:
    """Validate a schema-v2 package without registering it."""

    source = Path(source_path).resolve()
    temporary_root: Optional[tempfile.TemporaryDirectory] = None
    try:
        if source.is_dir():
            manifest_path = _locate_v2_package_manifest(source)
            _assert_path_within(
                manifest_path,
                source,
                "trained artifact manifest escapes the selected package directory",
            )
            manifest_path = Path(manifest_path).resolve()
            package_root = manifest_path.parent
        elif source.suffix.lower() in {".yaml", ".yml", ".json"}:
            manifest_path = source
            package_root = source.parent
        else:
            if not source.is_file() or int(source.stat().st_size) > MAX_IMPORTED_ARTIFACT_PACKAGE_BYTES:
                raise TrainedArtifactError("trained artifact package is missing or too large")
            temporary_root = tempfile.TemporaryDirectory(prefix="noema-trained-artifact-validate-")
            extracted_root = Path(temporary_root.name)
            _extract_safe_artifact_archive(source, extracted_root)
            manifest_path = _locate_v2_package_manifest(extracted_root)
            _assert_path_within(
                manifest_path,
                extracted_root,
                "trained artifact manifest escapes the extracted package",
            )
            manifest_path = Path(manifest_path).resolve()
            package_root = manifest_path.parent
        payload = _load_mapping(manifest_path)
        if int(payload.get("schema_version") or 0) != 2:
            raise TrainedArtifactError(
                "generic executable artifact packages require schema_version: 2"
            )
        return inspect_trained_artifact(
            manifest_path,
            project_root=package_root,
            registry=registry,
        )
    finally:
        if temporary_root is not None:
            temporary_root.cleanup()


def _assert_path_within(path: Path, root: Path, message: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise TrainedArtifactError(message) from exc
    return resolved


def _managed_store_path(project_root: Path, *parts: str) -> Path:
    candidate = Path(project_root).resolve().joinpath(*parts)
    _assert_managed_store_path(project_root, candidate)
    return candidate


def _assert_managed_store_path(project_root: Path, candidate: Path) -> None:
    """Reject project-store paths redirected through pre-existing symlinks."""

    root = Path(project_root).resolve()
    path = Path(candidate)
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise TrainedArtifactError(
            "managed trained-artifact path escapes the project root: %s" % path
        ) from exc
    current = root
    for part in relative.parts:
        current = current / part
        # is_symlink also detects dangling links, which Path.exists omits.
        if current.is_symlink():
            raise TrainedArtifactError(
                "managed trained-artifact store must not contain symbolic links: %s"
                % current
            )
        try:
            current.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise TrainedArtifactError(
                "managed trained-artifact path escapes the project root: %s" % current
            ) from exc


def _locate_v2_package_manifest(package_root: Path) -> Path:
    root = Path(package_root).resolve()
    direct = [root / filename for filename in TRAINED_ARTIFACT_FILENAMES]
    direct_matches = [path for path in direct if path.is_file()]
    if len(direct_matches) == 1:
        return direct_matches[0]
    if len(direct_matches) > 1:
        raise TrainedArtifactError(
            "trained artifact package contains multiple root manifests"
        )
    matches = []
    for filename in TRAINED_ARTIFACT_FILENAMES:
        matches.extend(path for path in root.rglob(filename) if path.is_file())
    unique = sorted({path.resolve() for path in matches})
    if len(unique) != 1:
        raise TrainedArtifactError(
            "trained artifact package must contain exactly one trained_artifact manifest"
        )
    return unique[0]


def _extract_safe_artifact_archive(source: Path, destination: Path) -> None:
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise TrainedArtifactError(
            "trained artifact package must be a directory, manifest, or valid ZIP archive"
        )
    try:
        with zipfile.ZipFile(source, "r") as archive:
            members = archive.infolist()
            if not members or len(members) > MAX_IMPORTED_ARTIFACT_MEMBERS:
                raise TrainedArtifactError(
                    "trained artifact archive must contain between 1 and %d members"
                    % MAX_IMPORTED_ARTIFACT_MEMBERS
                )
            expanded_size = sum(max(0, int(member.file_size)) for member in members)
            if expanded_size > MAX_IMPORTED_ARTIFACT_UNCOMPRESSED_BYTES:
                raise TrainedArtifactError(
                    "trained artifact archive expands to %d bytes; maximum allowed is %d"
                    % (expanded_size, MAX_IMPORTED_ARTIFACT_UNCOMPRESSED_BYTES)
                )
            root = Path(destination).resolve()
            seen_exact_names = set()
            seen_portable_names = set()
            for member in members:
                if member.flag_bits & 0x1:
                    raise TrainedArtifactError(
                        "encrypted trained artifact archive members are not supported"
                    )
                raw_name = str(member.filename or "")
                pure = PurePosixPath(raw_name)
                unsafe_windows_part = any(
                    ":" in part
                    or part.rstrip(" .") != part
                    or part.split(".", 1)[0].casefold()
                    in {
                        "con",
                        "prn",
                        "aux",
                        "nul",
                        *("com%d" % index for index in range(1, 10)),
                        *("lpt%d" % index for index in range(1, 10)),
                    }
                    for part in pure.parts
                )
                if (
                    not raw_name
                    or "\\" in raw_name
                    or pure.is_absolute()
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or unsafe_windows_part
                ):
                    raise TrainedArtifactError(
                        "trained artifact archive contains an unsafe member path: %s"
                        % raw_name
                    )
                if raw_name in seen_exact_names:
                    raise TrainedArtifactError(
                        "trained artifact archive contains duplicate members"
                    )
                portable_name = unicodedata.normalize(
                    "NFC", pure.as_posix()
                ).casefold()
                if portable_name in seen_portable_names:
                    raise TrainedArtifactError(
                        "trained artifact archive contains colliding member paths"
                    )
                seen_exact_names.add(raw_name)
                seen_portable_names.add(portable_name)
                mode = int(member.external_attr >> 16)
                file_type = stat.S_IFMT(mode)
                if stat.S_ISLNK(mode):
                    raise TrainedArtifactError(
                        "trained artifact archive must not contain symbolic links"
                    )
                if member.is_dir():
                    if file_type not in {0, stat.S_IFDIR}:
                        raise TrainedArtifactError(
                            "trained artifact archive contains a special-file member"
                        )
                elif file_type not in {0, stat.S_IFREG}:
                    raise TrainedArtifactError(
                        "trained artifact archive contains a special-file member"
                    )
                target = (root / Path(*pure.parts)).resolve()
                try:
                    target.relative_to(root)
                except ValueError as exc:
                    raise TrainedArtifactError(
                        "trained artifact archive member escapes the package: %s"
                        % raw_name
                    ) from exc
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source_handle, target.open("wb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
    except TrainedArtifactError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise TrainedArtifactError("trained artifact package is not a valid ZIP archive") from exc


def _v2_declared_package_files(
    payload: Mapping[str, Any],
    package_root: Path,
) -> List[tuple[Path, Path]]:
    raw_paths = [str(_required_mapping(payload, "contract").get("path") or "")]
    raw_components = payload.get("components") or []
    raw_paths.extend(
        str(component.get("path") or "")
        for component in raw_components
        if isinstance(component, Mapping)
    )
    raw_support_files = payload.get("support_files") or []
    raw_paths.extend(
        str(support_file.get("path") or "")
        for support_file in raw_support_files
        if isinstance(support_file, Mapping)
    )
    result: List[tuple[Path, Path]] = []
    seen = set()
    root = Path(package_root).resolve()
    for raw in raw_paths:
        relative = Path(raw)
        if not raw or relative.is_absolute():
            raise TrainedArtifactError(
                "schema-v2 contract and component paths must be relative"
            )
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise TrainedArtifactError(
                "declared artifact file escapes the package: %s" % raw
            ) from exc
        if source in seen:
            raise TrainedArtifactError("declared artifact file is duplicated: %s" % raw)
        seen.add(source)
        if not source.is_file():
            raise TrainedArtifactError("declared artifact file is missing: %s" % raw)
        result.append((relative, source))
    return result


def _runtime_identity_sha256(inspected: Mapping[str, Any]) -> str:
    """Return the canonical runtime identity without changing legacy hash bytes.

    This identity covers the manifest, contract, runtime components, and support
    files that define an executable schema-v2 package.  It is deliberately not
    the SHA-256 of an archive file or directory serialization, and it is not the
    component-set identity used during model selection.
    """

    contract = dict(inspected.get("contract") or {})
    artifact = dict(inspected.get("artifact") or {})
    identity = {
        "schema_version": 2,
        "kind": TRAINED_ARTIFACT_KIND,
        # The exact manifest covers ABI, operation bindings, and provenance.
        "manifest_sha256": str(
            artifact.get("actual_sha256") or artifact.get("sha256") or ""
        ),
        "contract": {
            "semantic_sha256": str(
                contract.get("actual_sha256") or contract.get("sha256") or ""
            ),
            "file_sha256": str(
                contract.get("actual_file_sha256")
                or contract.get("file_sha256")
                or ""
            ),
        },
        "components": sorted(
            (
                {
                    "id": str(item.get("id") or ""),
                    "role": str(item.get("role") or ""),
                    "format": str(item.get("format") or ""),
                    "sha256": str(
                        item.get("actual_sha256") or item.get("sha256") or ""
                    ),
                }
                for item in inspected.get("components") or []
            ),
            key=lambda item: item["id"],
        ),
        "support_files": sorted(
            (
                {
                    "role": str(item.get("role") or ""),
                    "path": str(item.get("package_path") or ""),
                    "sha256": str(
                        item.get("actual_sha256") or item.get("sha256") or ""
                    ),
                }
                for item in inspected.get("support_files") or []
            ),
            key=lambda item: (item["role"], item["path"], item["sha256"]),
        ),
    }
    try:
        encoded = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise TrainedArtifactError(
            "trained artifact identity fields must contain finite, acyclic, "
            "JSON-compatible values"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _v2_package_identity(inspected: Mapping[str, Any]) -> str:
    """Return the legacy ``package_sha256`` runtime identity.

    Keep this private alias so existing callers and persisted execution plans
    retain exactly the same compatibility semantics and digest bytes.
    """

    return _runtime_identity_sha256(inspected)


def discover_trained_artifacts(
    project_root: Path,
    *,
    registry: Optional[OperationRegistry] = None,
    operation: str = "",
) -> List[JsonDict]:
    """Discover hash-pinned trained block artifacts registered in this project.

    Discovery deliberately uses a small set of project-owned roots instead of walking
    virtual environments, run bundles, or arbitrary user directories.
    """

    root = Path(project_root).resolve()
    operation_id = str(operation or "").strip()
    rows = []
    for manifest_path in _manifest_paths(root):
        try:
            row = inspect_trained_artifact(
                manifest_path,
                project_root=root,
                registry=registry,
            )
        except Exception as exc:
            row = {
                "schema_version": 1,
                "kind": TRAINED_ARTIFACT_KIND,
                "id": manifest_path.parent.name,
                "name": manifest_path.parent.name,
                "label": manifest_path.parent.name,
                "manifest_path": _project_path(manifest_path, root),
                "status": "invalid",
                "ready": False,
                "issues": [str(exc)],
                "publication_status": "blocked",
                "publication_ready": False,
                "publication_issues": [
                    "artifact inspection failed before publication readiness could be assessed"
                ],
                "publication_readiness": {
                    "status": "blocked",
                    "ready": False,
                    "issues": [
                        "artifact inspection failed before publication readiness could be assessed"
                    ],
                },
                "artifact": {},
                "compatible_operations": [],
                "source": {},
                "training": {},
            }
        if operation_id and not any(
            str(binding.get("operation") or "") == operation_id
            for binding in row.get("compatible_operations") or []
        ):
            continue
        rows.append(row)
    return sorted(rows, key=lambda item: (str(item.get("label") or ""), str(item.get("id") or "")))


def _runtime_component_set_sha256(components: Sequence[Mapping[str, Any]]) -> str:
    """Content identity used to bind a selected candidate to deployed components."""

    identity = {
        "schema_version": 1,
        "kind": "noema.trained_artifact_runtime_components",
        "components": sorted(
            (
                {
                    "id": str(item.get("id") or ""),
                    "role": str(item.get("role") or ""),
                    "format": str(item.get("format") or ""),
                    "sha256": str(
                        item.get("actual_sha256") or item.get("sha256") or ""
                    ),
                }
                for item in components
            ),
            key=lambda item: item["id"],
        ),
    }
    return hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def model_selection_history_issues(
    document: Mapping[str, Any],
    *,
    expected_runtime_component_set_sha256: str = "",
) -> List[str]:
    """Validate the canonical, publication-only model-selection history shape.

    The selected candidate binds ``runtime_component_set_sha256``.  It must not
    bind a package/file digest (which changes with packaging) or the full runtime
    identity (which includes the history itself and would create a hash cycle).
    Filesystem confinement and support-file membership are checked by each
    caller because core packages and paper evidence have different roots.
    """

    issues: List[str] = []

    def add(message: str) -> None:
        if message not in issues:
            issues.append(message)

    def text(value: Any) -> str:
        return str(value or "").strip()

    def valid_sha256(value: Any) -> bool:
        value_text = text(value)
        return len(value_text) == 64 and all(
            char in "0123456789abcdef" for char in value_text
        )

    if document.get("schema_version") != 1:
        add("selection history schema_version must be 1")
    if document.get("kind") != MODEL_SELECTION_HISTORY_KIND:
        add("selection history kind must be %s" % MODEL_SELECTION_HISTORY_KIND)
    if document.get("all_candidates_disclosed") is not True:
        add("selection history must attest all_candidates_disclosed=true")
    if document.get("publication_test_accessed") is not False:
        add(
            "selection history publication_test_accessed must be false; "
            "publication-test access is forbidden during model selection"
        )

    population_role = text(document.get("selection_population_role"))
    if population_role not in PUBLICATION_SELECTION_ROLES:
        add(
            "selection_population_role must be development_only or "
            "adaptation_validation"
        )
    population = document.get("selection_population")
    population_sha = ""
    if not isinstance(population, Mapping):
        add("selection_population must be a content-identified mapping")
    else:
        for field in ("dataset_id", "version", "split", "role"):
            if not text(population.get(field)):
                add("selection_population.%s must not be empty" % field)
        if text(population.get("role")) != population_role:
            add("selection_population.role must match selection_population_role")
        population_sha = text(population.get("sha256"))
        if not valid_sha256(population_sha):
            add(
                "selection_population.sha256 must be a 64-character lowercase "
                "SHA-256"
            )

    for field in ("objective", "selected_candidate_id"):
        if not text(document.get(field)):
            add("selection history %s must not be empty" % field)
    direction = text(document.get("direction"))
    if direction not in {"minimize", "maximize"}:
        add("selection history direction must be minimize or maximize")

    tolerance = 0.0
    selection_rule = document.get("selection_rule")
    if not isinstance(selection_rule, Mapping):
        add("selection_rule must be a structured mapping")
    else:
        if selection_rule.get("method") != "objective_extremum":
            add("selection_rule.method must be objective_extremum")
        if selection_rule.get("tie_breaker") != "lexicographic_candidate_id":
            add(
                "selection_rule.tie_breaker must be "
                "lexicographic_candidate_id"
            )
        raw_tolerance = selection_rule.get("tolerance", 0.0)
        if (
            isinstance(raw_tolerance, bool)
            or not isinstance(raw_tolerance, (int, float))
            or not math.isfinite(float(raw_tolerance))
            or float(raw_tolerance) < 0.0
        ):
            add("selection_rule.tolerance must be finite and non-negative")
        else:
            tolerance = float(raw_tolerance)

    search_program = document.get("search_program")
    if not isinstance(search_program, Mapping):
        add("selection history search_program must be a mapping")
    else:
        search_path = text(search_program.get("path"))
        if not search_path:
            add("selection history search_program.path must not be empty")
        if not valid_sha256(search_program.get("sha256")):
            add(
                "selection history search_program.sha256 must be a "
                "64-character lowercase SHA-256"
            )
        command = search_program.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(token, str) and token for token in command
        ):
            add("selection history search_program.command must be a non-empty argv list")
        elif search_path and search_path not in command:
            add("selection history search_program.command must invoke search_program.path")

    raw_candidates = document.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        add("selection history candidates must be a non-empty list")
        raw_candidates = []
    candidates: Dict[str, Mapping[str, Any]] = {}
    completed: List[Mapping[str, Any]] = []
    for index, raw_candidate in enumerate(raw_candidates):
        label = "selection history candidates[%d]" % index
        if not isinstance(raw_candidate, Mapping):
            add("%s must be a mapping" % label)
            continue
        candidate_id = text(raw_candidate.get("id"))
        if not candidate_id:
            add("%s.id must not be empty" % label)
        elif candidate_id in candidates:
            add("%s.id is duplicated" % label)
        else:
            candidates[candidate_id] = raw_candidate

        if "configuration_sha256" in raw_candidate:
            add(
                "%s.configuration_sha256 is ambiguous; use "
                "configuration_identity_sha256" % label
            )
        if "artifact_sha256" in raw_candidate:
            add(
                "%s.artifact_sha256 is ambiguous; use "
                "runtime_component_set_sha256" % label
            )
        if not text(raw_candidate.get("configuration_path")):
            add("%s.configuration_path must not be empty" % label)
        if not valid_sha256(raw_candidate.get("configuration_identity_sha256")):
            add(
                "%s.configuration_identity_sha256 must be a 64-character "
                "lowercase SHA-256" % label
            )
        candidate_population_sha = text(
            raw_candidate.get("selection_population_sha256")
        )
        if not population_sha or candidate_population_sha != population_sha:
            add(
                "%s.selection_population_sha256 must match the validation "
                "population" % label
            )
        if raw_candidate.get("publication_test_accessed") is not False:
            add("%s must declare publication_test_accessed=false" % label)

        status = text(raw_candidate.get("status"))
        if status not in {"completed", "failed", "resource_rejected"}:
            add("%s.status is invalid" % label)
        if status == "completed":
            if not valid_sha256(raw_candidate.get("runtime_component_set_sha256")):
                add(
                    "%s.runtime_component_set_sha256 must identify the completed "
                    "candidate" % label
                )
            objective_value = raw_candidate.get("objective_value")
            if (
                not isinstance(objective_value, (int, float))
                or isinstance(objective_value, bool)
                or not math.isfinite(float(objective_value))
            ):
                add("%s.objective_value must be finite" % label)
            else:
                completed.append(raw_candidate)
        elif status in {"failed", "resource_rejected"} and not text(
            raw_candidate.get("failure_reason")
        ):
            add("%s.failure_reason must not be empty" % label)

    selected_id = text(document.get("selected_candidate_id"))
    selected = candidates.get(selected_id)
    if selected is None:
        add("selected_candidate_id is not a disclosed candidate")
    elif text(selected.get("status")) != "completed":
        add("selected_candidate_id must name a completed candidate")
    else:
        selected_components_sha = text(selected.get("runtime_component_set_sha256"))
        if (
            expected_runtime_component_set_sha256
            and selected_components_sha != expected_runtime_component_set_sha256
        ):
            add(
                "selected candidate runtime_component_set_sha256 does not bind "
                "the deployed runtime components"
            )
        if completed and direction in {"minimize", "maximize"}:
            values = [float(candidate["objective_value"]) for candidate in completed]
            optimum = min(values) if direction == "minimize" else max(values)
            tied = sorted(
                text(candidate.get("id"))
                for candidate in completed
                if abs(float(candidate["objective_value"]) - optimum) <= tolerance
            )
            expected_selected = tied[0] if tied else ""
            if selected_id != expected_selected:
                add(
                    "selected_candidate_id violates the frozen objective/tie rule; "
                    "expected %r" % expected_selected
                )

    return issues


def _inspect_publication_readiness_v2(
    payload: Mapping[str, Any],
    *,
    package_root: Path,
    project_root: Path,
    components: Sequence[Mapping[str, Any]],
    support_files: Sequence[Mapping[str, Any]],
    source_class: str,
) -> JsonDict:
    """Inspect publication-only model-selection evidence.

    These failures never change the compatibility/runtime ``ready`` verdict.  A
    development artifact remains runnable while publication protocols can require
    this independently auditable boundary.
    """

    issues: List[str] = []

    def add_issue(message: str) -> None:
        if message not in issues:
            issues.append(message)

    def valid_sha256(value: Any) -> bool:
        text = str(value or "").strip()
        return len(text) == 64 and all(char in "0123456789abcdef" for char in text)

    if source_class == "imported":
        add_issue(
            "imported artifacts cannot self-assert publication readiness; "
            "trusted local publication selection evidence is required"
        )

    runtime_component_set_sha256 = _runtime_component_set_sha256(components)
    support_by_path = {
        str(item.get("package_path") or "").strip(): item
        for item in support_files
        if str(item.get("package_path") or "").strip()
    }

    def verify_support_reference(
        raw_path: Any,
        expected_sha: Any,
        *,
        role: str,
        label: str,
    ) -> Optional[Path]:
        path_text = str(raw_path or "").strip()
        sha_text = str(expected_sha or "").strip()
        if not valid_sha256(sha_text):
            add_issue("%s.sha256 must be a 64-character lowercase SHA-256" % label)
        path_issues: List[str] = []
        resolved = _confined_package_path(
            package_root,
            path_text,
            "%s.path" % label,
            path_issues,
        )
        for item in path_issues:
            add_issue(item)
        support = support_by_path.get(path_text)
        if not isinstance(support, Mapping):
            add_issue("%s.path must be declared in support_files" % label)
        else:
            if str(support.get("role") or "") != role:
                add_issue("%s support file must use role=%s" % (label, role))
            if str(support.get("sha256") or "") != sha_text:
                add_issue("%s.sha256 must match its support_files declaration" % label)
            if list(support.get("issues") or []):
                add_issue("%s support file failed artifact integrity validation" % label)
        if resolved is not None:
            if resolved.is_symlink() or not resolved.is_file():
                add_issue("%s.path must be a regular package file" % label)
            elif valid_sha256(sha_text) and _file_sha256(resolved) != sha_text:
                add_issue("%s SHA-256 does not match its file" % label)
        return resolved

    raw_publication = payload.get("publication")
    if not isinstance(raw_publication, Mapping):
        raw_publication = {}
        add_issue("publication.selection_history is required")
    raw_reference = raw_publication.get("selection_history")
    if not isinstance(raw_reference, Mapping):
        raw_reference = {}
        add_issue("publication.selection_history must be a mapping")
    history_path = verify_support_reference(
        raw_reference.get("path"),
        raw_reference.get("sha256"),
        role="model_selection_history",
        label="publication.selection_history",
    )
    history: JsonDict = {}
    if history_path is not None and history_path.is_file() and not history_path.is_symlink():
        try:
            history = _load_mapping(history_path)
            json.dumps(
                history,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except Exception as exc:
            add_issue("publication.selection_history is invalid: %s" % exc)
            history = {}

    if history:
        for issue in model_selection_history_issues(
            history,
            expected_runtime_component_set_sha256=runtime_component_set_sha256,
        ):
            add_issue(issue)

        search_program = history.get("search_program")
        if isinstance(search_program, Mapping):
            verify_support_reference(
                search_program.get("path"),
                search_program.get("sha256"),
                role="model_selection_search_program",
                label="selection history search_program",
            )

        raw_candidates = history.get("candidates")
        if isinstance(raw_candidates, list):
            for index, raw_candidate in enumerate(raw_candidates):
                if not isinstance(raw_candidate, Mapping):
                    continue
                label = "selection history candidates[%d]" % index
                verify_support_reference(
                    raw_candidate.get("configuration_path"),
                    raw_candidate.get("configuration_identity_sha256"),
                    role="model_selection_configuration",
                    label="%s configuration" % label,
                )

    ready = not issues
    return {
        "status": "ready" if ready else "blocked",
        "ready": ready,
        "issues": issues,
        "runtime_component_set_sha256": runtime_component_set_sha256,
        # Backward-compatible publication API alias. New histories must use the
        # explicitly named runtime_component_set_sha256 field.
        "runtime_artifact_sha256": runtime_component_set_sha256,
        "selection_history": {
            "path": (
                _project_path(history_path, project_root)
                if history_path is not None
                else str(raw_reference.get("path") or "")
            ),
            "sha256": str(raw_reference.get("sha256") or ""),
            "actual_sha256": (
                _file_sha256(history_path)
                if history_path is not None
                and history_path.is_file()
                and not history_path.is_symlink()
                else ""
            ),
        },
        "selected_candidate_id": str(history.get("selected_candidate_id") or ""),
    }


def inspect_trained_artifact(
    manifest_path: Path,
    *,
    project_root: Path,
    registry: Optional[OperationRegistry] = None,
) -> JsonDict:
    root = Path(project_root).resolve()
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise TrainedArtifactError("trained artifact manifest is missing: %s" % path)
    payload = _load_mapping(path)
    schema_version = int(payload.get("schema_version") or 0)
    if schema_version == 2:
        return _inspect_trained_artifact_v2(
            path,
            payload,
            project_root=root,
            registry=registry,
        )
    if schema_version != 1:
        raise TrainedArtifactError(
            "trained artifact manifest requires supported schema_version 1 or 2"
        )
    if str(payload.get("kind") or "") != TRAINED_ARTIFACT_KIND:
        raise TrainedArtifactError(
            "unsupported trained artifact kind: %s" % str(payload.get("kind") or "")
        )

    artifact_id = _required_text(payload, "id")
    name = _required_text(payload, "name")
    label = str(payload.get("label") or name).strip()
    artifact_config = _required_mapping(payload, "artifact")
    raw_artifact_path = _required_text(artifact_config, "path", "artifact.path")
    artifact_path = Path(raw_artifact_path).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = (path.parent / artifact_path).resolve()
    else:
        artifact_path = artifact_path.resolve()
    expected_sha = _required_text(artifact_config, "sha256", "artifact.sha256").lower()
    artifact_format = _required_text(artifact_config, "format", "artifact.format")
    issues = []
    if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
        issues.append("artifact.sha256 must be a 64-character lowercase SHA-256")
    actual_sha = ""
    size_bytes = 0
    if not artifact_path.is_file():
        issues.append("trained artifact file is missing")
    else:
        size_bytes = int(artifact_path.stat().st_size)
        actual_sha = _file_sha256(artifact_path)
        if expected_sha and actual_sha != expected_sha:
            issues.append("trained artifact SHA-256 does not match the manifest")

    raw_application = payload.get("application")
    if raw_application is None:
        application: JsonDict = {}
    elif not isinstance(raw_application, Mapping):
        raise TrainedArtifactError("application must be a mapping")
    else:
        application = dict(raw_application)
    application_mode = _optional_text(application, "mode", "application.mode")
    if application_mode and application_mode not in {"single_binding", "all_group_bindings"}:
        issues.append(
            "application.mode must be single_binding or all_group_bindings"
        )
    if application_mode:
        application["mode"] = application_mode

    support_files = []
    support_paths: Dict[Path, JsonDict] = {}
    raw_support_files = payload.get("support_files") or []
    if not isinstance(raw_support_files, list):
        raise TrainedArtifactError("support_files must be a list")
    for index, raw_support_file in enumerate(raw_support_files):
        if not isinstance(raw_support_file, Mapping):
            raise TrainedArtifactError("support_files[%d] must be a mapping" % index)
        raw_support_path = _required_text(
            raw_support_file,
            "path",
            "support_files[%d].path" % index,
        )
        support_path = Path(raw_support_path).expanduser()
        if not support_path.is_absolute():
            support_path = (path.parent / support_path).resolve()
        else:
            support_path = support_path.resolve()
        support_sha = _required_text(
            raw_support_file,
            "sha256",
            "support_files[%d].sha256" % index,
        ).lower()
        support_issues = []
        if len(support_sha) != 64 or any(char not in "0123456789abcdef" for char in support_sha):
            support_issues.append("sha256 must be a 64-character lowercase SHA-256")
        support_actual_sha = ""
        support_size_bytes = 0
        if support_path in support_paths:
            support_issues.append("path is duplicated")
        elif not support_path.is_file():
            support_issues.append("file is missing")
        else:
            support_size_bytes = int(support_path.stat().st_size)
            support_actual_sha = _file_sha256(support_path)
            if support_sha and support_actual_sha != support_sha:
                support_issues.append("SHA-256 does not match the manifest")
        row = {
            **dict(raw_support_file),
            "path": _project_path(support_path, root),
            "sha256": support_sha,
            "actual_sha256": support_actual_sha,
            "size_bytes": support_size_bytes,
            "issues": support_issues,
        }
        support_files.append(row)
        support_paths[support_path] = row
        issues.extend(
            "support_files[%d]: %s" % (index, issue)
            for issue in support_issues
        )

    raw_bindings = payload.get("compatible_operations")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise TrainedArtifactError("compatible_operations must be a non-empty list")
    bindings = []
    for index, raw_binding in enumerate(raw_bindings):
        if not isinstance(raw_binding, Mapping):
            raise TrainedArtifactError("compatible_operations[%d] must be a mapping" % index)
        operation_id = _required_text(
            raw_binding,
            "operation",
            "compatible_operations[%d].operation" % index,
        )
        raw_params = raw_binding.get("params") or {}
        if not isinstance(raw_params, Mapping):
            raise TrainedArtifactError("compatible_operations[%d].params must be a mapping" % index)
        params = dict(raw_params)
        if "artifact_manifest_path" in params:
            # The managed manifest is the runtime authority. Never retain a
            # caller-supplied relative path after inspection/import.
            params["artifact_manifest_path"] = _project_path(path, root)
        binding_issues: List[str] = []
        # Bindings store paths relative to the manifest for portability. API clients
        # receive a project-relative path that can be applied directly to recipe params.
        for param_name in ("checkpoint_path", "artifact_path", "model_path"):
            raw_value = str(params.get(param_name) or "").strip()
            if not raw_value:
                continue
            bound_path = Path(raw_value).expanduser()
            if not bound_path.is_absolute():
                bound_path = (path.parent / bound_path).resolve()
            else:
                bound_path = bound_path.resolve()
            if bound_path != artifact_path:
                binding_issues.append(
                    "%s does not reference the hash-verified artifact.path" % param_name
                )
            params[param_name] = _project_path(bound_path, root)
        for param_name in ("path", "adapter_path", "module_path"):
            raw_value = str(params.get(param_name) or "").strip()
            if not raw_value:
                continue
            bound_path = Path(raw_value).expanduser()
            if not bound_path.is_absolute():
                bound_path = (path.parent / bound_path).resolve()
            else:
                bound_path = bound_path.resolve()
            if support_paths and bound_path not in support_paths:
                binding_issues.append(
                    "%s is not declared in support_files" % param_name
                )
            elif not bound_path.is_file():
                binding_issues.append("%s does not reference a readable file" % param_name)
            params[param_name] = _project_path(bound_path, root)
        if "checkpoint_sha256" in params:
            binding_sha = str(params.get("checkpoint_sha256") or "").strip().lower()
            if binding_sha != expected_sha:
                binding_issues.append("checkpoint_sha256 does not match artifact.sha256")
            params["checkpoint_sha256"] = expected_sha
        if "checkpoint_format" in params:
            binding_format = str(params.get("checkpoint_format") or "").strip()
            if binding_format != artifact_format:
                binding_issues.append("checkpoint_format does not match artifact.format")
            params["checkpoint_format"] = artifact_format
        available = True
        if registry is not None:
            try:
                operation_description = registry.get(operation_id).describe()
                schema = dict(operation_description.get("params_schema") or {})
                if not bool(schema.get("additionalProperties", False)):
                    accepted = set((schema.get("properties") or {}).keys())
                    unexpected = sorted(set(params.keys()) - accepted)
                    if unexpected:
                        binding_issues.append(
                            "binding has unsupported parameter(s): %s" % ", ".join(unexpected)
                        )
            except Exception:
                available = False
                binding_issues.append("compatible operation is not available in the active registry")
        if binding_issues:
            issues.extend("%s: %s" % (operation_id, issue) for issue in binding_issues)
        required_inputs = raw_binding.get("required_inputs") or []
        if not isinstance(required_inputs, list) or any(not str(item).strip() for item in required_inputs):
            raise TrainedArtifactError(
                "compatible_operations[%d].required_inputs must be a list of input names" % index
            )
        binding_group = _optional_text(
            raw_binding,
            "binding_group",
            "compatible_operations[%d].binding_group" % index,
        )
        role = _optional_text(
            raw_binding,
            "role",
            "compatible_operations[%d].role" % index,
        )
        preferred_step_id = _optional_text(
            raw_binding,
            "preferred_step_id",
            "compatible_operations[%d].preferred_step_id" % index,
        )
        binding = {
            "operation": operation_id,
            "label": str(raw_binding.get("label") or label),
            "description": str(raw_binding.get("description") or payload.get("description") or ""),
            "required_inputs": [str(item) for item in required_inputs],
            "params": params,
            "available": available,
            "issues": binding_issues,
        }
        if binding_group:
            binding["binding_group"] = binding_group
        if role:
            binding["role"] = role
        if preferred_step_id:
            binding["preferred_step_id"] = preferred_step_id
        bindings.append(binding)

    if application_mode == "all_group_bindings":
        grouped_bindings: Dict[str, List[JsonDict]] = {}
        for binding in bindings:
            group = str(binding.get("binding_group") or "").strip()
            if group:
                grouped_bindings.setdefault(group, []).append(binding)
        if not grouped_bindings:
            issues.append(
                "application.mode all_group_bindings requires at least one binding_group"
            )
        for group, group_bindings in grouped_bindings.items():
            if len(group_bindings) < 2:
                issues.append(
                    "binding_group %s requires at least two compatible operation bindings"
                    % group
                )
            preferred_ids = [
                str(binding.get("preferred_step_id") or "").strip()
                for binding in group_bindings
                if str(binding.get("preferred_step_id") or "").strip()
            ]
            if len(preferred_ids) != len(set(preferred_ids)):
                issues.append(
                    "binding_group %s has duplicate preferred_step_id values" % group
                )

    ready = not issues
    publication_issues = [
        "publication readiness requires a portable schema_version=2 trained artifact"
    ]
    if not ready:
        publication_issues.insert(0, "artifact is not runtime-ready")
    publication_readiness = {
        "status": "blocked",
        "ready": False,
        "issues": publication_issues,
    }
    return {
        "schema_version": 1,
        "kind": TRAINED_ARTIFACT_KIND,
        "id": artifact_id,
        "name": name,
        "label": label,
        "description": str(payload.get("description") or ""),
        "manifest_path": _project_path(path, root),
        "status": "ready" if ready else "invalid",
        "ready": ready,
        "issues": issues,
        "publication_status": "blocked",
        "publication_ready": False,
        "publication_issues": publication_issues,
        "publication_readiness": publication_readiness,
        "artifact": {
            **dict(artifact_config),
            "path": _project_path(artifact_path, root),
            "sha256": expected_sha,
            "actual_sha256": actual_sha,
            "format": artifact_format,
            "size_bytes": size_bytes,
        },
        "application": application,
        "support_files": support_files,
        "compatible_operations": bindings,
        "source": dict(payload.get("source") or {}),
        "source_class": trained_artifact_source_class(
            payload.get("source") or {}, path
        ),
        "training": dict(payload.get("training") or {}),
        "evaluation": dict(payload.get("evaluation") or {}),
    }


def _inspect_trained_artifact_v2(
    path: Path,
    payload: Mapping[str, Any],
    *,
    project_root: Path,
    registry: Optional[OperationRegistry],
) -> JsonDict:
    """Inspect the architecture-neutral executable artifact ABI."""

    # The manifest is a cross-language wire contract.  Reject YAML-only scalar
    # types and recursive aliases before any field is copied into JSON result,
    # plan, or registry evidence.  The exact manifest bytes are hashed below,
    # while this check guarantees that consumers can represent its semantics.
    try:
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise TrainedArtifactError(
            "trained artifact identity fields must contain finite, acyclic, "
            "JSON-compatible values"
        ) from exc

    root = Path(project_root).resolve()
    package_root = Path(path).resolve().parent
    if str(payload.get("kind") or "") != TRAINED_ARTIFACT_KIND:
        raise TrainedArtifactError(
            "unsupported trained artifact kind: %s" % str(payload.get("kind") or "")
        )
    artifact_id = _required_text(payload, "id")
    name = _required_text(payload, "name")
    label = str(payload.get("label") or name).strip()
    issues: List[str] = []
    raw_source = payload.get("source") or {}
    if not isinstance(raw_source, Mapping):
        issues.append("source must be a mapping")
        raw_source = {}
    source = dict(raw_source)
    source_class = trained_artifact_source_class(source, path)

    raw_contract = _required_mapping(payload, "contract")
    contract_id = _required_text(raw_contract, "id", "contract.id")
    contract_version = _positive_integer(
        raw_contract.get("version"), "contract.version", issues
    )
    raw_contract_path = _required_text(raw_contract, "path", "contract.path")
    contract_path = _confined_package_path(
        package_root,
        raw_contract_path,
        "contract.path",
        issues,
    )
    if Path(raw_contract_path).name in TRAINED_ARTIFACT_FILENAMES:
        issues.append("contract.path must not overwrite the trained artifact manifest")
    contract_sha = _required_sha256(raw_contract, "sha256", "contract.sha256", issues)
    contract_file_sha = _required_sha256(
        raw_contract,
        "file_sha256",
        "contract.file_sha256",
        issues,
    )
    contract_actual_sha = ""
    contract_actual_semantic_sha = ""
    contract_size = 0
    contract_payload: JsonDict = {}
    if contract_path is not None:
        if not contract_path.is_file():
            issues.append("contract file is missing")
        else:
            contract_size = int(contract_path.stat().st_size)
            if contract_size <= 0 or contract_size > MAX_ARTIFACT_CONTRACT_BYTES:
                issues.append(
                    "contract file size %d bytes is outside the allowed range (max %d)"
                    % (contract_size, MAX_ARTIFACT_CONTRACT_BYTES)
                )
            contract_actual_sha = _file_sha256(contract_path)
            if contract_file_sha and contract_actual_sha != contract_file_sha:
                issues.append(
                    "contract file SHA-256 does not match contract.file_sha256"
                )
            try:
                contract_payload = _load_mapping(contract_path)
            except Exception as exc:
                issues.append("contract file is invalid: %s" % exc)
            if contract_payload:
                try:
                    contract_actual_semantic_sha = hashlib.sha256(
                        json.dumps(
                            contract_payload,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    ).hexdigest()
                except (TypeError, ValueError, OverflowError, RecursionError) as exc:
                    raise TrainedArtifactError(
                        "trainable slot contract must contain finite, acyclic, "
                        "JSON-compatible values"
                    ) from exc
                if contract_sha and contract_actual_semantic_sha != contract_sha:
                    issues.append(
                        "contract semantic SHA-256 does not match contract.sha256"
                    )
                if str(contract_payload.get("kind") or "") != TRAINABLE_SLOT_CONTRACT_KIND:
                    issues.append(
                        "contract file kind must be %s" % TRAINABLE_SLOT_CONTRACT_KIND
                    )
                if str(contract_payload.get("id") or "") != contract_id:
                    issues.append("contract file id does not match contract.id")
                try:
                    file_version = int(contract_payload.get("version") or 0)
                except (TypeError, ValueError):
                    file_version = 0
                if file_version != contract_version:
                    issues.append("contract file version does not match contract.version")
    contract = {
        **dict(raw_contract),
        "id": contract_id,
        "version": contract_version,
        "path": _project_path(contract_path, root) if contract_path is not None else raw_contract_path,
        "sha256": contract_sha,
        "actual_sha256": contract_actual_semantic_sha,
        "file_sha256": contract_file_sha,
        "actual_file_sha256": contract_actual_sha,
        "size_bytes": contract_size,
    }

    raw_components = payload.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise TrainedArtifactError("schema-v2 components must be a non-empty list")
    component_rows: List[JsonDict] = []
    runtime_components: Dict[str, JsonDict] = {}
    component_ids = set()
    component_paths = set()
    for index, raw_component in enumerate(raw_components):
        if not isinstance(raw_component, Mapping):
            raise TrainedArtifactError("components[%d] must be a mapping" % index)
        component_id = _required_text(raw_component, "id", "components[%d].id" % index)
        raw_component_path = _required_text(
            raw_component,
            "path",
            "components[%d].path" % index,
        )
        component_format = _required_text(
            raw_component,
            "format",
            "components[%d].format" % index,
        ).lower()
        component_sha = _required_sha256(
            raw_component,
            "sha256",
            "components[%d].sha256" % index,
            issues,
        )
        component_issues: List[str] = []
        if component_id in component_ids:
            component_issues.append("id is duplicated")
        component_ids.add(component_id)
        component_path = _confined_package_path(
            package_root,
            raw_component_path,
            "components[%d].path" % index,
            component_issues,
        )
        if component_path is not None and component_path in component_paths:
            component_issues.append("path is duplicated")
        if component_path is not None:
            component_paths.add(component_path)
        if contract_path is not None and component_path == contract_path:
            component_issues.append("path duplicates contract.path")
        if Path(raw_component_path).name in TRAINED_ARTIFACT_FILENAMES:
            component_issues.append(
                "path must not overwrite the trained artifact manifest"
            )
        actual_sha = ""
        size_bytes = 0
        if component_path is not None:
            if not component_path.is_file():
                component_issues.append("file is missing")
            else:
                size_bytes = int(component_path.stat().st_size)
                if size_bytes <= 0 or size_bytes > MAX_ARTIFACT_COMPONENT_BYTES:
                    component_issues.append(
                        "file size %d bytes is outside the allowed range (max %d)"
                        % (size_bytes, MAX_ARTIFACT_COMPONENT_BYTES)
                    )
                actual_sha = _file_sha256(component_path)
                if component_sha and actual_sha != component_sha:
                    component_issues.append("SHA-256 does not match the manifest")
        row = {
            **dict(raw_component),
            "id": component_id,
            "path": _project_path(component_path, root) if component_path is not None else raw_component_path,
            "sha256": component_sha,
            "actual_sha256": actual_sha,
            "format": component_format,
            "size_bytes": size_bytes,
            "issues": component_issues,
        }
        component_rows.append(row)
        runtime_components[component_id] = {
            **dict(raw_component),
            "id": component_id,
            "path": raw_component_path,
            "sha256": component_sha,
            "format": component_format,
        }
        issues.extend(
            "components[%d]: %s" % (index, issue) for issue in component_issues
        )

    raw_support_files = payload.get("support_files") or []
    if not isinstance(raw_support_files, list):
        raise TrainedArtifactError("support_files must be a list")
    support_files: List[JsonDict] = []
    support_paths = set()
    for index, raw_support_file in enumerate(raw_support_files):
        if not isinstance(raw_support_file, Mapping):
            raise TrainedArtifactError("support_files[%d] must be a mapping" % index)
        raw_support_path = _required_text(
            raw_support_file,
            "path",
            "support_files[%d].path" % index,
        )
        support_sha = _required_sha256(
            raw_support_file,
            "sha256",
            "support_files[%d].sha256" % index,
            issues,
        )
        support_role = _optional_text(
            raw_support_file,
            "role",
            "support_files[%d].role" % index,
        )
        support_issues: List[str] = []
        support_path = _confined_package_path(
            package_root,
            raw_support_path,
            "support_files[%d].path" % index,
            support_issues,
        )
        if support_path is not None and support_path in support_paths:
            support_issues.append("path is duplicated")
        if support_path is not None:
            support_paths.add(support_path)
        if support_path is not None and (
            support_path == contract_path or support_path in component_paths
        ):
            support_issues.append("path duplicates a contract or component file")
        if Path(raw_support_path).name in TRAINED_ARTIFACT_FILENAMES:
            support_issues.append(
                "path must not overwrite the trained artifact manifest"
            )
        actual_sha = ""
        size_bytes = 0
        if support_path is not None:
            if not support_path.is_file():
                support_issues.append("file is missing")
            else:
                size_bytes = int(support_path.stat().st_size)
                if size_bytes <= 0 or size_bytes > MAX_ARTIFACT_SUPPORT_FILE_BYTES:
                    support_issues.append(
                        "file size %d bytes is outside the allowed range (max %d)"
                        % (size_bytes, MAX_ARTIFACT_SUPPORT_FILE_BYTES)
                    )
                actual_sha = _file_sha256(support_path)
                if support_sha and actual_sha != support_sha:
                    support_issues.append("SHA-256 does not match the manifest")
        row = {
            **dict(raw_support_file),
            "path": (
                _project_path(support_path, root)
                if support_path is not None
                else raw_support_path
            ),
            "package_path": raw_support_path,
            "sha256": support_sha,
            "actual_sha256": actual_sha,
            "size_bytes": size_bytes,
            "issues": support_issues,
        }
        if support_role:
            row["role"] = support_role
        support_files.append(row)
        issues.extend(
            "support_files[%d]: %s" % (index, issue)
            for issue in support_issues
        )

    raw_runtime = _required_mapping(payload, "runtime")
    backend = str(raw_runtime.get("backend") or "").strip().lower().replace("_", "-")
    if backend == "onnx-runtime":
        backend = "onnxruntime"
    if backend != "onnxruntime":
        issues.append("runtime.backend must be onnxruntime for schema_version=2")
    abi_version = _positive_integer(
        raw_runtime.get("abi_version"), "runtime.abi_version", issues
    )
    from noema_lab.core.trained_artifact_runtime import (
        SUPPORTED_TRAINED_ARTIFACT_ABI_VERSIONS,
    )

    if abi_version not in SUPPORTED_TRAINED_ARTIFACT_ABI_VERSIONS:
        issues.append(
            "runtime.abi_version %d is unsupported; supported versions are %s"
            % (
                abi_version,
                ", ".join(
                    str(value)
                    for value in sorted(
                        SUPPORTED_TRAINED_ARTIFACT_ABI_VERSIONS
                    )
                ),
            )
        )
    raw_entrypoints = raw_runtime.get("entrypoints")
    if not isinstance(raw_entrypoints, list) or not raw_entrypoints:
        raise TrainedArtifactError("runtime.entrypoints must be a non-empty list")
    entrypoints: List[JsonDict] = []
    entrypoint_ids = set()
    for index, raw_entrypoint in enumerate(raw_entrypoints):
        if not isinstance(raw_entrypoint, Mapping):
            raise TrainedArtifactError(
                "runtime.entrypoints[%d] must be a mapping" % index
            )
        entrypoint_id = _required_text(
            raw_entrypoint,
            "id",
            "runtime.entrypoints[%d].id" % index,
        )
        component_id = _required_text(
            raw_entrypoint,
            "component",
            "runtime.entrypoints[%d].component" % index,
        )
        entrypoint_issues: List[str] = []
        if entrypoint_id in entrypoint_ids:
            entrypoint_issues.append("id is duplicated")
        entrypoint_ids.add(entrypoint_id)
        if component_id not in runtime_components:
            entrypoint_issues.append("component does not exist")
        elif backend == "onnxruntime" and str(
            runtime_components[component_id].get("format") or ""
        ).lower() != "onnx":
            entrypoint_issues.append(
                "onnxruntime entrypoints require a component with format=onnx"
            )
        inputs = _tensor_abi_list(
            raw_entrypoint.get("inputs"),
            "runtime.entrypoints[%d].inputs" % index,
            entrypoint_issues,
        )
        outputs = _tensor_abi_list(
            raw_entrypoint.get("outputs"),
            "runtime.entrypoints[%d].outputs" % index,
            entrypoint_issues,
        )
        entrypoint = {
            **dict(raw_entrypoint),
            "id": entrypoint_id,
            "component": component_id,
            "inputs": inputs,
            "outputs": outputs,
            "issues": entrypoint_issues,
        }
        entrypoints.append(entrypoint)
        issues.extend(
            "runtime.entrypoints[%d]: %s" % (index, issue)
            for issue in entrypoint_issues
        )
    runtime_input = {
        **dict(raw_runtime),
        "backend": backend,
        "abi_version": abi_version,
        "entrypoints": entrypoints,
    }

    runtime_result: JsonDict
    if issues:
        runtime_result = {
            **runtime_input,
            "available": False,
            "issues": [],
            "unavailable_reasons": ["artifact schema or integrity validation failed"],
        }
    else:
        from noema_lab.core.trained_artifact_runtime import validate_runtime_entrypoints

        runtime_validation = validate_runtime_entrypoints(
            package_root,
            runtime_components,
            runtime_input,
        )
        runtime_issues = [str(item) for item in runtime_validation.get("issues") or []]
        issues.extend("runtime: %s" % item for item in runtime_issues)
        runtime_result = {
            **runtime_input,
            **runtime_validation,
        }

    raw_application = payload.get("application") or {"mode": "single_binding"}
    if not isinstance(raw_application, Mapping):
        raise TrainedArtifactError("application must be a mapping")
    application = dict(raw_application)
    application_mode = str(application.get("mode") or "single_binding").strip()
    if application_mode not in {"single_binding", "all_group_bindings"}:
        issues.append("application.mode must be single_binding or all_group_bindings")
    application["mode"] = application_mode

    raw_bindings = payload.get("compatible_operations")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise TrainedArtifactError("compatible_operations must be a non-empty list")
    bindings: List[JsonDict] = []
    effective_entrypoints = list(runtime_result.get("entrypoints") or entrypoints)
    entrypoint_by_id = {
        str(item.get("id") or ""): item for item in effective_entrypoints
    }
    referenced_entrypoints = set()
    for index, raw_binding in enumerate(raw_bindings):
        if not isinstance(raw_binding, Mapping):
            raise TrainedArtifactError(
                "compatible_operations[%d] must be a mapping" % index
            )
        operation_id = _required_text(
            raw_binding,
            "operation",
            "compatible_operations[%d].operation" % index,
        )
        runtime_entrypoint = _required_text(
            raw_binding,
            "runtime_entrypoint",
            "compatible_operations[%d].runtime_entrypoint" % index,
        )
        binding_issues: List[str] = []
        if runtime_entrypoint not in entrypoint_by_id:
            binding_issues.append("runtime_entrypoint does not exist")
        else:
            referenced_entrypoints.add(runtime_entrypoint)
        raw_params = raw_binding.get("params") or {}
        if not isinstance(raw_params, Mapping):
            raise TrainedArtifactError(
                "compatible_operations[%d].params must be a mapping" % index
            )
        params = dict(raw_params)
        if "artifact_manifest_path" in params:
            # Bind recipes to the inspected, hash-verified package manifest.
            # This is especially important after an external package has been
            # copied into Noema's project-owned artifact store: a package-local
            # value such as ``trained_artifact.yaml`` must not leak into the
            # recipe as a path relative to the project root.
            params["artifact_manifest_path"] = _project_path(path, root)
        required_inputs = raw_binding.get("required_inputs") or []
        if not isinstance(required_inputs, list) or any(
            not isinstance(item, str) or not item.strip() for item in required_inputs
        ):
            raise TrainedArtifactError(
                "compatible_operations[%d].required_inputs must be a list of input names"
                % index
            )
        required_inputs = [str(item) for item in required_inputs]
        available_operation = True
        if registry is not None:
            try:
                operation = registry.get(operation_id).describe()
                artifact_abi = operation.get("trained_artifact_abi") or {}
                if artifact_abi:
                    entrypoint = entrypoint_by_id.get(runtime_entrypoint) or {}
                    _validate_operation_runtime_abi(
                        artifact_abi,
                        entrypoint,
                        runtime_components,
                        runtime_entrypoint,
                        required_inputs,
                        binding_issues,
                    )
                    params = _canonicalize_operation_binding_params(
                        artifact_abi,
                        params,
                        runtime_entrypoint,
                        _project_path(path, root),
                        binding_issues,
                    )
                    required_inputs = _canonical_operation_required_inputs(
                        artifact_abi,
                    )
                schema = dict(operation.get("params_schema") or {})
                if not bool(schema.get("additionalProperties", False)):
                    accepted_params = set((schema.get("properties") or {}).keys())
                    unexpected = sorted(set(params) - accepted_params)
                    if unexpected:
                        binding_issues.append(
                            "binding has unsupported parameter(s): %s"
                            % ", ".join(unexpected)
                        )
                accepted_inputs = set((operation.get("input_kinds") or {}).keys())
                accepted_inputs.update((operation.get("optional_input_kinds") or {}).keys())
                unexpected_inputs = sorted(set(required_inputs) - accepted_inputs)
                if unexpected_inputs:
                    binding_issues.append(
                        "required_inputs has unknown operation input(s): %s"
                        % ", ".join(unexpected_inputs)
                    )
            except Exception:
                available_operation = False
                binding_issues.append(
                    "compatible operation is not available in the active registry"
                )
        binding_group = _optional_text(
            raw_binding,
            "binding_group",
            "compatible_operations[%d].binding_group" % index,
        )
        role = _optional_text(
            raw_binding,
            "role",
            "compatible_operations[%d].role" % index,
        )
        preferred_step_id = _optional_text(
            raw_binding,
            "preferred_step_id",
            "compatible_operations[%d].preferred_step_id" % index,
        )
        runtime_available = bool(runtime_result.get("available")) and not issues
        binding = {
            "operation": operation_id,
            "label": str(raw_binding.get("label") or label),
            "description": str(
                raw_binding.get("description") or payload.get("description") or ""
            ),
            "runtime_entrypoint": runtime_entrypoint,
            "required_inputs": required_inputs,
            "params": params,
            "tensor_abi": dict(entrypoint_by_id.get(runtime_entrypoint) or {}),
            "available": available_operation and runtime_available and not binding_issues,
            "issues": binding_issues,
        }
        if binding_group:
            binding["binding_group"] = binding_group
        if role:
            binding["role"] = role
        if preferred_step_id:
            binding["preferred_step_id"] = preferred_step_id
        bindings.append(binding)
        issues.extend(
            "%s: %s" % (operation_id, issue) for issue in binding_issues
        )

    unbound_entrypoints = sorted(set(entrypoint_by_id) - referenced_entrypoints)
    if unbound_entrypoints:
        issues.append(
            "runtime entrypoint(s) are not bound to an operation: %s"
            % ", ".join(unbound_entrypoints)
        )
    _validate_atomic_bindings(application_mode, bindings, issues)
    _validate_csi_feedback_pair_invariants(
        bindings,
        entrypoint_by_id,
        source_class,
        issues,
    )

    manifest_sha = _file_sha256(path)
    valid = not issues
    runtime_available = bool(runtime_result.get("available"))
    ready = valid and runtime_available
    status = "ready" if ready else "unavailable" if valid else "invalid"
    publication_readiness = _inspect_publication_readiness_v2(
        payload,
        package_root=package_root,
        project_root=root,
        components=component_rows,
        support_files=support_files,
        source_class=source_class,
    )
    if not ready:
        publication_readiness["issues"] = list(
            dict.fromkeys(
                ["artifact is not runtime-ready"]
                + list(publication_readiness.get("issues") or [])
            )
        )
        publication_readiness["ready"] = False
        publication_readiness["status"] = "blocked"
    for binding in bindings:
        binding["available"] = bool(binding.get("available")) and ready
    inspected = {
        "schema_version": 2,
        "kind": TRAINED_ARTIFACT_KIND,
        "id": artifact_id,
        "name": name,
        "label": label,
        "description": str(payload.get("description") or ""),
        "manifest_path": _project_path(path, root),
        "status": status,
        "valid": valid,
        "ready": ready,
        "issues": issues,
        "publication_status": str(publication_readiness["status"]),
        "publication_ready": bool(publication_readiness["ready"]),
        "publication_issues": list(publication_readiness["issues"]),
        "publication_readiness": publication_readiness,
        # Preserve the v1 summary shape for existing discovery/UI clients.  The
        # actual executable files and hashes are in components.
        "artifact": {
            "path": _project_path(path, root),
            "sha256": manifest_sha,
            "actual_sha256": manifest_sha,
            "manifest_file_sha256": manifest_sha,
            "format": "noema_trained_artifact_manifest_v2",
            "size_bytes": int(path.stat().st_size),
        },
        "contract": contract,
        "components": component_rows,
        "runtime": runtime_result,
        "application": application,
        "support_files": support_files,
        "compatible_operations": bindings,
        "source": source,
        "source_class": source_class,
        # Provenance is intentionally descriptive. It never participates in
        # compatibility, graph validation, or runtime availability.
        "training": dict(payload.get("training") or {}),
        "evaluation": dict(payload.get("evaluation") or {}),
    }
    # Identity-bearing fields are part of the wire contract and must remain
    # portable across YAML/JSON implementations. Bind that identity into every
    # executable operation parameter set. Runtime callers must present it again,
    # preventing an ABI-compatible package from being swapped at the same path
    # after a recipe or execution plan has been created.
    runtime_component_set_sha256 = _runtime_component_set_sha256(component_rows)
    runtime_identity_sha256 = _runtime_identity_sha256(inspected)
    inspected["runtime_component_set_sha256"] = runtime_component_set_sha256
    inspected["runtime_identity_sha256"] = runtime_identity_sha256
    # Preserve the established runtime compatibility contract. Despite its
    # historical name, package_sha256 is the full runtime identity rather than
    # a file/archive digest.
    inspected["package_sha256"] = runtime_identity_sha256
    for binding in inspected["compatible_operations"]:
        params = dict(binding.get("params") or {})
        params["artifact_package_sha256"] = runtime_identity_sha256
        binding["params"] = params
    return inspected


def _validate_operation_runtime_abi(
    expected: Mapping[str, Any],
    entrypoint: Mapping[str, Any],
    components: Mapping[str, Mapping[str, Any]],
    runtime_entrypoint: str,
    required_inputs: Sequence[str],
    issues: List[str],
) -> None:
    """Prove at import time that a runtime entrypoint matches its operation adapter."""

    expected_entrypoint = str(expected.get("entrypoint_id") or "").strip()
    if not expected_entrypoint:
        issues.append("operation trained-artifact ABI is missing entrypoint_id")
    elif runtime_entrypoint != expected_entrypoint:
        issues.append(
            "runtime entrypoint %s does not match operation ABI entrypoint %s"
            % (runtime_entrypoint or "<empty>", expected_entrypoint)
        )
    actual_entrypoint = str(entrypoint.get("id") or "").strip()
    if actual_entrypoint and actual_entrypoint != runtime_entrypoint:
        issues.append(
            "bound runtime entrypoint %s resolves to entrypoint %s"
            % (runtime_entrypoint, actual_entrypoint)
        )

    expected_component = str(expected.get("component_id") or "").strip()
    actual_component = str(entrypoint.get("component") or "").strip()
    if not expected_component:
        issues.append("operation trained-artifact ABI is missing component_id")
    elif actual_component != expected_component:
        issues.append(
            "runtime component %s does not match operation ABI component %s"
            % (actual_component or "<empty>", expected_component)
        )
    expected_role = str(expected.get("component_role") or "").strip()
    component = components.get(actual_component) or {}
    actual_role = str(component.get("role") or "").strip()
    if not expected_role:
        issues.append("operation trained-artifact ABI is missing component_role")
    elif actual_role != expected_role:
        issues.append(
            "runtime component role %s does not match operation ABI role %s"
            % (actual_role or "<empty>", expected_role)
        )

    raw_required = expected.get("required_operation_inputs") or []
    if not isinstance(raw_required, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in raw_required
    ):
        issues.append(
            "operation trained-artifact ABI required_operation_inputs must be a list of input names"
        )
    canonical_required = _canonical_operation_required_inputs(expected)
    declared_required = [str(item).strip() for item in required_inputs]
    if len(declared_required) != len(set(declared_required)):
        issues.append("required_inputs contains duplicate operation input names")
    missing_required = sorted(set(canonical_required) - set(declared_required))
    extra_required = sorted(set(declared_required) - set(canonical_required))
    if missing_required or extra_required:
        details = []
        if missing_required:
            details.append("missing %s" % ", ".join(missing_required))
        if extra_required:
            details.append("unexpected %s" % ", ".join(extra_required))
        issues.append(
            "required_inputs do not match the operation trained-artifact ABI (%s)"
            % "; ".join(details)
        )

    for direction in ("inputs", "outputs"):
        expected_tensors = expected.get(direction) or {}
        if not isinstance(expected_tensors, Mapping):
            issues.append("operation trained-artifact ABI %s must be a mapping" % direction)
            continue
        actual_list = entrypoint.get(direction) or []
        actual = {
            str(item.get("name") or ""): item
            for item in actual_list
            if isinstance(item, Mapping)
        }
        expected_names = {str(name) for name in expected_tensors}
        actual_names = set(actual)
        if expected_names != actual_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            details = []
            if missing:
                details.append("missing %s" % ", ".join(missing))
            if extra:
                details.append("unexpected %s" % ", ".join(extra))
            issues.append(
                "runtime %s do not match the operation ABI (%s)"
                % (direction, "; ".join(details))
            )
            continue
        for name, raw_spec in expected_tensors.items():
            expected_spec = raw_spec if isinstance(raw_spec, Mapping) else {}
            actual_spec = actual[str(name)]
            expected_dtype = str(expected_spec.get("dtype") or "")
            actual_dtype = str(actual_spec.get("dtype") or "")
            if expected_dtype and expected_dtype != actual_dtype:
                issues.append(
                    "runtime %s %s dtype %s does not match operation ABI dtype %s"
                    % (direction[:-1], name, actual_dtype or "<empty>", expected_dtype)
                )
            expected_shape = list(expected_spec.get("shape") or [])
            actual_shape = list(actual_spec.get("shape") or [])
            if expected_shape and len(expected_shape) != len(actual_shape):
                issues.append(
                    "runtime %s %s rank %d does not match operation ABI rank %d"
                    % (direction[:-1], name, len(actual_shape), len(expected_shape))
                )
                continue
            for axis, (expected_dim, actual_dim) in enumerate(
                zip(expected_shape, actual_shape)
            ):
                if isinstance(expected_dim, int) and expected_dim != actual_dim:
                    issues.append(
                        "runtime %s %s axis %d size %s does not match operation ABI size %s"
                        % (direction[:-1], name, axis, actual_dim, expected_dim)
                    )


def _canonical_operation_required_inputs(
    artifact_abi: Mapping[str, Any],
) -> List[str]:
    result: List[str] = []
    raw_additional = artifact_abi.get("required_operation_inputs") or []
    if isinstance(raw_additional, (list, tuple)):
        for value in raw_additional:
            name = str(value).strip()
            if name and name not in result:
                result.append(name)
    return result


def _canonicalize_operation_binding_params(
    artifact_abi: Mapping[str, Any],
    supplied: Mapping[str, Any],
    runtime_entrypoint: str,
    manifest_locator: str,
    issues: List[str],
) -> JsonDict:
    expected_params = artifact_abi.get("binding_params")
    if not isinstance(expected_params, Mapping):
        issues.append("operation trained-artifact ABI binding_params must be a mapping")
        return dict(supplied)
    for required_name in ("artifact_manifest_path", "artifact_entrypoint"):
        if required_name not in expected_params:
            issues.append(
                "operation trained-artifact ABI binding_params is missing %s"
                % required_name
            )
    declared_entrypoint = str(expected_params.get("artifact_entrypoint") or "").strip()
    if declared_entrypoint and declared_entrypoint != runtime_entrypoint:
        issues.append(
            "operation trained-artifact ABI binding_params artifact_entrypoint "
            "does not match entrypoint_id"
        )
    params = dict(supplied)
    for raw_name, expected_value in expected_params.items():
        name = str(raw_name)
        if name == "artifact_manifest_path":
            # A manifest path is deployment state. Never trust or preserve the
            # package-supplied locator; bind to the inspected managed manifest.
            params[name] = manifest_locator
            continue
        canonical_value = runtime_entrypoint if name == "artifact_entrypoint" else expected_value
        if name in params and params.get(name) != canonical_value:
            issues.append(
                "binding parameter %s conflicts with the operation-owned trained-artifact value"
                % name
            )
        params[name] = canonical_value
    return params


def _tensor_abi_list(value: Any, label: str, issues: List[str]) -> List[JsonDict]:
    if not isinstance(value, list) or not value:
        issues.append("%s must be a non-empty list" % label)
        return []
    tensors: List[JsonDict] = []
    names = set()
    for index, raw_tensor in enumerate(value):
        if not isinstance(raw_tensor, Mapping):
            issues.append("%s[%d] must be a mapping" % (label, index))
            continue
        tensor = dict(raw_tensor)
        name = str(tensor.get("name") or "").strip()
        dtype = str(tensor.get("dtype") or "").strip().lower()
        semantic = str(tensor.get("semantic") or "").strip()
        shape = tensor.get("shape")
        if not name:
            issues.append("%s[%d].name must not be empty" % (label, index))
        elif name in names:
            issues.append("%s[%d].name is duplicated" % (label, index))
        names.add(name)
        from noema_lab.core.trained_artifact_runtime import SUPPORTED_TENSOR_DTYPES

        if dtype not in SUPPORTED_TENSOR_DTYPES:
            issues.append(
                "%s[%d].dtype must be one of %s"
                % (label, index, ", ".join(sorted(SUPPORTED_TENSOR_DTYPES)))
            )
        if not semantic:
            issues.append("%s[%d].semantic must not be empty" % (label, index))
        if not isinstance(shape, list) or not shape:
            issues.append("%s[%d].shape must be a non-empty list" % (label, index))
            normalized_shape: List[Any] = []
        elif len(shape) > 16:
            issues.append("%s[%d].shape rank must not exceed 16" % (label, index))
            normalized_shape = list(shape)
        else:
            normalized_shape = []
            for axis, dimension in enumerate(shape):
                if isinstance(dimension, bool):
                    issues.append(
                        "%s[%d].shape[%d] must be a positive integer or symbolic axis"
                        % (label, index, axis)
                    )
                elif isinstance(dimension, int):
                    if dimension < 1:
                        issues.append(
                            "%s[%d].shape[%d] must be positive" % (label, index, axis)
                        )
                    normalized_shape.append(int(dimension))
                elif isinstance(dimension, str) and re.fullmatch(
                    r"[A-Za-z][A-Za-z0-9_]*", dimension
                ):
                    normalized_shape.append(dimension)
                else:
                    issues.append(
                        "%s[%d].shape[%d] must be a positive integer or symbolic axis"
                        % (label, index, axis)
                    )
                    normalized_shape.append(str(dimension))
        complex_representation = str(tensor.get("complex_representation") or "").strip()
        if complex_representation and complex_representation not in {
            "real_imag_last_axis",
            "real_imag_channels",
            "separate_real_imag_tensors",
        }:
            issues.append(
                "%s[%d].complex_representation is unsupported" % (label, index)
            )
        tensor.update(
            {
                "name": name,
                "dtype": dtype,
                "semantic": semantic,
                "shape": normalized_shape,
                "dynamic_axes": [
                    axis
                    for axis, dimension in enumerate(normalized_shape)
                    if isinstance(dimension, str)
                ],
            }
        )
        tensors.append(tensor)
    return tensors


def _validate_atomic_bindings(
    application_mode: str,
    bindings: Sequence[JsonDict],
    issues: List[str],
) -> None:
    if application_mode != "all_group_bindings":
        return
    grouped: Dict[str, List[JsonDict]] = {}
    for binding in bindings:
        group = str(binding.get("binding_group") or "").strip()
        if not group:
            issues.append(
                "application.mode all_group_bindings requires binding_group on every binding"
            )
            continue
        grouped.setdefault(group, []).append(binding)
    if not grouped:
        issues.append(
            "application.mode all_group_bindings requires at least one binding_group"
        )
    for group, group_bindings in grouped.items():
        if len(group_bindings) < 2:
            issues.append("binding_group %s requires at least two bindings" % group)
        for field in ("preferred_step_id", "role", "runtime_entrypoint"):
            values = [
                str(binding.get(field) or "").strip()
                for binding in group_bindings
                if str(binding.get(field) or "").strip()
            ]
            if field in {"role", "runtime_entrypoint"} and len(values) != len(group_bindings):
                issues.append("binding_group %s requires %s on every binding" % (group, field))
            if len(values) != len(set(values)):
                issues.append("binding_group %s has duplicate %s values" % (group, field))


def _validate_csi_feedback_pair_invariants(
    bindings: Sequence[JsonDict],
    entrypoints: Mapping[str, Mapping[str, Any]],
    source_class: str,
    issues: List[str],
) -> None:
    """Validate the paired CSI codec as one shape-compatible deployment unit."""

    encoder_binding = next(
        (
            item
            for item in bindings
            if str(item.get("operation") or "") == "model.csi_feedback_encoder"
        ),
        None,
    )
    decoder_binding = next(
        (
            item
            for item in bindings
            if str(item.get("operation") or "") == "model.csi_feedback_decoder"
        ),
        None,
    )
    if encoder_binding is None and decoder_binding is None:
        return
    if encoder_binding is None or decoder_binding is None:
        issues.append(
            "CSI feedback artifacts must bind both encoder and decoder entrypoints"
        )
        return

    encoder = entrypoints.get(str(encoder_binding.get("runtime_entrypoint") or "")) or {}
    decoder = entrypoints.get(str(decoder_binding.get("runtime_entrypoint") or "")) or {}
    encoder_csi = _effective_entrypoint_tensor(encoder, "inputs", "csi_ri")
    encoder_feedback = _effective_entrypoint_tensor(
        encoder, "outputs", "feedback_code"
    )
    decoder_feedback = _effective_entrypoint_tensor(
        decoder, "inputs", "feedback_code"
    )
    decoder_csi = _effective_entrypoint_tensor(decoder, "outputs", "csi_hat_ri")
    if not all((encoder_csi, encoder_feedback, decoder_feedback, decoder_csi)):
        issues.append(
            "CSI feedback pair is missing a canonical csi_ri, feedback_code, or csi_hat_ri tensor"
        )
        return

    _validate_tensor_shape_relation(
        encoder_feedback,
        decoder_feedback,
        "CSI encoder feedback output and decoder feedback input",
        issues,
    )
    _validate_tensor_shape_relation(
        encoder_csi,
        decoder_csi,
        "CSI encoder input and decoder reconstruction output",
        issues,
    )

    feedback_shape = list(encoder_feedback.get("shape") or [])
    fixed_feedback_dimension = (
        feedback_shape[-1]
        if feedback_shape and isinstance(feedback_shape[-1], int)
        else None
    )
    for binding in (encoder_binding, decoder_binding):
        configured = (binding.get("params") or {}).get("feedback_dimension")
        if fixed_feedback_dimension is not None and configured is not None:
            try:
                configured_dimension = int(configured)
            except (TypeError, ValueError):
                issues.append("CSI feedback binding feedback_dimension must be an integer")
                continue
            if configured_dimension != fixed_feedback_dimension:
                issues.append(
                    "CSI feedback binding dimension %d does not match runtime dimension %d"
                    % (configured_dimension, fixed_feedback_dimension)
                )

    if source_class in {"published_reference", "reference_baseline"}:
        csi_shape = list(encoder_csi.get("shape") or [])
        if len(csi_shape) != 4 or any(
            not isinstance(csi_shape[axis], int) for axis in (1, 2, 3)
        ):
            issues.append(
                "CSI reference artifacts must expose fixed RI, transmit-antenna, and subcarrier dimensions"
            )
        if fixed_feedback_dimension is None:
            issues.append(
                "CSI reference artifacts must expose a fixed feedback dimension"
            )


def _effective_entrypoint_tensor(
    entrypoint: Mapping[str, Any],
    direction: str,
    name: str,
) -> JsonDict:
    validated = entrypoint.get("validated_signature") or {}
    candidates = validated.get(direction) or entrypoint.get(direction) or []
    return next(
        (
            dict(item)
            for item in candidates
            if isinstance(item, Mapping) and str(item.get("name") or "") == name
        ),
        {},
    )


def _validate_tensor_shape_relation(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    label: str,
    issues: List[str],
) -> None:
    left_shape = list(left.get("shape") or [])
    right_shape = list(right.get("shape") or [])
    if len(left_shape) != len(right_shape):
        issues.append(
            "%s ranks differ (%d versus %d)"
            % (label, len(left_shape), len(right_shape))
        )
        return
    for axis, (left_dimension, right_dimension) in enumerate(
        zip(left_shape, right_shape)
    ):
        if isinstance(left_dimension, int) and isinstance(right_dimension, int):
            mismatch = left_dimension != right_dimension
        elif isinstance(left_dimension, str) and isinstance(right_dimension, str):
            mismatch = left_dimension != right_dimension
        else:
            mismatch = False
        if mismatch:
            issues.append(
                "%s axis %d differs (%s versus %s)"
                % (label, axis, left_dimension, right_dimension)
            )


def _positive_integer(value: Any, label: str, issues: List[str]) -> int:
    if isinstance(value, bool):
        issues.append("%s must be a positive integer" % label)
        return 0
    try:
        result = int(value)
    except (TypeError, ValueError):
        issues.append("%s must be a positive integer" % label)
        return 0
    if result < 1 or result != value:
        issues.append("%s must be a positive integer" % label)
    return result


def _required_sha256(
    payload: Mapping[str, Any],
    key: str,
    label: str,
    issues: List[str],
) -> str:
    value = str(payload.get(key) or "").strip().lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        issues.append("%s must be a 64-character lowercase SHA-256" % label)
    return value


def _confined_package_path(
    package_root: Path,
    value: Any,
    label: str,
    issues: List[str],
) -> Optional[Path]:
    raw = str(value or "").strip()
    if not raw:
        issues.append("%s must not be empty" % label)
        return None
    relative = Path(raw)
    if relative.is_absolute():
        issues.append("%s must be relative to the artifact package" % label)
        return None
    resolved = (Path(package_root).resolve() / relative).resolve()
    try:
        resolved.relative_to(Path(package_root).resolve())
    except ValueError:
        issues.append("%s escapes the artifact package" % label)
        return None
    return resolved


def _manifest_paths(project_root: Path) -> Iterable[Path]:
    seen = set()
    roots = (
        project_root / "differentiable_exports",
        project_root / "trained_artifacts",
        project_root / ".noema" / "trained_artifacts",
        project_root / ".noema" / "training_exports",
    )
    for root in roots:
        if not root.is_dir():
            continue
        for filename in TRAINED_ARTIFACT_FILENAMES:
            for path in root.rglob(filename):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved


def _load_mapping(path: Path) -> JsonDict:
    try:
        value = load_strict_yaml_or_json(path)
    except StructuredInputError as exc:
        detail = str(exc).lower()
        if any(
            marker in detail
            for marker in (
                "recursive yaml",
                "unsupported yaml value type",
                "nan or infinity",
                "non-finite numeric",
            )
        ):
            raise TrainedArtifactError(
                "trained artifact identity fields must contain finite, acyclic, "
                "JSON-compatible values"
            ) from exc
        raise TrainedArtifactError(
            "invalid trained-artifact YAML/JSON input %s: %s" % (path, exc)
        ) from exc
    if not isinstance(value, Mapping):
        raise TrainedArtifactError("trained artifact manifest must be a mapping: %s" % path)
    return dict(value)


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise TrainedArtifactError("%s must be a mapping" % key)
    return value


def _required_text(payload: Mapping[str, Any], key: str, label: str = "") -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise TrainedArtifactError("%s must not be empty" % (label or key))
    return value


def _optional_text(payload: Mapping[str, Any], key: str, label: str = "") -> str:
    if key not in payload or payload.get(key) is None:
        return ""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TrainedArtifactError("%s must be a non-empty string" % (label or key))
    return value.strip()


def _safe_artifact_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return (slug or "learned-checkpoint")[:80].rstrip("-")


def _clean_display_text(value: str) -> str:
    # Labels are rendered in the UI and serialized into YAML. Keep them concise and
    # single-line while preserving ordinary Unicode names.
    return " ".join(str(value or "").split())[:160]


def _preflight_safe_npz(path: Path) -> None:
    """Bound ZIP expansion before NumPy is allowed to materialize checkpoint arrays."""

    try:
        if not zipfile.is_zipfile(path):
            raise TrainedArtifactError("learned checkpoint is not a valid NPZ archive")
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.infolist()
            if not members or len(members) > MAX_IMPORTED_CHECKPOINT_MEMBERS:
                raise TrainedArtifactError(
                    "checkpoint archive must contain between 1 and %d members"
                    % MAX_IMPORTED_CHECKPOINT_MEMBERS
                )
            names = [item.filename for item in members]
            if len(names) != len(set(names)):
                raise TrainedArtifactError("checkpoint archive contains duplicate members")
            if any(item.flag_bits & 0x1 for item in members):
                raise TrainedArtifactError("encrypted checkpoint archive members are not supported")
            expanded_size = sum(max(0, int(item.file_size)) for item in members)
            if expanded_size > MAX_IMPORTED_CHECKPOINT_UNCOMPRESSED_BYTES:
                raise TrainedArtifactError(
                    "checkpoint archive expands to %d bytes; maximum allowed is %d"
                    % (expanded_size, MAX_IMPORTED_CHECKPOINT_UNCOMPRESSED_BYTES)
                )
    except TrainedArtifactError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise TrainedArtifactError("learned checkpoint is not a valid NPZ archive") from exc


def _project_path(path: Path, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(resolved)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
