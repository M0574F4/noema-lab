from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
import yaml

from datamodule import build_train_loader, build_validation_loader, load_data_contract
from losses import image_mse
from model import (
    ARCHITECTURE,
    ReferenceDeepJSCCModel,
    export_onnx_components,
)
from scenario import build_scenario
try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


def _device(config: Mapping[str, Any]) -> torch.device:
    requested = str((config.get("training") or {}).get("device", "cuda_if_available"))
    if requested == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _validate(
    scenario,
    loader,
    snr_values,
    symbol_channel_options,
    device,
    seed: int,
) -> float:
    scenario.eval()
    losses = []
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        with torch.no_grad():
            for batch in loader:
                batch = _to_device(batch, device)
                for snr_db in snr_values:
                    for active_channels in symbol_channel_options:
                        signals = scenario(
                            batch,
                            snr_db=float(snr_db),
                            active_symbol_channels=int(active_channels),
                        )
                        losses.append(
                            float(
                                image_mse(
                                    signals["reconstruction"],
                                    signals["source_image"],
                                ).cpu()
                            )
                        )
    return float(np.mean(losses)) if losses else float("inf")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_id(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", ".", str(value or "deepjscc").lower()).strip(".")
    return text or "deepjscc"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_artifact_manifest(
    config: Mapping[str, Any],
    encoder_path: Path,
    decoder_path: Path,
    component_hashes: Mapping[str, str],
    *,
    best_seed: int,
    validation_mse: float,
    manifest_path_override: Path | None = None,
    symbol_channels_override: int | None = None,
) -> Dict[str, Any]:
    training = dict(config.get("training") or {})
    recipe = dict(config.get("recipe") or {})
    model = dict(config.get("model") or {})
    data = dict(config.get("data") or {})
    sender_step = str(recipe.get("sender_step") or "sender")
    receiver_step = str(recipe.get("receiver_step") or "receiver")
    symbol_channels = int(
        symbol_channels_override
        if symbol_channels_override is not None
        else model.get("symbol_channels") or 32
    )
    manifest_path = (
        Path(manifest_path_override)
        if manifest_path_override is not None
        else Path(
            str(
                training.get("artifact_manifest_path")
                or "trained_artifact.yaml"
            )
        )
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    contract_source, contract = _find_training_contract(config)
    contract_path = manifest_path.parent / "training_contract.yaml"
    if contract_source.resolve() != contract_path.resolve():
        shutil.copyfile(contract_source, contract_path)
    contract_sha = _canonical_sha256(contract)
    contract_file_sha = _sha256(contract_path)
    portable_encoder = _package_relative_path(encoder_path, manifest_path)
    portable_decoder = _package_relative_path(decoder_path, manifest_path)
    maximum_symbol_channels = int(model.get("symbol_channels") or 32)
    rate_suffix = (
        ""
        if symbol_channels == maximum_symbol_channels
        else ".c%d" % symbol_channels
    )
    artifact_id = (
        str(
            training.get("artifact_id")
            or "%s.deepjscc" % _safe_id(recipe.get("name", "deepjscc"))
        )
        + rate_suffix
    )
    artifact_name = str(training.get("artifact_name") or "Learned DeepJSCC model")
    artifact_label = str(training.get("artifact_label") or artifact_name)
    if rate_suffix:
        artifact_name = "%s (%d latent channels)" % (
            artifact_name,
            symbol_channels,
        )
        artifact_label = "%s · κ=%.3g" % (
            artifact_label,
            float(symbol_channels) / 64.0,
        )
    data_contract_path = Path(str(data.get("contract_path") or "")).resolve()
    if not data_contract_path.is_file():
        raise FileNotFoundError("DeepJSCC data contract is required to package the artifact")
    if data_contract_path.parent != manifest_path.parent.resolve():
        raise ValueError(
            "DeepJSCC data contract must remain beside the returned trained artifact"
        )
    payload = {
        "schema_version": 2,
        "kind": "noema.trained_block_artifact",
        "id": artifact_id,
        "name": artifact_name,
        "label": artifact_label,
        "description": "Jointly trained DeepJSCC image encoder and decoder.",
        "contract": {
            "id": str(contract.get("id") or ""),
            "version": int(contract.get("version") or contract.get("schema_version") or 1),
            "path": "training_contract.yaml",
            "sha256": contract_sha,
            "file_sha256": contract_file_sha,
        },
        "components": [
            {
                "id": "encoder",
                "role": "encoder",
                "path": portable_encoder,
                "sha256": str(component_hashes["encoder_sha256"]),
                "format": "onnx",
            },
            {
                "id": "decoder",
                "role": "decoder",
                "path": portable_decoder,
                "sha256": str(component_hashes["decoder_sha256"]),
                "format": "onnx",
            },
        ],
        "runtime": {
            "backend": "onnxruntime",
            "abi_version": 1,
            "entrypoints": [
                {
                    "id": "encoder",
                    "component": "encoder",
                    "inputs": [
                        {
                            "name": "images",
                            "dtype": "float32",
                            "shape": ["batch", 3, "image_height", "image_width"],
                            "layout": "NCHW",
                            "semantic": "normalized_rgb_image_0_1",
                        }
                    ],
                    "outputs": [
                        {
                            "name": "symbols_ri",
                            "dtype": "float32",
                            "shape": ["batch", 2 * symbol_channels, "symbol_height", "symbol_width"],
                            "layout": "NCHW",
                            "semantic": "complex_channel_symbols",
                            "complex_representation": "real_imag_channels",
                        }
                    ],
                },
                {
                    "id": "decoder",
                    "component": "decoder",
                    "inputs": [
                        {
                            "name": "symbols_ri",
                            "dtype": "float32",
                            "shape": ["batch", 2 * symbol_channels, "symbol_height", "symbol_width"],
                            "layout": "NCHW",
                            "semantic": "received_complex_channel_symbols",
                            "complex_representation": "real_imag_channels",
                        }
                    ],
                    "outputs": [
                        {
                            "name": "reconstruction",
                            "dtype": "float32",
                            "shape": ["batch", 3, "image_height", "image_width"],
                            "layout": "NCHW",
                            "semantic": "normalized_rgb_reconstruction_0_1",
                        }
                    ],
                },
            ],
        },
        "application": {"mode": "all_group_bindings"},
        "compatible_operations": [
            {
                "operation": "model.deepjscc_external_encode",
                "binding_group": "deepjscc_sender_receiver",
                "role": "encoder",
                "preferred_step_id": sender_step,
                "label": artifact_label,
                "description": "Portable image-to-complex-symbol encoder.",
                "runtime_entrypoint": "encoder",
                "required_inputs": ["images"],
                "params": {
                    "runtime": "learned_artifact",
                    "artifact_manifest_path": manifest_path.name,
                    "artifact_entrypoint": "encoder",
                },
            },
            {
                "operation": "model.deepjscc_external_decode",
                "binding_group": "deepjscc_sender_receiver",
                "role": "decoder",
                "preferred_step_id": receiver_step,
                "label": artifact_label,
                "description": "Portable complex-symbol-to-image decoder.",
                "runtime_entrypoint": "decoder",
                "required_inputs": ["symbols"],
                "params": {
                    "runtime": "learned_artifact",
                    "artifact_manifest_path": manifest_path.name,
                    "artifact_entrypoint": "decoder",
                },
            },
        ],
        "source": {
            "origin": "noema_standalone_training_project",
            "recipe": str(recipe.get("name") or ""),
            "recipe_sha256": str(recipe.get("sha256") or ""),
            "training_template": str(config.get("training_template") or ""),
            "data_contract": {
                "path": "data_contract.yaml",
                "sha256": str(data.get("contract_sha256") or ""),
                "file_sha256": str(data.get("contract_file_sha256") or ""),
            },
        },
        "training": {
            "framework": "torch",
            "objective": "image.mse",
            "best_seed": int(best_seed),
            "validation_mse": float(validation_mse),
            "architecture": ARCHITECTURE,
            "symbol_channels": symbol_channels,
            "data_partitions": {
                "train_image_ids": list(data.get("train_image_ids") or []),
                "validation_image_ids": list(
                    data.get("validation_image_ids") or []
                ),
                "test_images_used": False,
            },
        },
        "evaluation": {
            "selection_metric": "validation_image_mse",
            "selection_direction": "minimize",
            "validation_image_mse": float(validation_mse),
        },
    }
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary_path.replace(manifest_path)
    return payload


def _find_training_contract(config: Mapping[str, Any]) -> tuple[Path, Dict[str, Any]]:
    training = dict(config.get("training") or {})
    configured = str(training.get("contract_path") or config.get("training_contract_path") or "").strip()
    candidates = [Path(configured)] if configured else []
    candidates.extend((Path("training_contract.yaml"), Path("../training_contract.yaml")))
    for path in candidates:
        if not path.is_file():
            continue
        payload = load_strict_yaml_or_json(path)
        if not isinstance(payload, dict):
            raise ValueError("training_contract.yaml must contain a mapping")
        if not str(payload.get("kind") or "").startswith("noema.trainable_slot_contract"):
            raise ValueError("training_contract.yaml is not a Noema trainable-slot contract")
        if not str(payload.get("id") or "").strip():
            raise ValueError("training_contract.yaml requires an id")
        return path, payload
    raise FileNotFoundError(
        "training_contract.yaml is required to package the returned artifact; "
        "run this demonstration project inside its exported Noema training-contract bundle"
    )


def _package_relative_path(component_path: Path, manifest_path: Path) -> str:
    try:
        return str(component_path.resolve().relative_to(manifest_path.parent.resolve()))
    except ValueError as exc:
        raise ValueError("artifact components must be inside the trained-artifact package") from exc


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    config = load_strict_yaml_or_json(Path("train_config.yaml"))
    if not isinstance(config, dict):
        raise ValueError("train_config.yaml must contain a mapping")
    training = dict(config.get("training") or {})
    model_config = dict(config.get("model") or {})
    load_data_contract(config)
    snr_values = [float(value) for value in (config.get("channel") or {}).get("snr_db") or [12.0]]
    maximum_symbol_channels = int(model_config.get("symbol_channels") or 32)
    configured_symbol_channel_options = sorted(
        {
            int(value)
            for value in model_config.get("symbol_channel_options")
            or [maximum_symbol_channels]
        }
    )
    if not configured_symbol_channel_options or configured_symbol_channel_options[0] < 1:
        raise ValueError(
            "model.symbol_channel_options must contain positive integers"
        )
    symbol_channel_options = sorted(
        {
            value
            for value in configured_symbol_channel_options
            if value <= maximum_symbol_channels
        }
        | {maximum_symbol_channels}
    )
    device = _device(config)
    train_loader = build_train_loader(config)
    validation_loader = build_validation_loader(config)
    epochs = int(training.get("epochs", 100))
    patience = int(training.get("early_stopping_patience", 15))
    initialization_seeds = [int(value) for value in training.get("initialization_seeds") or [23]]
    history = []
    selected = None

    for seed in initialization_seeds:
        _seed_everything(seed)
        model = ReferenceDeepJSCCModel(
            symbol_channels=maximum_symbol_channels
        )
        scenario = build_scenario(config, model=model).to(device)
        optimizer = torch.optim.Adam(
            scenario.model.parameters(),
            lr=float(training.get("learning_rate", 1e-3)),
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
        best_state = None
        best_validation = float("inf")
        stale_epochs = 0
        snr_rng = np.random.default_rng(seed + 10_000)
        rate_rng = np.random.default_rng(seed + 20_000)
        for epoch in range(epochs):
            scenario.train()
            train_losses = []
            for batch in train_loader:
                batch = _to_device(batch, device)
                snr_db = float(snr_rng.choice(snr_values))
                active_channels = int(rate_rng.choice(symbol_channel_options))
                optimizer.zero_grad(set_to_none=True)
                signals = scenario(
                    batch,
                    snr_db=snr_db,
                    active_symbol_channels=active_channels,
                )
                loss = image_mse(signals["reconstruction"], signals["source_image"])
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            validation_mse = _validate(
                scenario,
                validation_loader,
                snr_values,
                symbol_channel_options,
                device,
                seed=int(training.get("validation_noise_seed", 9001)),
            )
            train_mse = float(np.mean(train_losses)) if train_losses else float("inf")
            row = {
                "seed": seed,
                "epoch": epoch + 1,
                "train_image_mse": train_mse,
                "validation_image_mse": validation_mse,
            }
            history.append(row)
            print(
                "seed=%d epoch=%d train_mse=%.8f validation_mse=%.8f"
                % (seed, epoch + 1, train_mse, validation_mse)
            )
            if validation_mse < best_validation:
                best_validation = validation_mse
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in scenario.model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if patience >= 0 and stale_epochs > patience:
                    break
        if best_state is None:
            raise RuntimeError("DeepJSCC training did not produce a finite candidate")
        candidate = ReferenceDeepJSCCModel(
            symbol_channels=maximum_symbol_channels
        )
        candidate.load_state_dict(best_state, strict=True)
        if selected is None or best_validation < selected["validation_mse"]:
            selected = {"model": candidate, "seed": seed, "validation_mse": best_validation}

    if selected is None or not math.isfinite(float(selected["validation_mse"])):
        raise RuntimeError("DeepJSCC training did not produce a finite validation metric")
    manifest_path = Path(str(training.get("artifact_manifest_path") or "trained_artifact.yaml"))
    component_dir = manifest_path.parent / "artifacts"
    crop_size = int((config.get("data") or {}).get("crop_size") or 64)
    exported_rates = []
    for active_channels in symbol_channel_options:
        rate_id = "kappa_%s" % (
            ("%0.3f" % (float(active_channels) / 64.0))
            .rstrip("0")
            .rstrip(".")
            .replace(".", "p")
        )
        rate_component_dir = (
            component_dir
            if active_channels == maximum_symbol_channels
            else component_dir / rate_id
        )
        encoder_path = rate_component_dir / "encoder.onnx"
        decoder_path = rate_component_dir / "decoder.onnx"
        rate_manifest_path = (
            manifest_path
            if active_channels == maximum_symbol_channels
            else manifest_path.with_name(
                "%s_%s%s"
                % (
                    manifest_path.stem,
                    rate_id,
                    manifest_path.suffix,
                )
            )
        )
        component_hashes = export_onnx_components(
            selected["model"],
            encoder_path,
            decoder_path,
            example_image_shape=(1, 3, crop_size, crop_size),
            active_symbol_channels=active_channels,
        )
        _write_artifact_manifest(
            config,
            encoder_path,
            decoder_path,
            component_hashes,
            best_seed=int(selected["seed"]),
            validation_mse=float(selected["validation_mse"]),
            manifest_path_override=rate_manifest_path,
            symbol_channels_override=active_channels,
        )
        exported_rates.append(
            {
                "symbol_channels": active_channels,
                "channel_uses_per_source_pixel": (
                    float(active_channels) / 64.0
                ),
                "manifest": str(rate_manifest_path),
                **dict(component_hashes),
            }
        )
        print(
            "exported κ=%.3g paired artifact: %s"
            % (
                float(active_channels) / 64.0,
                rate_manifest_path,
            )
        )
    history_path = Path(str(training.get("history_path") or "training_history.json"))
    _write_json(
        history_path,
        {
            "schema_version": 1,
            "objective": "image.mse",
            "best_seed": int(selected["seed"]),
            "best_validation_image_mse": float(selected["validation_mse"]),
            "symbol_channel_options": symbol_channel_options,
            "exported_rates": exported_rates,
            "epochs": history,
        },
    )
    print("wrote trained artifact: %s" % training.get("artifact_manifest_path", "trained_artifact.yaml"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
