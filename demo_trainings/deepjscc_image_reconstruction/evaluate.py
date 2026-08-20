from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import numpy as np

from datamodule import build_validation_loader
try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sessions(manifest_path: Path):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("evaluation requires onnxruntime; install requirements.txt") from exc
    manifest = load_strict_yaml_or_json(manifest_path)
    if not isinstance(manifest, dict) or int(manifest.get("schema_version") or 0) != 2:
        raise ValueError("trained artifact must use schema_version: 2")
    components = {str(row.get("id") or ""): dict(row) for row in manifest.get("components") or []}
    sessions = {}
    for component_id in ("encoder", "decoder"):
        component = components.get(component_id)
        if not component or str(component.get("format") or "").lower() != "onnx":
            raise ValueError("trained artifact is missing its %s ONNX component" % component_id)
        path = (manifest_path.parent / str(component.get("path") or "")).resolve()
        expected_sha = str(component.get("sha256") or "")
        if not path.is_file() or _sha256(path) != expected_sha:
            raise ValueError("%s ONNX component failed SHA-256 verification" % component_id)
        sessions[component_id] = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
    return manifest, sessions


def _forward(sessions, images: np.ndarray, snr_db: float, config: dict) -> tuple[np.ndarray, int]:
    symbols_ri = np.asarray(
        sessions["encoder"].run(["symbols_ri"], {"images": images})[0],
        dtype=np.float32,
    )
    channels = symbols_ri.shape[1] // 2
    symbols = symbols_ri[:, :channels] + 1j * symbols_ri[:, channels:]
    power = dict(config.get("symbol_power") or {})
    if bool(power.get("enabled", False)):
        target = float(power.get("target_power", 1.0))
        eps = float(power.get("eps", 1e-8))
        symbols *= math.sqrt(target / max(float(np.mean(np.abs(symbols) ** 2)), eps))
    noise_variance = float(10.0 ** (-float(snr_db) / 10.0))
    channel_type = str((config.get("channel") or {}).get("type") or "awgn")
    noise = (
        np.random.standard_normal(symbols.shape)
        + 1j * np.random.standard_normal(symbols.shape)
    ).astype(np.complex64)
    if channel_type == "flat_rayleigh":
        gain_shape = (int(symbols.shape[0]),) + (1,) * (
            symbols.ndim - 1
        )
        channel_gain = (
            np.random.standard_normal(gain_shape)
            + 1j * np.random.standard_normal(gain_shape)
        ).astype(np.complex64) / np.float32(math.sqrt(2.0))
        received = (
            channel_gain * symbols
            + noise * np.float32(math.sqrt(noise_variance / 2.0))
        )
    elif channel_type == "awgn":
        received = symbols + noise * np.float32(
            math.sqrt(noise_variance / 2.0)
        )
    else:
        raise ValueError("Unsupported evaluation channel: %s" % channel_type)
    received_ri = np.ascontiguousarray(
        np.concatenate([received.real, received.imag], axis=1),
        dtype=np.float32,
    )
    reconstruction = np.asarray(
        sessions["decoder"].run(
            ["reconstruction"],
            {"symbols_ri": received_ri},
        )[0],
        dtype=np.float32,
    )
    return reconstruction, int(symbols.size)


def main() -> int:
    config = load_strict_yaml_or_json(Path("train_config.yaml"))
    training = dict(config.get("training") or {})
    manifest_path = Path(str(training.get("artifact_manifest_path") or "trained_artifact.yaml"))
    model_config = dict(config.get("model") or {})
    maximum_symbol_channels = int(model_config.get("symbol_channels") or 32)
    symbol_channel_options = sorted(
        {
            int(value)
            for value in model_config.get("symbol_channel_options")
            or [maximum_symbol_channels]
            if 0 < int(value) <= maximum_symbol_channels
        }
        | {maximum_symbol_channels}
    )
    manifest_paths = []
    for active_channels in symbol_channel_options:
        rate_id = "kappa_%s" % (
            ("%0.3f" % (float(active_channels) / 64.0))
            .rstrip("0")
            .rstrip(".")
            .replace(".", "p")
        )
        candidate = (
            manifest_path
            if active_channels == maximum_symbol_channels
            else manifest_path.with_name(
                "%s_%s%s"
                % (manifest_path.stem, rate_id, manifest_path.suffix)
            )
        )
        manifest_paths.append((active_channels, candidate))
    loader = build_validation_loader(config)
    snr_values = [float(value) for value in (config.get("evaluation") or {}).get("snr_db") or (config.get("channel") or {}).get("snr_db") or [12.0]]
    rows = []
    random_state = np.random.get_state()
    manifests = []
    try:
        base_seed = int(
            (config.get("evaluation") or {}).get("noise_seed", 19001)
        )
        for active_channels, active_manifest_path in manifest_paths:
            manifest, sessions = _load_sessions(active_manifest_path)
            manifests.append(
                {
                    "path": str(active_manifest_path),
                    "schema_version": int(manifest.get("schema_version") or 0),
                    "contract": dict(manifest.get("contract") or {}),
                    "symbol_channels": active_channels,
                }
            )
            for snr_index, snr_db in enumerate(snr_values):
                np.random.seed(base_seed + snr_index)
                mse_values = []
                channel_uses = 0
                image_pixels = 0
                for batch in loader:
                    images = np.ascontiguousarray(
                        batch["image"].cpu().numpy(), dtype=np.float32
                    )
                    reconstruction, uses = _forward(
                        sessions, images, snr_db, config
                    )
                    mse_values.append(
                        float(np.mean((reconstruction - images) ** 2))
                    )
                    channel_uses += uses
                    image_pixels += int(
                        images.shape[0]
                        * images.shape[2]
                        * images.shape[3]
                    )
                mse = (
                    float(np.mean(mse_values))
                    if mse_values
                    else float("inf")
                )
                psnr_db = (
                    float(-10.0 * math.log10(max(mse, 1e-12)))
                    if math.isfinite(mse)
                    else float("nan")
                )
                rows.append(
                    {
                        "snr_db": snr_db,
                        "image_mse": mse,
                        "psnr_db": psnr_db,
                        "symbol_channels": active_channels,
                        "channel_uses_per_pixel": (
                            float(channel_uses)
                            / float(max(image_pixels, 1))
                        ),
                    }
                )
    finally:
        np.random.set_state(random_state)
    report = {
        "schema_version": 1,
        "kind": "noema.deepjscc_external_validation",
        "split": "validation",
        "purpose": "post_export_runtime_validation_not_held_out_test",
        "held_out_test_owner": "noema_ordinary_recipe_or_benchmark",
        "trained_artifacts": manifests,
        "operating_points": rows,
        "mean_psnr_db": float(np.mean([row["psnr_db"] for row in rows])) if rows else float("nan"),
    }
    output_path = Path(str((config.get("evaluation") or {}).get("metrics_path") or "evaluation_metrics.json"))
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote validation metrics: %s" % output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
