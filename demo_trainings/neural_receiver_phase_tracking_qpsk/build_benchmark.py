from __future__ import annotations

"""Build the paired six-method phase-tracking receiver campaign."""

import argparse
import copy
import hashlib
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from noema_lab.core.reproducibility import derive_seed
from noema_lab.core.trained_artifacts import inspect_trained_artifact
from noema_lab.ops import build_registry
try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


DEFAULT_SNR_DB = (-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
DEFAULT_HELD_OUT_SEEDS = (91001, 92001, 93001)
DEFAULT_BIT_COUNT = 262_144
DEFAULT_PACKET_BITS = 1024


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the post-training QPSK carrier-tracking benchmark pack."
    )
    parser.add_argument("--artifact", help="Returned trained_artifact.yaml")
    parser.add_argument("--recipe", default="noema_recipe.yaml")
    parser.add_argument("--output", default="benchmark_pack.yaml")
    parser.add_argument("--project-root")
    parser.add_argument(
        "--snr-db",
        default=",".join(str(value) for value in DEFAULT_SNR_DB),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in DEFAULT_HELD_OUT_SEEDS),
    )
    parser.add_argument("--bit-count", type=int, default=DEFAULT_BIT_COUNT)
    parser.add_argument(
        "--packet-bits",
        type=int,
        default=DEFAULT_PACKET_BITS,
        help="Data bits per temporal packet; total bit-count must be divisible by this value.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Bits per BLER block (defaults to packet-bits).",
    )
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path = Path(args.recipe).expanduser().resolve()
    artifact_path = _artifact_path(args.artifact)
    history_path = Path("training_history.json").resolve()
    evaluation_path = Path("evaluation_metrics.json").resolve()
    missing = [
        label
        for label, path in (
            ("trained artifact", artifact_path),
            ("training history", history_path),
            ("held-out evaluation", evaluation_path),
        )
        if not path.is_file()
    ]
    if missing:
        parser.error(
            "Run train.py and evaluate.py before building the campaign; missing %s"
            % ", ".join(missing)
        )
    project_root = _project_root(args.project_root)
    snr_grid = _finite_floats(args.snr_db, "snr-db")
    seeds = _integer_list(args.seeds, "seeds")
    if args.packet_bits < 2 or args.packet_bits % 2:
        raise ValueError("packet-bits must be a positive even integer")
    if args.bit_count < args.packet_bits or args.bit_count % args.packet_bits:
        raise ValueError("bit-count must be a positive multiple of packet-bits")
    block_size = int(args.block_size or args.packet_bits)
    if block_size < 1 or block_size > args.packet_bits:
        raise ValueError("block-size must be between 1 and packet-bits")

    recipe = _load_mapping(recipe_path, "source recipe")
    _require_step(recipe, "wireless_channel", "wireless.channel")
    _require_step(
        recipe,
        "carrier_impairment",
        "wireless.carrier_phase_impairment",
    )
    _require_step(
        recipe,
        "demodulator",
        "demodulation.phase_tracking_receiver_adapter",
    )
    _assert_held_out_operation_seeds(
        recipe_path,
        seeds,
        {
            "data": 0,
            "wireless_channel": 100_000,
            "carrier_impairment": 200_000,
        },
        {
            "data": "random_bits",
            "wireless_channel": "wireless_channel",
            "carrier_impairment": "carrier_phase_impairment",
        },
        master_offset=300_000,
    )
    artifact_evidence = _validate_artifact(
        artifact_path,
        project_root=project_root,
    )
    _load_json(history_path, "training history")
    evaluation = _load_json(evaluation_path, "evaluation metrics")
    if str(evaluation.get("component_sha256") or "") != artifact_evidence[
        "component_sha256"
    ]:
        raise ValueError(
            "evaluation metrics are not bound to the selected artifact component"
        )

    benchmark_recipe_path = _write_benchmark_recipe_source(
        recipe,
        output_path.parent / "benchmark_recipe.yaml",
    )

    pack = _build_pack(
        recipe_reference=_relative_path(benchmark_recipe_path, output_path.parent),
        artifact_reference=_relative_path(artifact_path, project_root),
        artifact_package_sha256=artifact_evidence["runtime_identity_sha256"],
        artifact_evidence=_evidence_file_spec(artifact_path, output_path.parent),
        training_history=_evidence_file_spec(history_path, output_path.parent),
        evaluation_metrics=_evidence_file_spec(evaluation_path, output_path.parent),
        snr_grid=snr_grid,
        seeds=seeds,
        bit_count=int(args.bit_count),
        packet_bits=int(args.packet_bits),
        block_size=block_size,
    )
    output_path.write_text(yaml.safe_dump(pack, sort_keys=False), encoding="utf-8")
    _print_commands(output_path, project_root)
    return 0


def _build_pack(
    *,
    recipe_reference: str,
    artifact_reference: str,
    artifact_package_sha256: str,
    artifact_evidence: Mapping[str, str],
    training_history: Mapping[str, str],
    evaluation_metrics: Mapping[str, str],
    snr_grid: list[float],
    seeds: list[int],
    bit_count: int,
    packet_bits: int,
    block_size: int,
) -> dict[str, Any]:
    runtime_identity = _required_sha256(
        artifact_package_sha256,
        "artifact_package_sha256",
    )
    methods = (
        (
            "uncompensated_qpsk",
            "Uncompensated QPSK",
            "baseline",
            {"mode": "uncompensated"},
        ),
        (
            "pilot_interpolation",
            "Pilot interpolation",
            "baseline",
            {"mode": "pilot_interpolation"},
        ),
        (
            "pilot_smoothing",
            "Pilot smoothing",
            "baseline",
            {
                "mode": "pilot_smoothing",
                "pilot_smoothing_nearest_pilots": 5,
            },
        ),
        (
            "decision_directed_pll",
            "Decision-directed PLL",
            "baseline",
            {"mode": "decision_directed_pll"},
        ),
        (
            "learned_receiver",
            "Learned temporal receiver",
            "candidate",
            {
                "mode": "learned_artifact",
                "artifact_manifest_path": artifact_reference,
                "artifact_entrypoint": "phase_tracking_receiver",
                "artifact_package_sha256": runtime_identity,
            },
        ),
        (
            "oracle_phase",
            "Oracle phase correction",
            "upper_bound",
            {"mode": "oracle"},
        ),
    )
    recipes = []
    for snr_db in snr_grid:
        for paired_seed in seeds:
            for method_id, label, role, receiver_params in methods:
                recipes.append(
                    {
                        "id": "%s_snr%s_seed%d"
                        % (method_id, _number_id(snr_db), paired_seed),
                        "label": "%s · %g dB · seed %d"
                        % (label, snr_db, paired_seed),
                        "role": role,
                        "path": recipe_reference,
                        "params": {
                            "method_id": method_id,
                            "matrix_selection": {
                                "channel.snr_db": float(snr_db),
                                "benchmark.paired_seed": int(paired_seed),
                            },
                            "metadata": {
                                "seed": int(paired_seed) + 300_000,
                                "benchmark_method": method_id,
                                "benchmark_method_label": label,
                                "benchmark_paired_seed": int(paired_seed),
                                "benchmark_held_out": True,
                            },
                            "step_params": {
                                "data": {
                                    "seed": int(paired_seed),
                                    "bit_count": int(packet_bits),
                                    "batch_size": int(bit_count // packet_bits),
                                },
                                "wireless_channel": {
                                    "snr_db": float(snr_db),
                                    "seed": int(paired_seed) + 100_000,
                                },
                                "carrier_impairment": {
                                    "seed": int(paired_seed) + 200_000,
                                },
                                "demodulator": dict(receiver_params),
                                "coded_bler": {"block_size": int(block_size)},
                            },
                        },
                    }
                )

    method_order = [item[0] for item in methods]
    series = [
        {"id": item[0], "label": item[1], "role": item[2]}
        for item in methods
    ]
    demo = {
        "schema_version": 1,
        "slug": "learned-qpsk-phase-tracking-receiver",
        "title": "Learning a packet-context QPSK phase tracker",
        "summary": (
            "Six receivers are compared under unknown packet phase, residual "
            "frequency offset, Wiener phase noise, and AWGN using paired payload "
            "and impairment seeds."
        ),
        "question": (
            "Can a learned temporal receiver use the same public pilots to improve "
            "on practical hand-designed phase trackers?"
        ),
        "tutorial": "../../../tutorials/learned_qpsk_phase_tracking_demo.html",
        "held_constant": [
            "QPSK framing and public pilot pattern",
            "carrier-impairment distribution",
            "paired payload, AWGN, and impairment seeds across methods",
        ],
        "changed": [
            "receiver implementation",
            "channel SNR",
            "held-out paired seed",
        ],
        "primary_metric": "channel.coded.ber",
        "comparison_axis": "channel.snr_db",
        "series": series,
        "plots": [
            {
                "id": "ber-vs-snr",
                "title": "Data-bit BER vs SNR",
                "kind": "line",
                "x": "channel.snr_db",
                "y": "channel.coded.ber",
                "group": "benchmark_method",
                "method_order": method_order,
                "style": {"aggregation": "mean_ci", "y_scale": "log"},
            },
            {
                "id": "bler-vs-snr",
                "title": "Data-block BLER vs SNR",
                "kind": "line",
                "x": "channel.snr_db",
                "y": "channel.coded.bler",
                "group": "benchmark_method",
                "method_order": method_order,
                "style": {"aggregation": "mean_ci", "y_scale": "log"},
            },
        ],
        "table_metrics": [
            "channel.snr_db",
            "channel.coded.ber",
            "channel.coded.bler",
            "channel.coded.error_count",
            "channel.coded.compare_bit_count",
            "channel.pilot_overhead_fraction",
            "channel.effective_payload_bits_per_channel_use",
            "channel.carrier.cfo_abs_mean_cycles_per_symbol",
            "channel.carrier.phase_noise_increment_std_rad",
            "channel.carrier.phase_drift_rms_rad",
        ],
        "training_evidence": [
            {
                "series": "learned_receiver",
                "trained_artifact_manifest": artifact_evidence,
                "training_history": training_history,
                "evaluation_metrics": evaluation_metrics,
            }
        ],
    }
    metrics = [
        ("channel.snr_db", "channel", "dB", "neutral"),
        ("channel.coded.ber", "transport", "fraction", "lower_is_better"),
        ("channel.coded.bler", "transport", "fraction", "lower_is_better"),
        ("channel.coded.error_count", "transport", "errors", "lower_is_better"),
        ("channel.coded.compare_bit_count", "transport", "bits", "neutral"),
        ("channel.pilot_overhead_fraction", "rate", "fraction", "lower_is_better"),
        (
            "channel.effective_payload_bits_per_channel_use",
            "rate",
            "bit/symbol",
            "higher_is_better",
        ),
        (
            "channel.carrier.cfo_abs_mean_cycles_per_symbol",
            "channel",
            "cycle/symbol",
            "neutral",
        ),
        (
            "channel.carrier.phase_noise_increment_std_rad",
            "channel",
            "rad",
            "neutral",
        ),
        (
            "channel.carrier.phase_drift_rms_rad",
            "channel",
            "rad",
            "neutral",
        ),
    ]
    return {
        "schema_version": 1,
        "id": "neural_receiver_ai_phy.learned_qpsk_phase_tracking_v2",
        "version": "2.0.0",
        "name": "Learned QPSK Phase Tracking Receiver",
        "description": (
            "Practical and learned carrier-phase trackers evaluated over a paired "
            "held-out impairment and SNR campaign."
        ),
        "suite": {
            "id": "neural_receiver",
            "name": "Neural Receiver / AI-PHY",
            "status": "experimental",
            "version": "v1-draft",
        },
        "dataset": {
            "id": "synthetic_random_bits",
            "modality": "bits",
            "version": "synthetic-random-bits-v1",
            "split": "held_out_seeded",
            "framing": "public_qpsk_pilots_v1",
        },
        "task": {
            "id": "neural_receiver_demapping",
            "kind": "transport_integrity",
            "modality": "bits",
        },
        "metrics": [
            {"id": key, "family": family, "unit": unit, "direction": direction}
            for key, family, unit, direction in metrics
        ],
        "baselines": [
            "uncompensated_qpsk",
            "pilot_interpolation",
            "pilot_smoothing",
            "decision_directed_pll",
        ],
        "recipes": recipes,
        "metadata": {
            "benchmark_tier": "experimental",
            "protocol_revision": "learned-qpsk-phase-tracking-post-training-v2",
            "trained_artifact_runtime_identity_sha256": runtime_identity,
            "paired_held_out_seeds": seeds,
            "snr_db_grid": snr_grid,
            "evaluation_bit_count_per_run": int(bit_count),
            "evaluation_packet_bits": int(packet_bits),
            "evaluation_packet_count_per_run": int(bit_count // packet_bits),
            "evaluation_block_size_bits": int(block_size),
            "oracle_is_upper_bound": True,
            "phase_truth_forwarded_to_learned_runtime": False,
            "demo": demo,
        },
    }


def _artifact_path(argument: str | None) -> Path:
    if argument:
        return Path(argument).expanduser().resolve()
    config = _load_mapping(Path("train_config.yaml"), "train_config.yaml")
    value = str((config.get("training") or {}).get("artifact_manifest_path") or "")
    if not value:
        raise ValueError("train_config.yaml does not declare artifact_manifest_path")
    return Path(value).expanduser().resolve()


def _project_root(argument: str | None) -> Path:
    if argument:
        return Path(argument).expanduser().resolve()
    config = _load_mapping(Path("train_config.yaml"), "train_config.yaml")
    value = str(config.get("project_root") or "")
    if not value:
        raise ValueError("train_config.yaml does not declare project_root")
    return Path(value).expanduser().resolve()


def _validate_artifact(path: Path, *, project_root: Path) -> dict[str, str]:
    manifest = _load_mapping(path, "trained artifact manifest")
    if int(manifest.get("schema_version") or 0) != 2:
        raise ValueError("trained artifact must use schema_version: 2")
    if str(manifest.get("kind") or "") != "noema.trained_block_artifact":
        raise ValueError("artifact kind must be noema.trained_block_artifact")
    _validate_learned_training_provenance(manifest)
    bindings = manifest.get("compatible_operations") or []
    compatible = next(
        (
            item
            for item in bindings
            if isinstance(item, Mapping)
            and str(item.get("operation") or "")
            == "demodulation.phase_tracking_receiver_adapter"
            and str(item.get("runtime_entrypoint") or "")
            == "phase_tracking_receiver"
        ),
        None,
    )
    if compatible is None:
        raise ValueError("artifact is not compatible with the phase-tracking receiver")
    components = manifest.get("components") or []
    if not isinstance(components, list) or not components:
        raise ValueError("artifact components must be a non-empty list")
    component = dict(components[0])
    component_path = path.parent / str(component.get("path") or "")
    if not component_path.is_file():
        raise FileNotFoundError("artifact component does not exist: %s" % component_path)
    expected = _required_sha256(component.get("sha256"), "component sha256")
    actual = _file_sha256(component_path)
    if actual != expected:
        raise ValueError("artifact component SHA-256 does not match")
    try:
        inspected = inspect_trained_artifact(
            path,
            project_root=project_root,
            registry=build_registry(),
        )
    except Exception as exc:
        raise ValueError("trained artifact runtime validation failed: %s" % exc) from exc
    if inspected.get("issues") or not inspected.get("ready"):
        details = list(inspected.get("issues") or [])
        details.extend(
            str(item)
            for item in (inspected.get("runtime") or {}).get(
                "unavailable_reasons",
                [],
            )
        )
        raise ValueError(
            "trained artifact is not runtime-ready: %s"
            % ("; ".join(details) or "runtime unavailable")
        )
    runtime_identity = _required_sha256(
        inspected.get("runtime_identity_sha256")
        or inspected.get("package_sha256"),
        "trained artifact runtime identity",
    )
    return {
        "manifest_sha256": _file_sha256(path),
        "component_sha256": actual,
        "runtime_identity_sha256": runtime_identity,
    }


def _validate_learned_training_provenance(
    manifest: Mapping[str, Any],
) -> None:
    training = manifest.get("training")
    if not isinstance(training, Mapping):
        raise ValueError(
            "learned phase-tracking artifact requires training provenance"
        )
    epoch = training.get("selected_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError(
            "learned phase-tracking artifact selected_epoch must identify a "
            "trained checkpoint after epoch zero"
        )
    required_truths = (
        "training_performed",
        "learned_checkpoint",
        "accepted_as_material_improvement",
    )
    missing = [name for name in required_truths if training.get(name) is not True]
    if missing:
        raise ValueError(
            "learned phase-tracking artifact lacks affirmative export provenance: %s"
            % ", ".join(missing)
        )
    if training.get("selected_epoch_is_pilot_smoothing_fallback") is not False:
        raise ValueError(
            "pilot-smoothing fallback or ambiguous fallback provenance cannot be "
            "published as a learned phase-tracking receiver"
        )


def _require_step(recipe: Mapping[str, Any], step_id: str, operation: str) -> None:
    step = next(
        (
            item
            for item in recipe.get("steps") or []
            if isinstance(item, Mapping) and str(item.get("id") or "") == step_id
        ),
        None,
    )
    if step is None or str(step.get("op") or "") != operation:
        raise ValueError("canonical recipe requires %s using %s" % (step_id, operation))


def _write_benchmark_recipe_source(
    recipe: Mapping[str, Any],
    destination: Path,
) -> Path:
    """Write the concrete source that benchmark entry overrides materialize.

    The reusable template owns a small UI matrix, while the publication
    benchmark owns its denser held-out SNR protocol.  A benchmark entry is
    already one concrete coordinate, so retaining the template matrix would
    leave two competing expansion instructions on the same recipe.
    """

    payload = copy.deepcopy(dict(recipe))
    metadata = dict(payload.get("metadata") or {})
    for field in (
        "matrix",
        "sweeps",
        "ui_sweeps",
        "matrix_selection",
        "matrix_variant_id",
        "matrix_index",
    ):
        metadata.pop(field, None)
    payload["metadata"] = metadata
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return destination.resolve()


def _assert_held_out_operation_seeds(
    recipe_path: Path,
    paired_seeds: list[int],
    step_offsets: Mapping[str, int],
    step_streams: Mapping[str, str],
    *,
    master_offset: int,
) -> None:
    """Reject benchmark RNG streams already used by generated captures."""

    captures: dict[str, tuple[Path, dict[str, Any]]] = {}
    for directory in (recipe_path.parent, recipe_path.parent.parent):
        for split in ("train", "validation", "test"):
            path = directory / ("capture_%s_recipe.yaml" % split)
            if path.is_file() and split not in captures:
                captures[split] = (
                    path,
                    _load_mapping(path, "%s capture recipe" % split),
                )
    missing = [
        split for split in ("train", "validation", "test") if split not in captures
    ]
    if missing:
        raise ValueError(
            "cannot prove benchmark seeds are held out; missing generated capture "
            "recipe(s): %s" % ", ".join(missing)
        )

    modulus = 2**31 - 1
    reserved_master: dict[int, str] = {}
    reserved_operation: dict[str, dict[int, str]] = {
        step_id: {} for step_id in step_offsets
    }
    for split, (path, capture_recipe) in captures.items():
        capture = dict(capture_recipe.get("dataset_capture") or {})
        mode = str(capture.get("seed_mode") or "fixed_seed")
        run_count = (
            int(capture.get("max_runs") or 1)
            if mode == "increment_run_seed"
            else 1
        )
        base_seed = int((capture_recipe.get("metadata") or {}).get("seed") or 0)
        base_namespace = str(
            (capture_recipe.get("metadata") or {}).get("seed_namespace")
            or capture_recipe.get("name")
            or ""
        )
        capture_split = str(capture.get("split") or split).strip() or "train"
        seed_namespace = "%s|dataset_capture_split=%s" % (
            base_namespace,
            capture_split,
        )
        steps = {
            str(item.get("id") or ""): item
            for item in capture_recipe.get("steps") or []
            if isinstance(item, Mapping)
        }
        for step_id in step_offsets:
            if step_id not in steps:
                raise ValueError("%s is missing required step %s" % (path, step_id))
        for run_index in range(max(1, run_count)):
            master_seed = (
                (base_seed + run_index) % modulus
                if mode == "increment_run_seed"
                else base_seed
            )
            master_seed = int(master_seed or modulus)
            reserved_master[master_seed] = split
            for step_id in step_offsets:
                params = dict(steps[step_id].get("params") or {})
                if params.get("seed") is not None:
                    operation_seed = int(params["seed"])
                else:
                    operation_seed = derive_seed(
                        master_seed,
                        seed_namespace,
                        step_id,
                        str(step_streams.get(step_id) or "default"),
                    )
                reserved_operation[step_id][operation_seed] = split

    overlaps = []
    for paired_seed in paired_seeds:
        benchmark_master = int(paired_seed) + int(master_offset)
        if benchmark_master in reserved_master:
            overlaps.append(
                "%d (benchmark master, %s capture)"
                % (paired_seed, reserved_master[benchmark_master])
            )
        for step_id, offset in step_offsets.items():
            operation_seed = int(paired_seed) + int(offset)
            if operation_seed in reserved_operation[step_id]:
                overlaps.append(
                    "%d (%s, %s capture)"
                    % (
                        paired_seed,
                        step_id,
                        reserved_operation[step_id][operation_seed],
                    )
                )
    if overlaps:
        raise ValueError(
            "benchmark seeds are not held out from dataset capture: %s"
            % ", ".join(overlaps)
        )


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("%s does not exist: %s" % (label, path))
    payload = load_strict_yaml_or_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("%s must contain a YAML mapping" % label)
    return dict(payload)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("%s does not exist: %s" % (label, path))
    payload = load_strict_yaml_or_json(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("%s must contain a non-empty JSON object" % label)
    return payload


def _finite_floats(value: str, label: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(not math.isfinite(item) for item in result):
        raise ValueError("%s must be a non-empty list of finite values" % label)
    return list(dict.fromkeys(result))


def _integer_list(value: str, label: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item < 0 for item in result):
        raise ValueError("%s must be non-empty nonnegative integers" % label)
    return list(dict.fromkeys(result))


def _required_sha256(value: Any, label: str) -> str:
    result = str(value or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError("%s must be a lowercase SHA-256 digest" % label)
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_file_spec(path: Path, relative_to: Path) -> dict[str, str]:
    return {
        "path": _relative_path(path, relative_to),
        "sha256": _file_sha256(path),
    }


def _number_id(value: float) -> str:
    return ("%.9g" % value).replace("-", "m").replace(".", "p")


def _relative_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _print_commands(output_path: Path, project_root: Path) -> None:
    print("wrote benchmark pack: %s" % output_path)
    print("from the Noema project root, validate:")
    print("  cd %s" % project_root)
    print("  noema benchmark validate %s" % output_path)
    print("from the Noema project root, run:")
    print("  cd %s" % project_root)
    print("  noema benchmark run %s" % output_path)
    print("after the run prints RESULT_ID, verify and publish:")
    print("  noema benchmark verify RESULT_ID")
    print(
        "  noema benchmark publish RESULT_ID --slug "
        "learned-qpsk-phase-tracking-receiver --out "
        "docs/demo/experiments/learned-qpsk-phase-tracking-receiver"
    )


if __name__ == "__main__":
    raise SystemExit(main())
