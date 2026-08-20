# External Adapter SDK

External adapters make researcher code a first-class Noema operation without editing the core
repository. An adapter is a small folder with:

- `noema_adapter.yaml`: operation IDs, wrapped Noema contract, callable location, and visible params.
- `adapter.py`: Python functions that implement the model or payload transform.

The registered operations appear in `noema ops`, recipe validation, graph rendering, run manifests,
dashboard block labels, timing summaries, and result artifacts exactly like built-in operations.

## Quick Start

Create a starter payload-bit image codec adapter:

```bash
uv run noema adapter scaffold adapters/my_codec --name my_codec --kind bits
uv run noema adapter validate adapters/my_codec/noema_adapter.yaml
```

Create a non-codec task metric or dataset adapter:

```bash
uv run noema adapter scaffold adapters/my_metric --name my_metric --kind classification_metric
uv run noema adapter scaffold adapters/my_dataset --name my_dataset --kind classification_dataset
```

Use it in CLI commands:

```bash
uv run noema --adapter adapters/my_codec recipe validate recipes/my_recipe.yaml
uv run noema --adapter adapters/my_codec recipe lint recipes/my_recipe.yaml
uv run noema --adapter adapters/my_codec recipe run recipes/my_recipe.yaml
```

Use it in the dashboard:

```bash
uv run noema --adapter adapters/my_codec ui serve --host 127.0.0.1 --port 8766
```

You can also set `NOEMA_ADAPTER_PATHS` to one or more manifest files or adapter directories,
separated by the platform path separator (`:` on Linux).

## Manifest Contract

Minimal manifest:

```yaml
schema_version: 1
name: my_codec
version: 0.1.0
operations:
  - id: model.my_codec_encode_bits
    name: My codec encoder
    wraps: model.external_encode_bits
    adapter:
      path: adapter.py
      callable: encode_bits
      call_style: array_params
    fixed_params:
      bit_storage: unpacked_bits
      bit_order: big
    differentiability:
      framework: blackbox
      gradient: none
      trainable_params: false
      exportable: false
      reason: "Declare torch/sionna + full/surrogate only when the adapter can be exported for training."
    params_schema:
      type: object
      properties:
        model_path:
          type: string
          default: ""
      additionalProperties: true
```

Checkpoint-backed adapter manifests can also declare how a trained model returned to Noema:

```yaml
schema_version: 1
name: my_deepjscc_method
version: 0.2.0
description: DeepJSCC checkpoint returned from a Noema differentiable export.
training:
  source_recipe: recipes/deepjscc_kodak_awgn_train.yaml
  source_recipe_sha256: "<sha256-of-source-recipe>"
  differentiable_export_id: 20260710T120000Z_deepjscc_export
  capture_id: optional_capture_id
  framework: torch
  checkpoint_path: checkpoints/model.pt
  checkpoint_sha256: optional_known_sha256
  input_schema:
    dtype: float32
    shape: [N, H, W, C]
  output_schema:
    dtype: complex64
    shape: [N, channel_uses]
  model_card: MODEL_CARD.md
operations:
  - id: model.my_deepjscc_encode
    name: My DeepJSCC encoder
    wraps: model.deepjscc_external_encode
    adapter:
      path: adapter.py
      callable: encode_symbols
      call_style: dict
  - id: model.my_deepjscc_decode
    name: My DeepJSCC decoder
    wraps: model.deepjscc_external_decode
    adapter:
      path: adapter.py
      callable: decode_symbols
      call_style: dict
```

Relative `training.checkpoint_path`, `training.source_recipe`, and `training.model_card` values are
resolved relative to `noema_adapter.yaml`. When `training.checkpoint_path` is present, validation
requires the file to exist and computes `training.checkpoint_sha256`. If you provide
`checkpoint_sha256`, Noema checks that it matches the file. The normalized training metadata is passed
to adapter callables as `params["training"]`; `params["checkpoint_path"]` and
`params["checkpoint_sha256"]` are also populated when they are not already supplied by operation
params. Run summaries, artifacts, and manifests record the same metadata so benchmark results can be
traced back to the differentiable export and checkpoint.

`wraps` chooses the built-in contract and artifact handling:

| Wrapped operation | Input artifact | Output artifact |
| --- | --- | --- |
| `model.external_encode_bits` | `image.batch.numpy` | `channel.payload_bits.numpy` |
| `model.external_decode_bits` | `channel.payload_bits.numpy` or `channel.bits.numpy` | `image.batch.numpy` |
| `model.external_encode_indices` | `image.batch.numpy` | `semantic.indices.numpy` |
| `model.external_decode_indices` | `semantic.indices.numpy` | `image.batch.numpy` |
| `model.external_encode_latents` | `image.batch.numpy` | `semantic.latents.numpy` |
| `model.external_decode_latents` | `semantic.latents.numpy` | `image.batch.numpy` |
| `model.deepjscc_external_encode` | `image.batch.numpy` | `channel.symbols.complex_numpy` |
| `model.deepjscc_external_decode` | `channel.rx_symbols.complex_numpy` or `channel.symbols.complex_numpy` | `image.batch.numpy` |
| `source.external_classification_dataset` | none | `task.labels.json`, `task.predictions.json` |
| `metrics.external_classification` | `task.labels.json`, `task.predictions.json` | `metrics.report` |

Relative `adapter.path` values are resolved relative to the manifest file. `adapter.module` can be
used instead of `adapter.path` for importable Python packages.

Adapter registration imports and resolves every declared callable before an execution plan can create
a run directory. Registration is all-or-nothing across the discovered manifests: one missing import
does not leave earlier adapter operations partially installed. Provenance binds the manifest, callable
file, and a deterministic inventory of local Python sources under the adapter package root; any byte
change requires rebuilding the registry and plan.

This preflight is a **trusted-code boundary**, not a sandbox. Importing an adapter executes Python in
the Noema process. Only load adapters and Python/pickle-based checkpoints from trusted producers.
Prefer data-only artifacts such as NPZ, ONNX, or Safetensors; when a PyTorch checkpoint is unavoidable,
use a supported `weights_only=True` loader and independently pin its digest. The adapter source-tree
fingerprint detects drift but does not make imported code safe or authenticate its author.

## Callable Signatures

The default `call_style` is `array_params`:

```python
def encode_bits(images, params):
    ...
    return {"array": bits, "metadata": {"bit_count": int(bits.size)}}
```

Other supported call styles:

- `array`: `fn(array)`
- `dict`: `fn({"array": array, "metadata": metadata, "params": params})`

The callable may return either a NumPy-compatible array or:

```python
{"array": array, "metadata": {...}}
```

For image decoders, return `uint8` images shaped `[N, H, W, 3]`.

## Differentiability Metadata

External adapter operations may include optional `differentiability` metadata with the same fields as
built-in operations:

| Field | Values |
| --- | --- |
| `framework` | `torch`, `sionna`, `tensorflow`, `numpy`, `blackbox`, `none` |
| `gradient` | `full`, `stop`, `surrogate`, `none` |
| `trainable_params` | `true` or `false` |
| `exportable` | `true` or `false` |
| `reason` | optional human-readable explanation |

If omitted, the wrapped operation's metadata is inherited. If neither the adapter nor wrapped
operation declares metadata, Noema reports the safe default: NumPy, no gradient, no trainable params,
not exportable. `trainable_params` is legacy compatibility metadata: setting it to `true` grants
neither **Built-in fine-tuning** nor **Portable replacement**. Built-in fine-tuning requires both an
implementation-owned `fine_tuning_supported` declaration and a callable `fine_tuning_provider`;
portable replacement requires a
validated operation `trained_artifact_abi` plus a compatible runtime binding. External manifest
differentiability metadata is useful only for describing an adapter retained as unchanged support on
a live gradient route.

A portable ABI must bind both `artifact_manifest_path` and `artifact_entrypoint`, and the bound
entrypoint must equal the ABI's `entrypoint_id`. It must also declare a non-empty `component_id` and
`component_role`; this matches the artifact import gate. Parameter metadata alone cannot advertise
portable replacement.

Task dataset and metric wrappers use `call_style: dict` by default:

```python
def load_classification_examples(request):
    return {
        "examples": [
            {"id": "a", "label": "clear", "prediction": "clear"},
            {"id": "b", "label": "faded", "prediction": "clear"},
        ],
        "metadata": {"dataset": "my_dataset"},
    }


def score_classification(request):
    reference = request["reference"]
    candidate = request["candidate"]
    return {
        "metrics": {"external.classification.accuracy": 0.5, "task.accuracy": 0.5},
        "per_example": [],
    }
```

Run the bundled non-codec adapter example:

```bash
uv run noema --adapter examples/adapters/classification_task \
  benchmark run examples/benchmarks/external_classification_adapter_smoke.yaml
```

## Bit Payloads

For `model.external_encode_bits`, Noema always stores and transmits canonical unpacked channel bits:

- dtype: `np.uint8`
- shape: flat vector
- values: `0` or `1`
- one array element is one bit

If your encoder naturally emits packed bytes, set:

```yaml
fixed_params:
  bit_storage: packed_bytes
  bit_order: big
```

Then return byte data and provide `metadata.bit_count` when the last byte contains padding. Noema
will unpack before channel transmission and repack before `model.external_decode_bits`.

## Timing

Manifest operations inherit the timing behavior of the wrapped external operation. Today that is a
single local-Python adapter-call timing:

- encoder wrappers record `encoder.external_call`
- decoder wrappers record `decoder.external_call`

If your adapter internally measures model inference, payload packing, or native runtime calls, return
those detailed measurements in metadata in a later SDK revision. The current SDK intentionally records
the whole adapter call so no external model is silently treated as a built-in runtime.

## Validation

```bash
uv run noema adapter validate adapters/my_codec/noema_adapter.yaml --json
```

Validation checks:

- manifest schema version
- unique operation IDs inside the manifest
- wrapped operation IDs exist
- checkpoint-backed training metadata is well formed
- `training.checkpoint_path` exists when declared
- `training.checkpoint_sha256` is computed, and checked if a value is declared
- callable path/module can be imported
- callable object exists and is callable
- visible operation input/output kinds match the wrapped Noema contract

Use `--no-import` only when you want a structural check without importing researcher code.

After adapter validation, run `recipe lint` on the recipe that uses the adapter. Lint checks the
platform-level invariants around the adapter, including canonical `uint8` unpacked bit boundaries,
fixed channel count checks, symbol boundaries for DeepJSCC-style adapters, and explicit model
conversion artifacts for non-native runtimes.
