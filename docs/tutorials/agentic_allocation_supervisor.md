# Agentic Supervisory Allocation

This tutorial specifies a small agentic-AI experiment in which a model is a component of the
wireless system: it observes completed-link telemetry and public channel configuration, then chooses
which existing allocation policy will control the next complete Noema run. The model is a slow
supervisor. It never chooses a power value for each OFDM symbol or subcarrier.

```{note}
This is an experimental tutorial harness, not a validated agent-performance benchmark. The example
is inspired by the broad agent-and-tool workflow direction of
[MX-AI](https://arxiv.org/abs/2508.09197), but it is **not a reproduction** of that paper's system
model, tools, prompts, harness, experiments, or results.
```

The key rule is that the contract fixes the observations, actions, budgets, failures, and evaluation
rules; it does not fix the agent's actions. Users may change the provider, model, or prompt while the
experiment contract remains unchanged.

## Why this is supervisory

The underlying
[delayed-CSI allocation example](reliability_aware_ofdm_allocation_demo.md) already has a fast
allocator operating over short OFDM allocation states. Calling a language model in that loop would
make model latency part of the radio-control problem. This tutorial instead makes one decision after
a Noema run has completed and holds the selected allocator policy for the whole next run.

```text
completed run -> allowlisted telemetry -> one supervisor decision -> next complete run
                                            |
                                            +-> existing fast allocator handles all OFDM states
```

Decision latency is still measured. A timeout causes the declared fallback before the next run; the
tutorial does not pretend that model calls are free or instantaneous.

## Reused wireless system

The base recipe is `recipes/resource_delayed_csi_finite_blocklength.yaml`. It supplies:

- a Sionna 3GPP TDL-C OFDM trajectory with 128 subcarriers;
- four delayed, noisy complex-CSI snapshots whose newest sample is five OFDM symbols old;
- a causal allocator boundary that cannot consume the later channel state;
- equal-power, observed-CSI water-filling, uncertainty-shrunk water-filling, and causal-AR
  water-filling policies; and
- expected finite-blocklength goodput, predicted BLER, and power-constraint metrics.

The tutorial contract is `examples/agentic_allocation/contract.yaml`; its replaceable prompt is
`examples/agentic_allocation/prompt.md`.

## Contract at a glance

### Declared objective

The primary objective is to minimize the mean selected average transmit-power budget while keeping
mean predicted BLER at or below `0.25`. Expected finite-blocklength goodput is the declared reporting
tie-break when two valid controllers use the same mean power. The BLER and goodput values come from
the existing parallel-complex-AWGN finite-blocklength normal approximation.
The contract records `aggregation: mean`, the primary direction, the aggregate constraint, and the
goodput tie-break structurally; the campaign summary reports feasibility and orders only feasible
arms by those declared fields.

Predicted BLER is **not measured decoder BLER** and is not evidence about a particular deployed FEC
implementation. A study that needs measured decoder failures must replace the objective and recipe,
not silently relabel this metric.

### Observations and causality

Every observation carries an RFC3339 wall-clock timestamp, a decision index, the completed run that
made feedback available, and simulation-time indices for the transmitter-visible CSI summary.

The model may see only:

- the next run's public configuration: noise variance, CSI age/history length, estimation SNR,
  mobility, FFT size, and allocation interval;
- the immediately preceding completed run's action and outcome feedback; and
- summary statistics derived from delayed/noisy `csi_observation.transmitter_csi`.

The model must never receive:

- `csi_observation.actual_state` or a current/future channel realization;
- environment seeds;
- the outcome of the run it is about to configure;
- `resource.csi.observed_actual_*` metrics, which use hidden current truth; or
- the full allocation preview, whose `channel_gain`, `unit_power_snr_db`, and `inverse_unit_snr`
  fields expose evaluation-only current-channel information.

The harness therefore constructs provider input from an allowlist. Passing the complete Noema
metrics report to the model is a causality violation even though that report is valid evaluation
evidence after the run.

### Allowed actions

The single `configure_allocator` tool selects one of four existing policies and one normalized
average transmit-power budget from `[0.4, 0.6, 0.8, 1.0, 1.4]`:

| Policy | Fixed policy configuration in this v1 contract |
| --- | --- |
| `fixed` | none |
| `observed_csi_water_filling` | none |
| `robust_csi_water_filling` | `csi_gain_shrinkage=0.6` |
| `causal_ar_water_filling` | `csi_prediction_gain_confidence=0.4`; horizon equals CSI age |

The action has exactly two arguments: `policy` and `power_budget`. Policy-specific parameters remain
fixed so every provider receives the same compact action space. The harness applies the selected
budget to both `channel_state.average_power_budget` and `tx_power.target_power`; contradictory or
off-grid values are invalid. The agent cannot change seeds, other channel settings, the objective,
or the recipe topology, and it cannot return a direct power vector. Existing Noema constraints
continue to enforce nonnegative allocation and the exact per-state sum-power budget.

### Interaction and reset

The tutorial-sized contract contains two episodes with two complete Noema runs each and an explicit
paired seed schedule. At each between-run boundary the harness builds one observation, validates at most one
tool call, applies the selected policy and power budget to the next run, and retains that run's
feedback for the following decision. Reset clears controller memory and restarts the decision index;
the recorded episode seeds remain hidden from the provider.

## Run the deterministic smoke example

The default backend is `scripted`. It follows a deterministic valid action sequence and makes no
language-model performance claim. It exists so the episode, evidence, fallback, and verification paths can be
replayed without a model server or network access. With the agent, four static comparators, and the
rule-based comparator, the supplied two-by-two schedule executes 24 paired Noema runs.
Those wireless runs and their compressed evidence can take several minutes and meaningful disk
space on a laptop; use the validator before starting the campaign.

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra wireless noema agentic run \
  examples/agentic_allocation/contract.yaml \
  --provider scripted \
  --out .noema/agentic/delayed_csi_supervisor_scripted

uv run noema agentic verify \
  .noema/agentic/delayed_csi_supervisor_scripted
```

The retained evidence should bind the contract and prompt hashes, provider configuration, complete
observation/action history, effective recipe hashes, run IDs, seeds, decision costs, validation
failures, and fallbacks. It also retains deterministic gzip copies of each underlying run summary
and manifest, and verifies their run IDs, summary binding, metrics, and authored/effective recipe
hashes, so the result remains inspectable if the workspace run store later moves. Credentials and
authorization headers must never be retained.

Replay executes the recorded effective actions with the same paired seeds and never calls the
model:

```bash
uv run --extra wireless noema agentic replay \
  .noema/agentic/delayed_csi_supervisor_scripted \
  --out .noema/agentic/delayed_csi_supervisor_replay
```

The replay report records metric hashes and any mismatches; it does not reinterpret or regenerate
the original decisions.

## Bring your own model and prompt

The provider interface is intentionally separate from the action contract. Changing a model or
prompt must not expand what the model can observe or do. Copy `prompt.md`, edit it, and pass the new
path explicitly; the runner records its normalized UTF-8 text in the content-bound effective
contract and prompt evidence.

### Ollama

Use any locally installed Ollama model, including a Qwen-family model that fits the available CPU
and memory. The exact tag is user-selected and becomes part of the evidence.

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra wireless noema agentic run \
  examples/agentic_allocation/contract.yaml \
  --provider ollama \
  --base-url http://127.0.0.1:11434 \
  --model <ollama-model-tag> \
  --prompt examples/agentic_allocation/prompt.md \
  --out .noema/agentic/delayed_csi_supervisor_ollama
```

### OpenAI-compatible endpoint

This route works with a user-supplied server that implements the expected OpenAI-compatible chat and
tool-call surface. Keep credentials in the environment, not in the contract. For safety, a contract
may name only a credential variable in the `NOEMA_AGENT_*` namespace; omit `--api-key-env` for an
unauthenticated local endpoint.

```bash
cd "$(git rev-parse --show-toplevel)"
export NOEMA_AGENT_API_KEY=<provider-key>
uv run --extra wireless noema agentic run \
  examples/agentic_allocation/contract.yaml \
  --provider openai_compatible \
  --base-url <https://provider.example/v1> \
  --model <provider-model-id> \
  --api-key-env NOEMA_AGENT_API_KEY \
  --prompt examples/agentic_allocation/prompt.md \
  --out .noema/agentic/delayed_csi_supervisor_compatible
```

### Local Transformers model

For a direct CPU run, install the existing `textgen` extra and choose a sufficiently small
instruction model. A Qwen model ID is one option; pin its immutable revision for a result intended
for comparison.

```bash
cd "$(git rev-parse --show-toplevel)"
uv sync --extra wireless --extra textgen
uv run --extra wireless --extra textgen noema agentic run \
  examples/agentic_allocation/contract.yaml \
  --provider transformers \
  --model <hugging-face-model-id> \
  --model-revision <immutable-revision> \
  --device cpu \
  --prompt examples/agentic_allocation/prompt.md \
  --out .noema/agentic/delayed_csi_supervisor_transformers
```

Model weights can dominate laptop memory and runtime. Start with the scripted backend to validate
the experiment, then use a small CPU-suitable model. A larger model changes the controller under
test, not the wireless contract. The Transformers adapter requires a 40- or 64-character immutable
commit digest rather than a mutable branch name. Remote and Ollama campaigns retain the configured
model identifier and any safe model/fingerprint metadata returned by the provider, but a provider
that exposes no immutable revision cannot offer the same weight-level reproducibility.

As a concrete CPU compatibility check, the development harness was exercised with
`Qwen/Qwen2.5-0.5B-Instruct` pinned to revision
`7ae557604adf67be50417f59c2c2f167def9a775`. It produced a valid bounded action through the
Transformers adapter; this is a backend smoke check, not an endorsed model or an agent-quality
benchmark. Pre-cache weights before a campaign when model download or cold loading could exceed
the contract's decision deadline. The local adapter normalizes a few common single-tool JSON
wrappers, then applies the same strict policy and power-budget validation as every other provider.

## Failure and cost accounting

Each decision has a `30 s` timeout, one model-call budget, one tool-call budget, and no retry after
an invalid action. Provider errors, timeouts, missing tool calls, malformed JSON, unknown policies,
extra parameters, and out-of-grid values all select the declared `fixed` equal-power fallback at
power budget `1.4`.

A Python thread cannot safely cancel an in-process generation. If a local call crosses its deadline,
the harness takes the fallback and quarantines that backend for the remainder of the campaign;
later decisions also fall back without starting overlapping generations or resetting mutable model
state. The original in-process call may still finish in its daemon thread; use an isolated model
server when hard cancellation and resource isolation are required.

For every decision, report:

- end-to-end decision latency;
- provider-, model-, and tool-call counts;
- input/output token counts when the provider exposes them;
- normalized response hash and validation status;
- failure class and whether fallback was used; and
- requested and effective actions.

Fallback runs remain part of the agent result. Dropping them would hide a system-level failure mode.

## Comparators and pairing

Compare the agent against each of the four static policies at the recipe's reference budget `0.8`
and against the declared rule-based supervisor. The static arms isolate policy choice without
turning this tutorial into a 20-arm sweep; the agent and rule-based arm may still select any budget
in the declared grid. The rule-based supervisor receives exactly the same observations and action
space as the model; it selects robust water filling at budget `1.4` when the latest predicted BLER
exceeds a `0.20` safety threshold, otherwise causal-AR water filling at budget `0.8`. The margin
below the declared `0.25` ceiling makes this a conservative baseline rather than an oracle tuned to
the current hidden channel. With no completed history at an episode's first decision, it takes the
conservative robust branch.

Every method must use the same ordered public contexts and paired environment seeds. Compare
mean selected power, expected goodput, predicted BLER, reliability-constraint violations,
power-constraint errors, decision latency, invalid actions, timeouts, and fallback frequency. A
perfect-current-CSI method, if shown, is a non-deployable diagnostic reference rather than a
comparator.

Verify each result before interpreting the comparison:

```bash
uv run noema agentic verify .noema/agentic/<result-directory>
```

The result supports a bounded statement about supervisory policy selection under this contract. It
does not establish general RAN autonomy, real-time feasibility, standards conformance, field
performance, or reproduction of MX-AI.
