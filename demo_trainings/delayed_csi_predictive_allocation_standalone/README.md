# Standalone predictive allocation with delayed CSI

This is a small, isolated proof-of-mechanism experiment. It asks whether two
old CSI reports from a correlated two-path OFDM channel can support a learned
predictive power allocation that beats:

1. equal power, and
2. Shannon water-filling performed on the newest available, but stale, CSI.

The channel contains a persistent Doppler direction, so the temporal history
is informative. The learned policy receives two noisy CSI snapshots and never
receives the current channel. Every method has the same nonnegative sum-power
constraint and is scored by the same achievable-rate endpoint.

The demo also reports a model-aware predictive water-filling baseline, a
known-delayed-state causal reference optimizer, and perfect-current-CSI
water-filling. The last method is a noncausal oracle: a learned delayed-CSI
policy cannot legitimately beat it on the Shannon-rate objective.

## Run

From the repository root:

```bash
.venv/bin/python -I -B \
  demo_trainings/delayed_csi_predictive_allocation_standalone/demo.py self-test
.venv/bin/python -I -B \
  demo_trainings/delayed_csi_predictive_allocation_standalone/demo.py train
.venv/bin/python -I -B \
  demo_trainings/delayed_csi_predictive_allocation_standalone/demo.py development
.venv/bin/python -I -B \
  demo_trainings/delayed_csi_predictive_allocation_standalone/demo.py freeze-final
.venv/bin/python -I -B \
  demo_trainings/delayed_csi_predictive_allocation_standalone/demo.py heldout
```

The five commands can also be run as one resumable command:

```bash
.venv/bin/python -I -B \
  demo_trainings/delayed_csi_predictive_allocation_standalone/demo.py all
```

Outputs are written under
`.noema/demos/delayed_csi_predictive_allocation_v2/`. Existing training and
held-out artifacts are reused rather than overwritten or rerun.

### Development provenance

A preliminary v1 development run used the same simulator coordinate. Its
negative control incorrectly treated *any* departure from equal power as a
failure, including the expected result that a stale-dependent allocation is
worse when the current channel is independent. Before opening any v2 held-out
seed, v2 changed that control to the correct one-sided rule—no positive gain
over equal power—and restarted training, validation, development, and final
seed namespaces. The v2 result is therefore a prospective held-out
confirmation, but the simulator coordinate itself is not outcome-naive.

## Scope

This is a controlled physical channel mechanism case, not a field deployment,
NR codec, or realistic-domain benchmark. A positive result demonstrates that
predictable temporal structure can make delayed CSI useful; it does not repair
or replace any previously completed held-out study.
