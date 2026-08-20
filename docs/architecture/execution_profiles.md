# Recipe Execution Profiles

An execution profile names the stable execution spine of a recipe. It is a topology contract,
not a task label, model family, channel preset, or benchmark-maturity claim. Every top-level recipe
declares the contract at its root:

```yaml
execution_profile:
  id: layered_digital
  version: 1
```

The profile version changes only when the required spine changes. Changing an SNR, dataset,
algorithm, channel model, or operation parameter does not create a new profile version.

Profile declarations are enforced by the shared planner before a run directory is created. A recipe
that declares a standard profile but omits, reorders, or substitutes an incompatible required stage
is invalid; the profile is not merely a UI annotation. Lint uses the same inspection rules but
returns the complete set of profile issues in its report.

## `layered_digital`

This profile keeps source coding, channel coding, and physical transport as distinct stages. Its
required fixed-point spine is:

```text
payload_bit_boundary
  -> [packetizer]
  -> channel_encoder
  -> tx_bit_boundary
  -> physical transport
  -> rx_bit_boundary
  -> channel_bit_count_match
  -> channel_decoder
  -> [crc_check]
```

The application-specific sender and receiver sit outside that spine. Physical transport may be
expanded as `modulator -> [tx_power] -> wireless_channel -> demodulator`, or represented by a fused
bit-link operation when the recipe intentionally abstracts the PHY. Optional BER, BLER, power,
rate, and task metrics observe the spine without changing it.

JPEG, learned image codecs, UTF-8 text transport, protected digital links, neural receivers, and
OFDM resource-allocation recipes all use this profile. They differ in implementation, not in the
layered execution boundary.

## `joint_source_channel_symbols`

This profile transports learned semantic/channel symbols directly. The sender and receiver are
fused source-channel stages rather than a source codec followed by a separately exposed digital
channel code:

```text
sender
  -> tx_power
  -> tx_symbol_boundary
  -> wireless_channel
  -> rx_symbol_boundary
  -> channel_symbol_count_match
  -> receiver
```

`tx_power` is the canonical executable step ID for symbol-power normalization or allocation; the
operation and its human-readable label may describe the specific policy. The wireless stage may be
a physical channel or an explicit identity symbol link. Image DeepJSCC and text JSCC therefore
share one profile even though their sources, models, and evaluation metrics differ.

## `csi_feedback_downlink`

This profile represents a feedback/control loop whose output configures a subsequent downlink
transmission. It is not a forward payload codec. Its required spine is:

```text
channel_state
  -> feedback_encoder
  -> feedback_link
  -> feedback_decoder
  -> precoder
  -> evaluation
```

The UE observes realized downlink CSI, compresses it into a limited feedback representation, and
returns it over an explicit feedback link. The base station reconstructs CSI and computes a
precoder from that reconstruction. Evaluation must use the original realized CSI—not the
reconstruction—to measure achieved downlink rate. The first standard recipe assumes perfect CSI at
the UE and a quantized, error-free feedback link.

Encoder and decoder are one jointly trained, atomically returned artifact. Changing their
architecture, compression dimension, quantizer resolution, TDL preset, antenna count, or downlink
SNR does not create another execution profile.

## Estimation, sensing, and decision profiles

These profiles make domain-specific data flow explicit without selecting a particular template or
algorithm:

| Profile | Required ordered spine |
| --- | --- |
| `pilot_channel_estimation` | channel realization → pilot pattern → pilot observation → channel estimator → evaluation |
| `mimo_ofdm_channel_estimation` | MIMO-OFDM channel → OFDM pilot grid → pilot observation → channel estimator → evaluation |
| `beamforming_link_evaluation` | link scenario → beamformer → link evaluation |
| `range_localization` | anchor/tag geometry → range observations → localizer → position evaluation |
| `aoa_array_estimation` | angular scene → array observations → angle estimator → angle evaluation |
| `task_inference` | task data → inference/decision → task evaluation |
| `task_evaluation` | task data → task evaluation |

Each named stage is enforced during planning. The domain profiles also constrain the operation IDs
at stable physical boundaries—for example, `range_localization` cannot silently substitute an AoA
estimator for its localizer. Classical and trainable methods share a profile when they implement
the same boundary, so LS versus an estimator adapter, MUSIC versus an AoA adapter, and MRT versus a
beamforming adapter do not create new profiles.

`task_inference` is the general topology for VQA, retrieval, detection, and segmentation graphs
whose task data passes through an explicit inference or decision block. `task_evaluation` is kept
for genuinely direct scoring graphs, such as a smoke fixture that supplies reference and candidate
labels itself and therefore has no inference stage. Neither profile identifies the task: the
research-purpose declaration still owns that meaning.

The two channel-estimation profiles intentionally separate flat/SISO pilot experiments from a
MIMO-OFDM pilot-grid contract. Antenna count, subcarrier count, pilot spacing, SNR, and algorithm
remain parameters rather than new profiles. The planner pins only the scenario discriminators that
make the contracts distinct: `flat_siso` plus a unit pilot pattern for the first profile, and
`mimo_ofdm` plus a comb pilot grid for the second.

## `custom`

`custom` is an explicit declaration that a recipe does not claim one of the standard execution
or domain spines. It is appropriate for a genuinely novel topology or a contract smoke test that
does not realize a standard path. Custom does not mean unsupported or lower quality; it means
consumers must inspect the typed graph instead of assuming a standard sequence.

## What Does Not Create a Profile

Profiles stay deliberately small. The following choices are orthogonal to execution topology:

- Task or modality: image, text, bits, sensing, reconstruction, and classification.
- Algorithm: JPEG, CompressAI, repetition coding, a neural receiver, equal power, or water filling.
- Channel realization: identity, AWGN, Rayleigh fading, OFDM/TDL, or another backend preset.
- Parameters and sweeps: SNR, noise variance, transmit-power budget, code rate, and seeds.
- Side branches: CSI generation or power-control inputs, diagnostics, and metric observers. A CSI
  branch does not change a layered payload profile; `csi_feedback_downlink` applies only when the
  feedback-to-precoding loop is itself the executable spine.
- Benchmark status: smoke, experimental, and canonical benchmark tiers are protocol-maturity labels,
  not execution profiles.

A CSI or allocator branch may feed the transmitter or channel while the primary data path remains
`layered_digital`. Likewise, comparing layered digital transport with joint source-channel coding in
one benchmark does not create a third “hybrid” profile; the benchmark simply contains recipes with
two different declared profiles.
