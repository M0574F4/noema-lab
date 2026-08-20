# Rate and Channel Accounting

Noema uses fixed boundary operations so rate and channel measurements are comparable across tasks.

## Canonical Bit Contract

At channel-facing bit boundaries, bits are stored as:

- dtype: `np.uint8`
- shape: flat vector
- values: `0` or `1`
- one array element is one bit

Packed byte streams are allowed inside a codec or adapter, but they must be unpacked before channel
transmission. Metadata should record the original byte count and valid bit count.

## Fixed Points

Publishable bit-transport recipes should include:

```text
payload_bit_boundary -> channel_encoder -> tx_bit_boundary -> channel -> rx_bit_boundary -> channel_bit_count_match
```

Use these meanings:

- payload bits: codec or payload codec output before channel coding;
- transmitted bits: bitstream entering modulation or identity channel;
- received bits: bitstream leaving demodulation or identity channel;
- channel bit-count match: explicit assertion that transmitter/receiver channel boundaries have
  matching lengths.

For DeepJSCC/symbol recipes, use `tx_symbol_boundary`, `rx_symbol_boundary`, and
`channel_symbol_count_match`.

Power normalization is an explicit contract parameter. `normalization_scope: source_item` scales
each declared source item independently and is the default for publication protocols;
`normalization_scope: global` scales the complete tensor or stream once. The Torch training block
uses the same selector and treats tensor axis 0 as the source-item axis. A protocol must not compare
a global-normalized training path with a source-item-normalized benchmark path.

`wireless.channel.receiver_processing` also belongs to the physical contract. `matched` selects the
preset's declared receiver processing (including perfect-CSI zero-forcing for the currently
supported fading presets), while `none` requests an explicitly supported raw receive signal.
Materializations are bound to channel, receiver-processing, and channel-state selectors; an
unsupported combination is rejected during planning rather than discovered after a run starts.

## Disabled Channels

A disabled physical channel is still represented by an identity channel step. This keeps the graph,
rate accounting, and result manifests structurally comparable to noisy channel recipes.

## Protected Digital Baselines

Protected digital baselines do not send JPEG or learned-compression payload bytes directly through a
noisy physical channel. They use this explicit chain:

```text
payload bits -> packet/CRC -> LDPC encode -> QPSK -> AWGN -> soft demap -> LDPC decode -> CRC/failure policy
```

The portable smoke path uses the same accounting spine with `channel.packetize_crc32`, a repetition
channel code, QPSK, measured AWGN, hard decisions plus LLR artifacts, repetition decoding, and
`channel.crc32_check`. It remains a plumbing/smoke baseline and must not be relabelled as a
publication-grade FEC comparator.
Packet protocol v4 carries a compact, unrepeated ownership/offset/length header and its CRC in-band,
while a canonical SHA-256-bound packet-layout contract in immutable run evidence fixes the complete
source-item population, exact lengths, packet offsets, and ownership. The receiver validates every
recoverable header against that contract but does not use a damaged header to infer denominators or
output length. Total header loss therefore becomes an outage for a known source item instead of a
framing exception or a shortened payload. `gray_image` and `erasure` zero the complete failed item;
`report_outage` retains the exact-length best-effort received payload while still recording failure.
All header, CRC, padding, and contract-bound packet costs remain explicit in framed/transmitted-bit
and channel-use accounting. The contract is trusted control-plane evidence, not over-the-air side
information; its integrity is covered by run-bundle verification.

The standards-oriented digital path uses `channel.nr_ldpc_encoder` and
`channel.nr_ldpc_decoder`, backed by Sionna 2.x/PyTorch `TBEncoder`/`TBDecoder` and configured against 3GPP
TS 38.212 V18.8.0 (Release 18). It is a transport-block coder over an abstract flat QPSK/AWGN
link, not a full NR MCS, PRB/resource-grid, layer-mapping, DM-RS, control-channel, or PUSCH waveform
implementation. It records the effective transport-block size, TB and code-block
CRCs, segmentation count, base graph, lifting size/set, mother-code dimensions, circular-buffer
rate-matched codeword lengths, scrambling parameters, modulation order/layers, decoder iterations,
and per-block CRC status. The decoder accepts the soft-LLR artifact, checks the encoder-bound profile
identity, backend versions, and contiguous block partition, and applies the declared `zero_fill`,
`keep_estimate`, or `raise` policy on TB-CRC failure. The decoder iteration count is fixed by the
encoder-bound profile rather than accepted as an unrecorded receive-side override. Noema is not yet
published on PyPI. With an explicitly supplied wheel, install this optional path with
`python -m pip install "noema-lab[wireless] @ file:///absolute/path/to/noema_lab-<version>-<platform-tag>.whl"`;
in a source checkout, use `uv sync --extra wireless`.

High-SNR round trips and tamper tests establish local consistency, not standards conformance.
Standards-facing claims require independent standard vectors or a second implementation and frozen
BER/BLER sweeps with negative controls. A development-only feasibility pilot freezes the resource
budget and operating points before sealed-test access; the threshold cannot be retuned after
definitive outcomes.

`channel.communication_resource_accounting.v2` makes the additive boundaries executable while
requiring exact semantic kinds: `channel.payload_bits.numpy`,
`channel.framed_bits.numpy`, `channel.coded_bits.numpy`, and
`channel.symbols.complex_numpy`. The matching `channel.packetize_crc32.v2` emits the framed-bit
kind and accepts only the exact payload-bit kind. Generic `channel.bits.numpy` checkpoints are
useful for role-neutral bit-exact inspection but are deliberately rejected by the strict packetizer
and accounting ports.

The unversioned packetizer and accounting operations remain registered with their historical
contracts so archived recipes retain their meaning. They must not be cited as evidence that strict
kind separation was enforced.

The current NR encoder also rebuilds its output array descriptor from the actual coded vector
instead of inheriting the framed-input shape. Exact 4,320-to-8,640 and 8,640-to-17,280 regression
cases check the physical NPZ, embedded metadata, and persisted run summary. Historical v3 compact
evidence retained the stale shape descriptor but not the coded arrays themselves, so that study
cannot be retroactively promoted to bit-exact FEC evidence.

The strict operation records:

```text
native codec bits
  + declared codec wire overhead
  = serialized payload bits
  + packet/framing overhead
  = framed bits
  + FEC/rate-matching overhead
  = coded bits
  + modulation padding
  = transmitted padded bits
```

Pilot, physical-header, data, and grid-padding symbols are recorded separately and sum to total
channel uses. JPEG and CompressAI encoders expose native per-item byte/bit counts independently of
the selected wire format, so a native rate-distortion coordinate cannot silently count wrapper
bytes. Their `wire_format` parameter keeps `safe_json_base64` as the artifact-compatibility default
and offers `compact_binary_v1` (`noema.codec_wire.v1`) for modeled communication. The compact form
is bounded, codec-specific, independently decodable per source item, and carries native entropy
streams plus only receiver-required fields; it does not use pickle or executable-object
deserialization. Resource accounting records the selected payload format and both the generic wire
overhead and the backward-compatible safe-wrapper field. The report content-binds all fixed points
with an accounting identity.

Recipes may also use `channel.capacity_oracle_digital_link` as a clearly labeled theoretical
reference. It assumes QPSK, AWGN, and a declared code rate, then delivers the payload perfectly only
when:

```text
source_bits <= channel_uses * log2(1 + snr)
```

Otherwise it records an outage and applies the recipe's decode-failure policy, such as a gray image.
The operation reports packet success/outage, payload bits, coded/transmitted bits, channel uses,
bits per pixel, per-item channel uses, and maximum per-item channel uses per pixel. These numbers are useful for sanity checks and tutorial
plots, but they must be cited as a capacity-oracle baseline rather than a measured LDPC modem.

## Reporting

For OFDM-style implementations, `channel.channel_use_count` is the number of resource elements in
the grid that was actually executed. It includes zero padding required to fill the last grid;
`channel.payload_symbol_count` records the unpadded symbols and
`channel.grid_padding_symbol_count` records the difference. Energy and average power per executed
use are reported separately, so padding cannot disappear from the resource coordinate. When items
are independently realized, the same charged counts are retained per source item. For a joint grid,
shared tail padding is reported separately and charged in full to every item's maximum-use bound;
this conservative rule cannot make an item appear cheaper because it happened to share a batch.

An explicit Sionna OFDM state must be declared with `channel_state_mode: explicit`. Sampled channel
materializations use `channel_state_mode: none`. Sionna 2.x random streams are process-global, so
Noema serializes seed assignment and stochastic Sionna execution inside one critical section;
NumPy and Torch paths continue to use local generators. An omitted operation `seed` is derived from
the recipe master seed, namespace, step ID, and stream. An explicit operation seed is an intentional
override and is recorded as such.

Frozen paper bundles and manifests retain the Sionna 1.x/TensorFlow versions under which they were
created. This migration affects new executions only; historical evidence is never relabelled.

Benchmark packs should report public metric IDs such as:

- `codec.native_bit_count`
- `codec.serialized_payload_bit_count`
- `channel.payload_bit_count`
- `channel.framed_bit_count`
- `channel.coded_bit_count`
- `channel.transmitted_bit_count`
- `channel.channel_use_count`
- `channel.payload_symbol_count`
- `channel.grid_padding_symbol_count`
- `channel.ber`
- `rate.native_codec_bpp`
- `rate.serialized_payload_bpp`
- `rate.framed_bpp`
- `rate.coded_bpp`
- `rate.padded_bpp`

Step-local metric paths such as `tx_bit_boundary.channel.fixed.tx.bit_count` belong in benchmark
metadata as fixed-point evidence.
