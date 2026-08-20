# Transmitted codec payload security

Noema wraps JPEG, CompressAI, TCM, HPCM, and EVC entropy streams in the
data-only wire format `noema.safe_data_json_base64.v1`. The wrapper begins with
a fixed Noema magic marker and a versioned JSON envelope. Bytes inside the
typed data tree are encoded as canonical base64. The decoder supports only
string-keyed dictionaries, lists, tuples, bytes, and finite scalar values; it
does not import modules, resolve class names, or construct application objects.

Parsing is bounded before and during tree decoding. The current defaults cap
the complete wire payload at 256 MiB, decoded binary content at 128 MiB, each
string at 16 MiB, the tree at 1,000,000 values and 64 levels, and integers at
128 decimal digits. Unknown tags, duplicate JSON keys, non-canonical encodings,
non-finite numbers, invalid Unicode, and limit violations fail closed.

Legacy pickle payloads are intentionally incompatible. There is no automatic
fallback because detecting and then deserializing one would restore arbitrary
code execution at the receiver. Existing stored results whose
`payload_format` names a pickle wrapper must be regenerated. The wrapper is
part of the transmitted bit count, so regenerated rate measurements can differ
from old measurements because canonical JSON and base64 have different framing
overhead.

This boundary protects payload decoding, not model execution. Optional
third-party codec repositories still supply Python model definitions and native
entropy extensions and must be treated as trusted code. Direct TCM and HPCM
checkpoint loads use PyTorch's `weights_only=True`; EVC delegates checkpoint
reading to the upstream repository's `get_state_dict`, whose implementation and
repository revision remain in the trusted-code boundary.
