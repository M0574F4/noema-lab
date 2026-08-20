from __future__ import annotations

import hashlib
import json
from typing import Iterable, List

import numpy as np


def content_sha256_item_ids(items: Iterable[np.ndarray]) -> List[str]:
    """Return stable content identities for array-backed samples."""

    identities: List[str] = []
    for item in items:
        contiguous = np.ascontiguousarray(item)
        digest = hashlib.sha256()
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(json.dumps(list(contiguous.shape)).encode("utf-8"))
        digest.update(contiguous.tobytes())
        identities.append("sha256:%s" % digest.hexdigest())
    return identities
