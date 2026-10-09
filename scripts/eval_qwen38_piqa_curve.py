#!/usr/bin/env python3
"""Compatibility CLI; implementation lives in archlab.evaluation.qwen38_piqa."""

from archlab.evaluation.qwen38_piqa import (
    _canonical_sha256 as _canonical_sha256,
)
from archlab.evaluation.qwen38_piqa import (
    _checkpoint_identity as _checkpoint_identity,
)
from archlab.evaluation.qwen38_piqa import (
    _validate_cached_result as _validate_cached_result,
)
from archlab.evaluation.qwen38_piqa import main

if __name__ == "__main__":
    main()
