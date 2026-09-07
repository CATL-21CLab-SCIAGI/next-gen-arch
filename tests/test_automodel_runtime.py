"""CPU checks for the narrow in-process FLA configuration restriction."""

from types import SimpleNamespace

import pytest

from archlab.automodel.runtime import restrict_autotuner


def test_restrict_existing_configs_and_clear_unsafe_cache():
    configs = [SimpleNamespace(num_warps=w, num_stages=s, kwargs={"BV": 32})
               for w in (2, 4) for s in (2, 3, 4)]
    tuner = SimpleNamespace(configs=configs, cache={"old": configs[-1]}, cache_results=True)
    wrapped = SimpleNamespace(fn=SimpleNamespace(fn=tuner))
    evidence = restrict_autotuner(wrapped, num_warps=2, num_stages=4)
    assert evidence["before_count"] == 6 and evidence["after_count"] == 1
    assert tuner.configs == [configs[2]]  # retain the existing Config object
    assert tuner.cache == {} and not tuner.cache_results
    assert restrict_autotuner(wrapped, num_warps=2, num_stages=4)["after_count"] == 1


def test_reject_missing_safe_configs_and_unrecognized_wrapper():
    with pytest.raises(RuntimeError, match="wrapper"):
        restrict_autotuner(object(), num_warps=2)
    tuner = SimpleNamespace(configs=[SimpleNamespace(num_warps=4, num_stages=2)])
    with pytest.raises(RuntimeError, match="absent"):
        restrict_autotuner(tuner, num_warps=2)
