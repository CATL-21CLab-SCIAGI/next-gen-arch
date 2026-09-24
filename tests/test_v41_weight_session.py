import pytest

from archlab.megatron.miles_v41_weight_session import begin, end


def test_only_full_base_transaction_is_supported():
    begin([], "all", sync_base=True)
    end([])
    with pytest.raises(ValueError):
        begin([], "all", sync_base=False)
    with pytest.raises(ValueError):
        begin([], "draft")
    with pytest.raises(ValueError):
        end([], expected_lora_checksums={})
