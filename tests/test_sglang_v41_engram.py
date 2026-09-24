import pytest
import torch

from archlab.serving.sglang_v41_engram import BF16EngramEmbedding


@pytest.mark.parametrize("rows", [3, 19])
def test_bf16_row_shards_reassemble_exactly_without_quantization(rows):
    torch.manual_seed(79)
    weights = torch.randn(rows, 16).bfloat16()
    indices = torch.tensor([[0, rows - 1, 1], [1, 0, rows - 1]])
    contributions = []
    for rank in range(8):
        table = BF16EngramEmbedding(rows, 16, tp_rank=rank, tp_size=8,
                                    all_reduce=lambda x: x)
        table.weight.weight_loader(table.weight, weights)
        table.finish_load("test")
        contributions.append(table.owned_rows(indices))
        assert table.weight.dtype == torch.bfloat16
    torch.testing.assert_close(torch.stack(contributions).sum(0), weights[indices],
                               rtol=0, atol=0)


def test_rejects_unloaded_or_requantized_weights_and_cp():
    table = BF16EngramEmbedding(3, 4, tp_rank=0, tp_size=1, all_reduce=lambda x: x)
    indices = torch.tensor([[0, 2]])
    with pytest.raises(ValueError, match="not loaded"):
        table(indices)
    with pytest.raises(ValueError, match="BF16"):
        table.weight.weight_loader(table.weight, torch.ones(3, 4))
    weights = torch.randn(3, 4).bfloat16()
    table.weight.weight_loader(table.weight, weights)
    torch.testing.assert_close(table(indices), weights[indices], rtol=0, atol=0)
    with pytest.raises(ValueError, match="context parallelism"):
        table(indices, cp_all_tokens=True)


def test_row_shards_may_arrive_out_of_order_but_must_cover_every_owned_row():
    weights = torch.arange(19 * 4).reshape(19, 4).bfloat16()
    indices = torch.tensor([[0, 5, 12, 18]])
    parts = [(0, weights[:3]), (3, weights[3:11]), (11, weights[11:])]
    outputs = []
    for rank in range(8):
        table = BF16EngramEmbedding(19, 4, tp_rank=rank, tp_size=8, all_reduce=lambda x: x)
        for first, value in reversed(parts):
            table.load_piece(first, value)
        table.finish_load("sharded")
        outputs.append(table.owned_rows(indices))
    torch.testing.assert_close(torch.stack(outputs).sum(0), weights[indices], rtol=0, atol=0)
    table = BF16EngramEmbedding(19, 4, tp_rank=0, tp_size=1, all_reduce=lambda x: x)
    table.load_piece(0, weights[:3])
    with pytest.raises(ValueError, match="overlapping"):
        table.load_piece(2, weights[2:5])
    with pytest.raises(ValueError, match="not loaded"):
        table.finish_load("incomplete")


def test_streaming_native_bucket_finish_does_not_commit_partial_table():
    table = BF16EngramEmbedding(4, 2, tp_rank=0, tp_size=1, all_reduce=lambda x: x)
    table.transaction_open = True
    table.load_piece(0, torch.ones(2, 2, dtype=torch.bfloat16))
    table.finish_load("native bucket")
    assert not table._loaded
    with pytest.raises(ValueError, match="not loaded"):
        table.owned_rows(torch.tensor([0]))
    table.transaction_open = False
    with pytest.raises(ValueError, match="not loaded"):
        table.finish_load("premature commit")
    table.load_piece(2, torch.ones(2, 2, dtype=torch.bfloat16))
    table.finish_load("complete commit")
    assert table._loaded
