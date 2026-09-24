from types import SimpleNamespace

import pytest
import torch

from archlab.serving.sglang_v41_padding import install_live_source_boundary


@pytest.mark.parametrize("real,padded,lengths", [(100, 104, [37, 63]), (1, 8, [1]), (0, 8, [])])
def test_padded_extend_only_writes_real_source_rows(real, padded, lengths):
    class Backend:
        calls = 0

        def forward_low_ratio_sources(self, *, x, q_lora, positions, forward_batch):
            self.calls += 1
            ids = torch.repeat_interleave(torch.arange(len(lengths)),
                                         torch.tensor(lengths), output_size=positions.numel())
            assert ids.numel() == real
            torch.testing.assert_close(x, values[:real])
            torch.testing.assert_close(q_lora, values[:real])
            self.written = positions.tolist()

    class Attention:
        def forward(self, *args, **kwargs):
            raise AssertionError("unexpected attention execution")

        def _forward_prepare(self, x, positions, batch, backend):
            backend.forward_low_ratio_sources(x=x, q_lora=x, positions=positions, forward_batch=batch)
            # The downstream attention and communication must retain padding.
            return x

    values = torch.randn(padded, 3)
    backend, attention = Backend(), Attention()
    install_live_source_boundary(attention)
    batch = SimpleNamespace(num_token_non_padded_cpu=real, extend_seq_lens_cpu=lengths,
                            forward_mode=SimpleNamespace(is_extend=lambda: True))
    result = attention._forward_prepare(values, torch.arange(padded), batch, backend)
    assert result is values
    assert backend.calls == int(real > 0)
    if real:
        assert backend.written == list(range(real))


def test_decode_hash_padding_cannot_overwrite_a_live_request_history():
    from archlab.serving.sglang_v41_padding import install_hash_padding_boundary

    class Hasher:
        pad_row = 4

        def forward(self, ids, batch):
            self.history = {}
            for slot, token in zip(batch.req_pool_indices.tolist(), ids.tolist(), strict=True):
                self.history[slot] = token
            return ids

    hasher = Hasher()
    install_hash_padding_boundary(hasher)
    batch = SimpleNamespace(num_token_non_padded_cpu=1, req_pool_indices=torch.zeros(8, dtype=torch.long),
                            forward_mode=SimpleNamespace(is_decode=lambda: True))
    hasher.forward(torch.tensor([17] + [0] * 7), batch)
    assert hasher.history == {0: 17, 4: 0}
    assert batch.req_pool_indices.tolist() == [0] * 8


@pytest.mark.parametrize("padded", [0, 8])
def test_idle_attention_and_hash_never_enter_stateful_native_paths(padded):
    from archlab.serving.sglang_v41_padding import install_hash_padding_boundary

    class Native:
        primes = torch.empty(2)
        offsets = torch.empty(2, 3)

        def forward(self, *args, **kwargs):
            raise AssertionError("idle batch reached stateful native path")

        def _forward_prepare(self, *args, **kwargs):
            raise AssertionError("idle batch reached native attention")

    batch = SimpleNamespace(num_token_non_padded_cpu=0)
    attention, hasher = Native(), Native()
    install_live_source_boundary(attention)
    install_hash_padding_boundary(hasher)
    result = attention.forward(torch.ones(padded, 16), torch.arange(padded), batch)
    assert result.shape == (padded, 16)
    assert torch.count_nonzero(result) == 0
    hashes = hasher.forward(torch.ones(padded, dtype=torch.int64), batch)
    assert hashes.shape == (padded, 2, 3)
    assert torch.count_nonzero(hashes) == 0
