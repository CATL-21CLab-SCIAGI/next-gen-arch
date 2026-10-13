"""CPU cache-bridge contracts; actual TileLang/vLLM qualification is separate."""
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from archlab.training import serving_conversion as bridge


def _oracle(q, k, v, short, *, scaling, long_window, short_window, decode=False):
    """Independent joint-softmax arithmetic on chronological unpaged tensors."""
    k1, v1 = short
    groups = q.shape[2] // k.shape[2]
    k, v, k1, v1 = [t.float().repeat_interleave(groups, dim=2) for t in (k, v, k1, v1)]
    outputs = []
    for index in range(q.shape[1]):
        end = k.shape[1] if decode else index + 1
        short_end = k1.shape[1] if decode else index + 1
        ks, vs = k[:, max(0, end - long_window):end], v[:, max(0, end - long_window):end]
        k1s, v1s = (t[:, max(0, short_end - short_window):short_end] for t in (k1, v1))
        scores = torch.einsum("bhd,bjhd,bkhd->bhjk", q[:, index].float(), k1s, ks) * scaling
        probabilities = scores.flatten(-2).softmax(-1).reshape_as(scores)
        outputs.append(torch.einsum("bhjk,bjhd,bkhd->bhd", probabilities, v1s, vs))
    return torch.stack(outputs, dim=1).to(q.dtype)


class _CacheOwner(nn.Module):
    def __init__(self, name, *, hnd=False):
        super().__init__()
        self.layer_name = name
        if hnd:
            cache = torch.empty(2, 12, 2, 16, 8, dtype=torch.bfloat16).transpose(2, 3)
        else:
            cache = torch.empty(2, 12, 16, 2, 8, dtype=torch.bfloat16)
        self.register_buffer("kv_cache", cache.fill_(float("nan")), persistent=False)
        self.context = (None, self, self.kv_cache, None)
        self.calls = 0

    def forward(self, q, k, v):
        self.calls += 1
        _, _, cache, slots = self.context
        for index, slot in enumerate(slots.tolist()):
            if slot >= 0:
                block, offset = divmod(slot, cache.shape[2])
                cache[0, block, offset] = k[index]
                cache[1, block, offset] = v[index]
        return torch.full_like(q, float("nan"))  # Must never become the joint output.


def _set_metadata(owner, tokens, length, blocks=(3, 1, 4, 2, 7, 5)):
    table = torch.tensor([blocks], dtype=torch.int32)
    positions = torch.arange(length - tokens, length)
    slots = table[0, positions // 16].long() * 16 + positions % 16
    metadata = SimpleNamespace(query_start_loc=torch.tensor([0, tokens], dtype=torch.int32),
                               seq_lens=torch.tensor([length], dtype=torch.int32), block_table=table,
                               num_actual_tokens=tokens, causal=True, use_cascade=False,
                               common_prefix_len=0)
    owner.context = (metadata, owner, owner.kv_cache, slots)


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(bridge, "_attention_context", lambda owner: owner.context)
    monkeypatch.setattr(bridge, "_prefill", _oracle)
    monkeypatch.setattr(bridge, "_decode", lambda q, k, v, short, *, scaling,
                        query_position=None, long_window=None: _oracle(
        q, k, v, short, scaling=scaling, long_window=long_window or k.shape[1],
        short_window=short[0].shape[1], decode=True))
    return SimpleNamespace(num_heads=4, num_kv_heads=2, head_dim=8, scaling=.17,
                           is_global=False, window_span=32,
                           _archlab_conversion_config=SimpleNamespace(short_window=16),
                           archlab_conversion_long_attention=_CacheOwner("long"),
                           archlab_conversion_short_attention=_CacheOwner("short"))


def _projections(length, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(1, length, heads, 8, generator=generator).to(torch.bfloat16)
            for heads in (4, 2, 2, 2, 2)]


def _run(native, tensors):
    q, k, v, k1, v1 = tensors
    native._archlab_conversion_short = (k1, v1)
    with torch.no_grad():
        output, probabilities = bridge._paged_joint_attention(
            native, *(t.transpose(1, 2) for t in (q, k, v)), None, scaling=native.scaling,
        )
    assert probabilities is None
    return output


@pytest.mark.parametrize("prefill", [1, 15, 16, 17, 31, 32, 33, 34, 35])
@pytest.mark.parametrize("global_layer", [False, True])
@pytest.mark.parametrize("hnd", [False, True])
def test_prefill_decode_across_short_long_and_page_boundaries(native, prefill, global_layer, hnd):
    native.is_global = global_layer
    if hnd:
        native.archlab_conversion_long_attention = _CacheOwner("long", hnd=True)
        native.archlab_conversion_short_attention = _CacheOwner("short", hnd=True)
    owners = (native.archlab_conversion_long_attention, native.archlab_conversion_short_attention)
    history = _projections(prefill + 1)
    for owner in owners:
        _set_metadata(owner, prefill, prefill)
    actual = _run(native, [t[:, :prefill] for t in history])
    expected = _oracle(*history[:3], tuple(history[3:]), scaling=native.scaling,
                       long_window=100 if global_layer else 32, short_window=16)
    torch.testing.assert_close(actual, expected[:, :prefill], rtol=0, atol=0)
    for owner in owners:
        _set_metadata(owner, 1, prefill + 1)
    actual = _run(native, [t[:, -1:] for t in history])
    torch.testing.assert_close(actual, expected[:, -1:], rtol=0, atol=0)
    assert [owner.calls for owner in owners] == [2, 2]


def test_disjoint_requests_use_only_their_package_block_tables(native):
    owners = (native.archlab_conversion_long_attention, native.archlab_conversion_short_attention)
    first = _projections(40, seed=11)
    for owner in owners:
        _set_metadata(owner, 39, 39)
    _run(native, [t[:, :39] for t in first])
    # A new request has disjoint physical pages and a fresh complete prompt.
    second = _projections(18, seed=29)
    for owner in owners:
        _set_metadata(owner, 17, 17, blocks=(8, 6))
    _run(native, [t[:, :17] for t in second])
    for owner in owners:
        _set_metadata(owner, 1, 18, blocks=(8, 6))
    output = _run(native, [t[:, -1:] for t in second])
    expected = _oracle(*second[:3], tuple(second[3:]), scaling=native.scaling,
                       long_window=32, short_window=16)
    torch.testing.assert_close(output, expected[:, -1:], rtol=0, atol=0)
    # Returning to request A reads its retained pages, not request B's history.
    for owner in owners:
        _set_metadata(owner, 1, 40)
    output = _run(native, [t[:, -1:] for t in first])
    expected = _oracle(*first[:3], tuple(first[3:]), scaling=native.scaling,
                       long_window=32, short_window=16)
    torch.testing.assert_close(output, expected[:, -1:], rtol=0, atol=0)


def test_decode_retains_masked_history_but_passes_the_effective_window(native, monkeypatch):
    history = _projections(38, seed=81)
    owners = (native.archlab_conversion_long_attention, native.archlab_conversion_short_attention)
    calls = []

    def decode(q, k, v, short, *, scaling, query_position, long_window):
        calls.append((query_position, long_window, k.shape[1], short[0].shape[1]))
        return _oracle(q, k, v, short, scaling=scaling, long_window=long_window,
                       short_window=16, decode=True)

    monkeypatch.setattr(bridge, "_decode", decode)
    for owner in owners:
        _set_metadata(owner, 34, 34)
    _run(native, [t[:, :34] for t in history])
    for position in range(34, 38):
        for owner in owners:
            _set_metadata(owner, 1, position + 1)
        output = _run(native, [t[:, position:position + 1] for t in history])
        expected = _oracle(*[t[:, :position + 1] for t in history[:3]],
                           tuple(t[:, :position + 1] for t in history[3:]),
                           scaling=native.scaling, long_window=32, short_window=16)
        torch.testing.assert_close(output, expected[:, -1:], rtol=0, atol=0)
    assert calls == [(position, 32, 35, 16) for position in range(34, 38)]


def test_dummy_profiling_never_writes_request_caches(native):
    tensors = _projections(7)
    actual = _run(native, tensors)
    expected = _oracle(*tensors[:3], tuple(tensors[3:]), scaling=native.scaling,
                       long_window=32, short_window=16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for owner in (native.archlab_conversion_long_attention, native.archlab_conversion_short_attention):
        assert owner.calls == 0 and bool(owner.kv_cache.isnan().all())


@pytest.mark.parametrize("invalid", ["batch", "chunk", "different_lengths", "layout", "dtype",
                                     "null_block", "slot", "prefix", "noncausal", "missing_fields"])
def test_unsupported_metadata_and_layout_fail_before_cache_writes(native, invalid):
    owners = (native.archlab_conversion_long_attention, native.archlab_conversion_short_attention)
    for owner in owners:
        _set_metadata(owner, 2, 2)
    owner = owners[1]
    metadata, _, cache, slots = owner.context
    if invalid == "batch":
        metadata.query_start_loc = torch.tensor([0, 1, 2])
    elif invalid == "chunk":
        for item in owners:
            _set_metadata(item, 2, 4)
    elif invalid == "different_lengths":
        _set_metadata(owner, 2, 3)
    elif invalid == "layout":
        owner.context = (metadata, owner, cache[0], slots)
    elif invalid == "dtype":
        owner.context = (metadata, owner, cache.float(), slots)
    elif invalid == "null_block":
        metadata.block_table[0, 0] = 0
    elif invalid == "slot":
        slots[-1] += 1
    elif invalid == "prefix":
        metadata.common_prefix_len = 1
    elif invalid == "noncausal":
        metadata.causal = False
    else:
        del metadata.seq_lens
    with pytest.raises(ValueError):
        _run(native, _projections(2))
    assert [item.calls for item in owners] == [0, 0]


def _config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1,
                                        decode_context_parallel_size=1, prefill_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=1, enable_chunked_prefill=False),
        model_config=SimpleNamespace(enforce_eager=True, hf_config=SimpleNamespace(
            adapter={}, conversion={"short_window": 16, "backend": "tilelang"})),
        cache_config=SimpleNamespace(enable_prefix_caching=False, cache_dtype="auto"),
        quant_config=None, speculative_config=None,
        attention_config=SimpleNamespace(backend="FLASH_ATTN", flash_attn_version=2),
    )


@pytest.mark.parametrize("section,key,value", [
    ("parallel_config", "tensor_parallel_size", 2),
    ("parallel_config", "pipeline_parallel_size", 2),
    ("parallel_config", "decode_context_parallel_size", 2),
    ("parallel_config", "prefill_context_parallel_size", 2),
    ("scheduler_config", "max_num_seqs", 2),
    ("scheduler_config", "enable_chunked_prefill", True),
    ("model_config", "enforce_eager", False),
    ("cache_config", "enable_prefix_caching", True),
    ("cache_config", "cache_dtype", "fp8"),
    ("attention_config", "backend", "FLASHINFER"),
    ("attention_config", "flash_attn_version", 3),
])
def test_serving_settings_fail_closed(section, key, value):
    config = _config()
    bridge.validate_conversion_serving(config)
    setattr(getattr(config, section), key, value)
    with pytest.raises(ValueError):
        bridge.validate_conversion_serving(config)


def test_factory_preserves_global_window_and_checks_actual_fa_version(monkeypatch):
    constructed = []

    class Attention:
        version = 2

        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.impl = SimpleNamespace(vllm_flash_attn_version=self.version)

        def get_attn_backend(self):
            return SimpleNamespace(get_name=lambda: "FLASH_ATTN")

    module = ModuleType("vllm.model_executor.layers.attention")
    module.Attention = Attention
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = _config()
    config.cache_config.sliding_window = 1024
    factory = bridge.stock_attention_factory(config)
    native = SimpleNamespace(num_heads=4, num_kv_heads=2, head_dim=8, scaling=.17)
    factory(native, window=None, prefix="global.long")
    factory(native, window=16, prefix="global.short")
    assert [item["per_layer_sliding_window"] for item in constructed] == [None, 16]
    assert all(item["cache_config"].sliding_window is None for item in constructed)
    assert config.cache_config.sliding_window == 1024
    Attention.version = 3
    with pytest.raises(ValueError, match="actually use FlashAttention-2"):
        factory(native, window=None, prefix="unsupported")


def test_profiling_context_does_not_require_slot_mapping(monkeypatch):
    context_module = ModuleType("vllm.forward_context")
    context_module.get_forward_context = lambda: SimpleNamespace(attn_metadata=None, slot_mapping=None)
    attention_module = ModuleType("vllm.model_executor.layers.attention.attention")
    attention_module.get_attention_context = lambda name: pytest.fail("dummy context has no mapping")
    monkeypatch.setitem(sys.modules, context_module.__name__, context_module)
    monkeypatch.setitem(sys.modules, attention_module.__name__, attention_module)
    owner = _CacheOwner("profile")
    assert bridge._attention_context(owner) == (None, owner, owner.kv_cache, None)


def test_binding_keeps_native_forward_and_installs_two_distinct_cache_owners():
    from archlab.architectures.limite_conversion import (
        LimiteConversionConfig,
        LimiteSimplicialAttention,
    )

    class Native(nn.Module):
        def __init__(self, index, window):
            super().__init__()
            self.config = SimpleNamespace(_attn_implementation="sdpa")
            self.qkv_proj = nn.Linear(8, 384, bias=False)
            self.kv_size, self.num_kv_heads, self.head_dim = 128, 1, 128
            self.num_heads, self.scaling = 1, .25
            self.attention_dropout = 0.0
            self.layer_idx, self.window_span, self.is_global = index, window, window is None

        def forward(self, *args):
            return "publisher"

    layers = [SimpleNamespace(self_attn=LimiteSimplicialAttention(
        Native(index, window), LimiteConversionConfig(backend="tilelang")))
        for index, window in enumerate((32, None))]
    policy = SimpleNamespace(model=SimpleNamespace(layers=layers))
    construction = []

    def factory(native, *, window, prefix):
        construction.append((native, window, prefix))
        return _CacheOwner(prefix)

    original = Native.forward
    owners = bridge.bind_conversion_attention(policy, factory, prefix="model.")
    assert len(owners) == 4 and len({id(owner) for owner in owners}) == 4
    assert [window for _, window, _ in construction] == [35, 16, None, 16]
    assert all(name.startswith("model.layers.") for _, _, name in construction)
    assert Native.forward is original
    for layer in layers:
        replacement = layer.self_attn
        assert replacement.native.forward.__func__ is original
        assert replacement._joint_forward.__globals__["ALL_ATTENTION_FUNCTIONS"].get_interface(
            "unused", None) is bridge._paged_joint_attention
        assert replacement.native.config._attn_implementation == "archlab_paged_conversion"
