import torch

from lmpool.layers.rotary_embedding import RotaryEmbedding


def test_rotary_cache_uses_configured_default_dtype():
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        rotary = RotaryEmbedding(base=10_000, rotary_embedding=8, max_position=16)
    finally:
        torch.set_default_dtype(previous_dtype)

    assert rotary.cos_sin_cache.dtype is torch.bfloat16


def test_rotary_output_preserves_query_dtype_with_float32_cache():
    rotary = RotaryEmbedding(base=10_000, rotary_embedding=8, max_position=16)
    assert rotary.cos_sin_cache.dtype is torch.float32

    positions = torch.tensor([0, 1, 2], dtype=torch.long)
    query = torch.randn(3, 2, 8, dtype=torch.bfloat16)
    key = torch.randn(3, 1, 8, dtype=torch.bfloat16)

    query_out, key_out = RotaryEmbedding.forward.__wrapped__(
        rotary,
        positions,
        query,
        key,
    )

    assert query_out.dtype is torch.bfloat16
    assert key_out.dtype is torch.bfloat16
    assert torch.isfinite(query_out).all()
    assert torch.isfinite(key_out).all()
