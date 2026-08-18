"""Splitting a model must not change its output."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoModelForCausalLM, AutoTokenizer

from p2pgpu.cluster.stage import (
    StageError,
    block_kwargs,
    embed_inputs,
    final_norm,
    find_layer_list,
    rotary_embeddings,
)

MODELS = [
    ("sshleifer/tiny-gpt2", "transformer.h"),
    ("hf-internal-testing/tiny-random-LlamaForCausalLM", "model.layers"),
]


def _load(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    return tok, model


def _run_pipeline(model, layers, ids, cut):
    """Two stages, with a clone at the boundary standing in for the wire."""
    with torch.no_grad():
        hidden = embed_inputs(model, ids)
        pos = torch.arange(ids.shape[-1]).unsqueeze(0)
        rot = rotary_embeddings(model, hidden, pos)
        for lo, hi in [(0, cut), (cut, len(layers))]:
            for i in range(lo, hi):
                block = layers[i]
                out = block(hidden, **block_kwargs(block, hidden, pos, rot))
                hidden = out[0] if isinstance(out, tuple) else out
            hidden = hidden.clone()
        norm = final_norm(model)
        return norm(hidden) if norm is not None else hidden


@pytest.mark.parametrize("model_id,expected_path", MODELS)
def test_layer_list_is_found(model_id, expected_path):
    _tok, model = _load(model_id)
    _layers, path = find_layer_list(model)
    assert path == expected_path


@pytest.mark.parametrize("model_id,_path", MODELS)
def test_split_output_matches_whole_model(model_id, _path):
    """The whole point: cutting the stack must be numerically invisible."""
    tok, model = _load(model_id)
    layers, _ = find_layer_list(model)
    ids = tok("distributed computing across two machines", return_tensors="pt").input_ids

    with torch.no_grad():
        reference = model(ids, output_hidden_states=True).hidden_states[-1]

    for cut in range(1, len(layers) + 1):
        got = _run_pipeline(model, layers, ids, cut)
        assert torch.allclose(reference, got, atol=1e-4), f"mismatch at cut={cut}"


def test_gpt2_positional_embedding_is_not_skipped():
    """GPT-2 adds wpe before block 0; omitting it fails silently, not loudly."""
    tok, model = _load("sshleifer/tiny-gpt2")
    ids = tok("hello world", return_tensors="pt").input_ids
    with_pos = embed_inputs(model, ids)
    token_only = model.get_input_embeddings()(ids.long())
    assert not torch.allclose(with_pos, token_only)


def test_llama_blocks_receive_rotary_embeddings():
    """Llama blocks produce garbage without position_embeddings."""
    _tok, model = _load("hf-internal-testing/tiny-random-LlamaForCausalLM")
    layers, _ = find_layer_list(model)
    hidden = torch.randn(1, 4, model.config.hidden_size)
    pos = torch.arange(4).unsqueeze(0)
    rot = rotary_embeddings(model, hidden, pos)
    assert rot is not None
    assert "position_embeddings" in block_kwargs(layers[0], hidden, pos, rot)


def test_gpt2_blocks_are_not_given_position_ids():
    """GPT-2 blocks take no position argument; passing one is wrong."""
    _tok, model = _load("sshleifer/tiny-gpt2")
    layers, _ = find_layer_list(model)
    hidden = torch.randn(1, 4, model.config.hidden_size)
    kwargs = block_kwargs(layers[0], hidden, torch.arange(4).unsqueeze(0), None)
    assert kwargs == {}


def test_unknown_architecture_is_refused_clearly():
    class NotATransformer:
        pass

    with pytest.raises(StageError, match="Supported layouts"):
        find_layer_list(NotATransformer())
