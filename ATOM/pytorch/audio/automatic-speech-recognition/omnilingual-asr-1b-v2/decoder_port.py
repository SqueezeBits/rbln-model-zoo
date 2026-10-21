"""Convert interleaved fairseq2 rotary weights to the Hugging Face LLaMA convention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from checkpoint import Shape
from torch import Tensor


def permute_rotary(w: Tensor, heads: int, head_dim: int) -> Tensor:
    """Interleaved rotary rows -> ``rotate_half`` rows: a pure row permutation of q/k."""
    out = w.view(heads, head_dim // 2, 2, -1)
    return out.transpose(1, 2).reshape(w.shape)


def convert_decoder(sd: dict[str, Tensor], shape: Shape) -> dict[str, Tensor]:
    """HF state dict for the decoder, ``lm_head`` padded to ``text_frontend``'s height."""
    heads, head_dim = shape.dec_heads, shape.dec_head_dim
    out: dict[str, Tensor] = {}
    for i in range(shape.dec_layers):
        s, d = f"llama_decoder.layers.{i}.", f"model.layers.{i}."
        out[d + "self_attn.q_proj.weight"] = permute_rotary(
            sd[s + "self_attn.q_proj.weight"], heads, head_dim
        )
        out[d + "self_attn.k_proj.weight"] = permute_rotary(
            sd[s + "self_attn.k_proj.weight"], heads, head_dim
        )
        out[d + "self_attn.v_proj.weight"] = sd[s + "self_attn.v_proj.weight"]
        out[d + "self_attn.o_proj.weight"] = sd[s + "self_attn.output_proj.weight"]
        # fairseq2 GLU: gate_proj is the gated half, inner_proj the value half.
        out[d + "mlp.gate_proj.weight"] = sd[s + "ffn.gate_proj.weight"]
        out[d + "mlp.up_proj.weight"] = sd[s + "ffn.inner_proj.weight"]
        out[d + "mlp.down_proj.weight"] = sd[s + "ffn.output_proj.weight"]
        out[d + "input_layernorm.weight"] = sd[s + "self_attn_layer_norm.weight"]
        out[d + "post_attention_layernorm.weight"] = sd[s + "ffn_layer_norm.weight"]
    out["model.norm.weight"] = sd["llama_decoder.layer_norm.weight"]
    out["model.embed_tokens.weight"] = sd["text_frontend.weight"]
    head = sd["final_proj.weight"]
    if head.shape[0] < shape.text_rows:
        pad = torch.zeros(
            shape.text_rows - head.shape[0], head.shape[1], dtype=head.dtype
        )
        head = torch.cat([head, pad], dim=0)
    out["lm_head.weight"] = head
    return out


def hf_config(shape: Shape):
    from transformers import LlamaConfig

    spec = shape.spec
    return LlamaConfig(
        vocab_size=shape.text_rows,  # embedding height, not the logit width
        hidden_size=shape.dec_dim,
        intermediate_size=shape.dec_ffn,
        num_hidden_layers=shape.dec_layers,
        num_attention_heads=shape.dec_heads,
        num_key_value_heads=shape.dec_heads,
        head_dim=shape.dec_head_dim,
        max_position_embeddings=spec.max_position_embeddings,
        rope_theta=spec.rope_theta,
        rms_norm_eps=spec.rms_norm_eps,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        pad_token_id=spec.pad_token_id,
        bos_token_id=spec.bos_token_id,
        eos_token_id=spec.eos_token_id,
    )


@dataclass(frozen=True)
class RotaryCheckResult:
    pure_permutation: bool
    relative_error: float
    control_error: float

    def ok(self) -> bool:
        return (
            self.pure_permutation
            and self.relative_error < 1e-12
            and self.control_error > 1e-3
        )

    def lines(self) -> list[str]:
        mark = lambda good: "OK " if good else "FAIL"
        return [
            f"{mark(self.pure_permutation)} permutation is a pure row permutation",
            (
                f"{mark(self.relative_error < 1e-12)} rotate_half(permuted) . k == "
                f"interleaved(original) . k   rel {self.relative_error:.3e}"
            ),
            (
                f"{mark(self.control_error > 1e-3)} CONTROL: without the permutation q . k differs   "
                f"rel {self.control_error:.3e}  (must be large)"
            ),
        ]


class RotaryConventionCheck:
    """Prove :func:`permute_rotary` on the decoder geometry, in float64.

    What is compared matters. Rotating the permuted weights with ``rotate_half``
    does NOT reproduce fairseq2's ``q`` element for element: HF's output index
    ``i`` holds what fairseq2 puts at ``2i``, so the two ``q`` vectors are
    permutations of each other. The quantity that must agree is the one
    attention consumes, ``q . k``, which is permutation-invariant because ``q``
    and ``k`` carry the same permutation. The control (no permutation) must
    differ, or the test would pass just as well on a no-op.
    """

    def __init__(self, shape: Shape, positions: int = 4, seed: int = 0):
        shape.require_decoder()
        self._shape, self._positions, self._seed = shape, positions, seed

    def run(self) -> RotaryCheckResult:
        torch.manual_seed(self._seed)
        heads, hd, dim, t = (
            self._shape.dec_heads,
            self._shape.dec_head_dim,
            self._shape.dec_dim,
            self._positions,
        )
        wq = torch.randn(heads * hd, dim, dtype=torch.float64)
        wk = torch.randn(heads * hd, dim, dtype=torch.float64)
        x = torch.randn(1, t, dim, dtype=torch.float64)
        pos = torch.arange(t, dtype=torch.float64)
        inv = self._shape.spec.rope_theta ** (
            -torch.arange(0, hd, 2, dtype=torch.float64) / hd
        )
        ang = pos[:, None] * inv[None, :]  # (T, hd/2)

        def interleaved(weight: Tensor) -> Tensor:
            q = F.linear(x, weight).view(1, t, heads, hd // 2, 2)
            rot = torch.polar(torch.ones_like(ang), ang)[None, :, None, :]
            return (
                torch.view_as_real(torch.view_as_complex(q.contiguous()) * rot)
                .reshape(1, t, heads, hd)
                .transpose(1, 2)
            )

        def rotate_half(weight: Tensor) -> Tensor:
            q = F.linear(x, weight).view(1, t, heads, hd).transpose(1, 2)
            cos = torch.cat([ang.cos(), ang.cos()], dim=-1)[None, None]
            sin = torch.cat([ang.sin(), ang.sin()], dim=-1)[None, None]
            half = torch.cat([-q[..., hd // 2 :], q[..., : hd // 2]], dim=-1)
            return q * cos + half * sin

        def scores(q: Tensor, k: Tensor) -> Tensor:
            return q @ k.transpose(-1, -2)

        pq, pk = permute_rotary(wq, heads, hd), permute_rotary(wk, heads, hd)
        pure = torch.equal(wq.sum(dim=1).sort().values, pq.sum(dim=1).sort().values)
        reference = scores(interleaved(wq), interleaved(wk))
        scale = reference.abs().max().item()
        rel = (
            scores(rotate_half(pq), rotate_half(pk)) - reference
        ).abs().max().item() / scale
        control = (
            scores(rotate_half(wq), rotate_half(wk)) - reference
        ).abs().max().item() / scale
        return RotaryCheckResult(pure, rel, control)
