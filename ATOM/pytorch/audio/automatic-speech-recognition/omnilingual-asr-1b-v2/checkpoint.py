"""Read the public fairseq2 checkpoint and validate its tensor geometry."""

from __future__ import annotations

import pathlib

import torch
from model_spec import ModelSpec
from torch import Tensor


def load_state_dict(path: pathlib.Path) -> dict[str, Tensor]:
    """The checkpoint is ``{'model': state_dict, 'fs2': ...}``; mmap keeps RSS sane."""
    blob = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return blob.get("model", blob)


class Shape:
    """Geometry inferred from checkpoint tensors, checked before conversion."""

    def __init__(self, sd: dict[str, Tensor], spec: ModelSpec | None = None):
        self.spec = spec or ModelSpec()
        self.enc_layers = 1 + max(
            int(k.split(".")[2]) for k in sd if k.startswith("encoder.layers.")
        )
        self.enc_dim = sd["encoder.layers.0.self_attn.q_proj.weight"].shape[1]
        self.enc_ffn = sd["encoder.layers.0.ffn.inner_proj.weight"].shape[0]
        self.feature_dim = sd["encoder_frontend.model_dim_proj.weight"].shape[1]
        # LLM variants project the encoder onto the decoder width; CTC has no such layer.
        self.proj_dim = (
            sd["encoder_proj.weight"].shape[0] if "encoder_proj.weight" in sd else None
        )
        pos_v = sd["encoder_frontend.pos_encoder.conv.weight_v"]
        self.pos_kernel = pos_v.shape[2]
        self.pos_groups = pos_v.shape[0] // pos_v.shape[1]

        self.has_decoder = any(k.startswith("llama_decoder.layers.") for k in sd)
        self.ctc_vocab = (
            sd["final_proj.weight"].shape[0]
            if not self.has_decoder and "final_proj.weight" in sd
            else None
        )
        if self.has_decoder:
            self.dec_layers = 1 + max(
                int(k.split(".")[2])
                for k in sd
                if k.startswith("llama_decoder.layers.")
            )
            self.dec_dim = sd["llama_decoder.layers.0.self_attn.q_proj.weight"].shape[1]
            self.dec_ffn = sd["llama_decoder.layers.0.ffn.inner_proj.weight"].shape[0]
            self.dec_heads = self.spec.decoder_heads
            self.text_rows = sd["text_frontend.weight"].shape[0]
            self.logit_rows = sd["final_proj.weight"].shape[0]
            self.n_langs = sd["lang_embeddings.weight"].shape[0]
        else:
            self.dec_layers = self.dec_dim = self.dec_ffn = None
            self.dec_heads = self.text_rows = self.logit_rows = self.n_langs = None

    @property
    def dec_head_dim(self) -> int | None:
        return None if self.dec_dim is None else self.dec_dim // self.dec_heads

    def check(self) -> None:
        """Assert the invariants the conversion depends on, for what is present."""
        if self.enc_dim % self.spec.encoder_heads:
            raise ValueError(
                f"encoder dim {self.enc_dim} not divisible by {self.spec.encoder_heads} heads"
            )
        if not self.has_decoder:
            return
        if self.text_rows != self.logit_rows + self.spec.n_special_tokens:
            raise ValueError(
                f"text_frontend {self.text_rows} should be final_proj {self.logit_rows} "
                f"+ {self.spec.n_special_tokens}"
            )
        if self.dec_dim % self.dec_heads:
            raise ValueError(
                f"decoder dim {self.dec_dim} not divisible by {self.dec_heads} heads"
            )
        if self.proj_dim != self.dec_dim:
            raise ValueError(
                f"encoder_proj must land on the decoder width: {self.proj_dim} vs {self.dec_dim}"
            )

    def require_decoder(self) -> None:
        if not self.has_decoder:
            raise ValueError(
                "this state dict has no llama_decoder.* tensors; pass the full checkpoint, "
                "not encoder/weights.pth"
            )

    def describe(self) -> str:
        enc = (
            f"encoder {self.enc_layers}L dim {self.enc_dim} ffn {self.enc_ffn} "
            f"head_dim {self.enc_dim // self.spec.encoder_heads}"
        )
        if self.ctc_vocab is not None:
            return enc + f" | CTC head, vocab {self.ctc_vocab}"
        if not self.has_decoder:
            return enc + " | decoder absent (encoder-only state dict)"
        return (
            f"{enc} | decoder {self.dec_layers}L dim {self.dec_dim} ffn {self.dec_ffn} "
            f"head_dim {self.dec_head_dim} | vocab {self.logit_rows} "
            f"(+{self.spec.n_special_tokens}) | langs {self.n_langs}"
        )
