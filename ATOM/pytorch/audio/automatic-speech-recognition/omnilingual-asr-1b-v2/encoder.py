"""Static wav2vec2 encoder with explicit valid-frame masking for padded waveform buckets."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from checkpoint import Shape
from model_spec import ModelSpec
from torch import Tensor, nn


def fold_weight_norm(g: Tensor, v: Tensor) -> Tensor:
    """weight = g * v / ||v||, norm taken over every dim but the last.

    ``torch.nn.utils.weight_norm`` with dim=2 on this conv stores g shaped
    (1, 1, kernel), so the norm is per kernel position. Folded once here rather
    than carried into the graph as a runtime norm computation.
    """
    return g * v / v.norm(dim=(0, 1), keepdim=True)


class FeatureExtractor(nn.Module):
    """The 7 strided convs, each followed by a LayerNorm over channels and GELU."""

    def __init__(self, sd: dict[str, Tensor], prefix: str, spec: ModelSpec):
        super().__init__()
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for i, (kernel, stride) in enumerate(
            zip(spec.conv_kernels, spec.conv_strides, strict=True)
        ):
            w = sd[f"{prefix}layers.{i}.conv.weight"]
            conv = nn.Conv1d(w.shape[1], w.shape[0], kernel, stride=stride, bias=True)
            conv.weight.data.copy_(w)
            conv.bias.data.copy_(sd[f"{prefix}layers.{i}.conv.bias"])
            norm = nn.LayerNorm(w.shape[0])
            norm.weight.data.copy_(sd[f"{prefix}layers.{i}.layer_norm.weight"])
            norm.bias.data.copy_(sd[f"{prefix}layers.{i}.layer_norm.bias"])
            self.convs.append(conv)
            self.norms.append(norm)

    def forward(self, x: Tensor) -> Tensor:  # (B, S) -> (B, T, C)
        x = x.unsqueeze(1)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = conv(x)
            # The NPU GELU kernel requires the last dimension to be aligned.
            x = F.gelu(norm(x.transpose(1, 2))).transpose(1, 2)
        return x.transpose(1, 2)


class PerGroupConv1d(nn.Module):
    """Dense convolutions avoid unsupported grouped-convolution CPU kernels."""

    def __init__(self, weight: Tensor, bias: Tensor, groups: int, padding: int):
        super().__init__()
        channels, width, kernel = weight.shape
        if channels % groups or width * groups != channels:
            raise ValueError(
                "grouped conv weight must be (channels, channels / groups, kernel)"
            )
        self.width = width
        self.padding = (padding, padding - 1 if kernel % 2 == 0 else padding)
        self.convs = nn.ModuleList()
        for group in range(groups):
            conv = nn.Conv1d(width, width, kernel, padding=0, bias=True)
            conv.weight.data.copy_(weight[group * width : (group + 1) * width])
            conv.bias.data.copy_(bias[group * width : (group + 1) * width])
            self.convs.append(conv)

    def forward(self, x: Tensor) -> Tensor:  # (B, C, T) -> (B, C, T')
        w = self.width
        x = F.pad(x, self.padding)
        parts = [conv(x[:, g * w : (g + 1) * w]) for g, conv in enumerate(self.convs)]
        return torch.cat(parts, dim=1)


class Frontend(nn.Module):
    """Feature extractor, post-norm, projection to the encoder width, positional conv."""

    def __init__(self, sd: dict[str, Tensor], shape: Shape):
        super().__init__()
        p = "encoder_frontend."
        self.features = FeatureExtractor(sd, p + "feature_extractor.", shape.spec)
        self.post_norm = nn.LayerNorm(shape.feature_dim)
        self.post_norm.weight.data.copy_(sd[p + "post_extract_layer_norm.weight"])
        self.post_norm.bias.data.copy_(sd[p + "post_extract_layer_norm.bias"])
        self.proj = nn.Linear(shape.feature_dim, shape.enc_dim)
        self.proj.weight.data.copy_(sd[p + "model_dim_proj.weight"])
        self.proj.bias.data.copy_(sd[p + "model_dim_proj.bias"])

        weight = fold_weight_norm(
            sd[p + "pos_encoder.conv.weight_g"], sd[p + "pos_encoder.conv.weight_v"]
        )
        self.pos_conv = PerGroupConv1d(
            weight,
            sd[p + "pos_encoder.conv.bias"],
            shape.pos_groups,
            shape.pos_kernel // 2,
        )

    def forward(self, wav: Tensor, valid: Tensor) -> Tensor:
        x = self.proj(self.post_norm(self.features(wav)))
        x = x * valid.unsqueeze(-1)
        h = self.pos_conv(x.transpose(1, 2))
        return x + F.gelu(h.transpose(1, 2))


class EncoderLayer(nn.Module):
    """fairseq2 StandardTransformerEncoderLayer: pre-norm, biases on, GELU FFN."""

    def __init__(self, sd: dict[str, Tensor], prefix: str, dim: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads

        def linear(name: str) -> nn.Linear:
            w = sd[f"{prefix}{name}.weight"]
            m = nn.Linear(w.shape[1], w.shape[0], bias=True)
            m.weight.data.copy_(w)
            m.bias.data.copy_(sd[f"{prefix}{name}.bias"])
            return m

        def layer_norm(name: str) -> nn.LayerNorm:
            m = nn.LayerNorm(dim)
            m.weight.data.copy_(sd[f"{prefix}{name}.weight"])
            m.bias.data.copy_(sd[f"{prefix}{name}.bias"])
            return m

        self.q = linear("self_attn.q_proj")
        self.k = linear("self_attn.k_proj")
        self.v = linear("self_attn.v_proj")
        self.o = linear("self_attn.output_proj")
        self.attn_norm = layer_norm("self_attn_layer_norm")
        self.ffn_norm = layer_norm("ffn_layer_norm")
        self.ffn_in = linear("ffn.inner_proj")
        self.ffn_out = linear("ffn.output_proj")

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        b, t, _ = x.shape
        h = self.attn_norm(x)

        def split(m: nn.Linear) -> Tensor:
            return m(h).view(b, t, self.heads, self.head_dim).transpose(1, 2)

        a = F.scaled_dot_product_attention(
            split(self.q), split(self.k), split(self.v), attn_mask=valid
        )
        x = x + self.o(a.transpose(1, 2).reshape(b, t, -1))
        return x + self.ffn_out(F.gelu(self.ffn_in(self.ffn_norm(x))))


class Encoder(nn.Module):
    """Frontend, transformer encoder, final norm and decoder-width projection."""

    def __init__(self, sd: dict[str, Tensor], num_heads: int | None = None):
        super().__init__()
        shape = Shape(sd)
        heads = num_heads or shape.spec.encoder_heads
        if shape.enc_dim % heads:
            raise ValueError(
                f"num_heads={heads} does not divide encoder dim {shape.enc_dim}"
            )
        self.num_heads, self.model_dim = heads, shape.enc_dim
        self.frontend = Frontend(sd, shape)
        self.layers = nn.ModuleList(
            EncoderLayer(sd, f"encoder.layers.{i}.", shape.enc_dim, heads)
            for i in range(shape.enc_layers)
        )
        self.norm = nn.LayerNorm(shape.enc_dim)
        self.norm.weight.data.copy_(sd["encoder.layer_norm.weight"])
        self.norm.bias.data.copy_(sd["encoder.layer_norm.bias"])
        if shape.proj_dim is None:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(shape.enc_dim, shape.proj_dim)
            self.proj.weight.data.copy_(sd["encoder_proj.weight"])
            self.proj.bias.data.copy_(sd["encoder_proj.bias"])

    def forward(self, wav: Tensor, frame_mask: Tensor) -> Tensor:
        x = self.frontend(wav, frame_mask)
        attention_mask = frame_mask[:, None, None, :] > 0
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.proj(self.norm(x))
