"""Static frontend geometry and special token contract for omniASR_LLM_1B_v2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    sample_rate: int = 16_000
    #: The 7 strided convs of the wav2vec2 frontend; strides multiply to 320.
    conv_kernels: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_strides: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    #: 16 for every size; width and depth are what differ between them.
    encoder_heads: int = 16
    #: fairseq2's LLaMAConfig for this family.
    decoder_heads: int = 8
    #: ``text_frontend`` has this many rows more than ``final_proj``: the prompt
    #: addresses a special token the model never emits.
    n_special_tokens: int = 1
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 8192
    bos_token_id: int = 0
    pad_token_id: int = 1
    eos_token_id: int = 2

    def total_stride(self) -> int:
        product = 1
        for stride in self.conv_strides:
            product *= stride
        return product

    def frames_for_samples(self, num_samples: int) -> int:
        """Frames the frontend emits for ``num_samples`` at 16 kHz.

        Each conv is floor((n - kernel) / stride) + 1, applied in order. Not
        ``num_samples / 320``: the kernels shorten the sequence by a frame or two,
        and the valid frame count must match the compiled encoder signature.
        """
        n = int(num_samples)
        for kernel, stride in zip(self.conv_kernels, self.conv_strides, strict=True):
            n = (n - kernel) // stride + 1
        return n

    def frames_for_seconds(self, seconds: float) -> int:
        return self.frames_for_samples(round(seconds * self.sample_rate))
