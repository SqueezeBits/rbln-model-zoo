"""Verify every compiled bucket, padding invariance and decoder numerical conversion."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from artifact import digest, load_manifest, make_prompt
from checkpoint import Shape, load_state_dict
from decoder_port import RotaryConventionCheck, convert_decoder, hf_config
from encoder import Encoder
from inference import Transcriber, bucket_inputs, prepare_audio


def metrics(actual, expected):
    if actual.numel() == 0 or expected.numel() == 0:
        raise ValueError("numerical comparison requires nonempty tensors")
    if not actual.isfinite().all() or not expected.isfinite().all():
        raise ValueError("numerical comparison rejected nonfinite tensors")
    actual, expected = actual.double(), expected.double()
    error = (actual - expected).abs()
    return {
        "max_abs": error.max().item(),
        "mean_abs": error.mean().item(),
        "relative_l2": (
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(expected).clamp_min(1e-12)
        ).item(),
        "cosine": F.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item(),
    }


def native_decoder(sd, shape, inputs):
    """Independent CPU reference using original interleaved rotary weights."""
    x = inputs
    t, heads, hd = x.shape[1], shape.dec_heads, shape.dec_head_dim
    positions = torch.arange(t, dtype=torch.float32)
    inv = shape.spec.rope_theta ** (-torch.arange(0, hd, 2, dtype=torch.float32) / hd)
    angles = positions[:, None] * inv[None, :]
    rotation = torch.polar(torch.ones_like(angles), angles)[None, :, None, :]

    def norm(value, weight):
        return (
            value
            * torch.rsqrt(
                value.square().mean(-1, keepdim=True) + shape.spec.rms_norm_eps
            )
            * weight
        )

    def rotate(value):
        pairs = value.reshape(1, t, heads, hd // 2, 2)
        return (
            torch.view_as_real(torch.view_as_complex(pairs.contiguous()) * rotation)
            .reshape(1, t, heads, hd)
            .transpose(1, 2)
        )

    for i in range(shape.dec_layers):
        prefix = f"llama_decoder.layers.{i}."
        h = norm(x, sd[prefix + "self_attn_layer_norm.weight"])
        q, k = [
            rotate(F.linear(h, sd[prefix + f"self_attn.{n}_proj.weight"]))
            for n in ("q", "k")
        ]
        v = (
            F.linear(h, sd[prefix + "self_attn.v_proj.weight"])
            .reshape(1, t, heads, hd)
            .transpose(1, 2)
        )
        h = (
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
            .transpose(1, 2)
            .reshape(1, t, -1)
        )
        x = x + F.linear(h, sd[prefix + "self_attn.output_proj.weight"])
        h = norm(x, sd[prefix + "ffn_layer_norm.weight"])
        gated = F.silu(F.linear(h, sd[prefix + "ffn.gate_proj.weight"]))
        h = gated * F.linear(h, sd[prefix + "ffn.inner_proj.weight"])
        x = x + F.linear(h, sd[prefix + "ffn.output_proj.weight"])
    return F.linear(
        norm(x, sd["llama_decoder.layer_norm.weight"])[:, -1], sd["final_proj.weight"]
    )


def main():
    from transformers import LlamaForCausalLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--language", default="eng_Latn")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    manifest = load_manifest(args.model_dir)
    if digest(args.checkpoint) != manifest["sources"][0]["sha256"]:
        raise ValueError("reference checkpoint differs from the artifact source")
    sd = load_state_dict(args.checkpoint)
    shape = Shape(sd)
    rotary = RotaryConventionCheck(shape).run()
    assert rotary.ok()
    cpu_encoder = Encoder(sd).eval()
    waveform = prepare_audio(args.audio)
    npu = Transcriber(
        args.model_dir, args.device, waveform=waveform, language_code=args.language
    )
    # Use the real utterance for every fitting bucket; shorter buckets get a real prefix.
    failures = []
    report = {
        "versions": manifest["versions"],
        "encoder": [],
        "audio_sha256": digest(args.audio),
    }
    with torch.inference_mode():
        for bucket in manifest["encoder_buckets"]:
            clip = waveform[
                :, : min(waveform.shape[1], bucket["samples"] - 1600)
            ].contiguous()
            padded, mask, valid = bucket_inputs(clip, bucket)
            cpu_short = cpu_encoder(clip, torch.ones(1, valid))
            cpu_padded = cpu_encoder(padded, mask)[:, :valid]
            padding = metrics(cpu_padded, cpu_short)
            if padding["relative_l2"] > 1e-4:
                failures.append(
                    f"bucket {bucket['seconds']}s CPU padding parity failed: {padding}"
                )
            encoded, _ = npu.encode(clip, bucket["seconds"])
            parity = metrics(encoded, cpu_short)
            if parity["relative_l2"] > 0.02 or parity["cosine"] < 0.999:
                failures.append(
                    f"bucket {bucket['seconds']}s NPU encoder parity failed: {parity}"
                )
            record = {
                "bucket_seconds": bucket["seconds"],
                "valid_samples": clip.shape[1],
                "valid_frames": valid,
                "cpu_padding": padding,
                "cpu_vs_npu": parity,
            }
            report["encoder"].append(record)
            print(json.dumps(record), flush=True)
        audio, _ = npu.encode(waveform)
        prompt = make_prompt(
            audio, npu.text, npu.language, npu.lookup, manifest, args.language
        )
        config = hf_config(shape)
        # CPU construction also initializes non-persistent rotary buffers.
        cpu_decoder = LlamaForCausalLM(config)
        cpu_decoder.load_state_dict(
            convert_decoder(sd, shape), strict=True, assign=True
        )
        cpu_decoder.eval()
        hf_logits = cpu_decoder(inputs_embeds=prompt, use_cache=False).logits[
            :, -1, : shape.logit_rows
        ]
        reference = native_decoder(sd, shape, prompt)
        conversion = metrics(hf_logits, reference)
        if conversion["relative_l2"] > 1e-4:
            failures.append(f"fairseq2-to-HF decoder conversion failed: {conversion}")
        output = npu.decoder(
            inputs_embeds=prompt,
            attention_mask=torch.ones(prompt.shape[:2], dtype=torch.int64),
            return_dict=True,
        )
        logits = output.logits[:, -1, : shape.logit_rows].clone()
        parity = metrics(logits, reference)
        if (
            parity["relative_l2"] > 0.02
            or logits.argmax().item() != reference.argmax().item()
        ):
            failures.append(f"NPU decoder parity failed: {parity}")
        teacher_steps = []
        forced = []
        cpu_logits = reference
        for step in range(3):
            token = int(cpu_logits.argmax())
            forced.append(token)
            position = output.generate_idx
            output = npu.decoder(
                input_ids=torch.tensor([[token]], dtype=torch.int64),
                cache_position=position + output.padded_cache_lengths,
                position_ids=position,
                generate_idx=position + 1,
                padded_cache_lengths=output.padded_cache_lengths,
                return_dict=True,
            )
            extended = torch.cat((prompt, npu.text[forced].unsqueeze(0)), dim=1)
            cpu_logits = cpu_decoder(inputs_embeds=extended, use_cache=False).logits[
                :, -1, : shape.logit_rows
            ]
            step_logits = output.logits[:, -1, : shape.logit_rows]
            values = metrics(step_logits, cpu_logits)
            if (
                values["relative_l2"] > 0.02
                or step_logits.argmax().item() != cpu_logits.argmax().item()
            ):
                failures.append(f"autoregressive step {step} parity failed: {values}")
            teacher_steps.append({"step": step + 1, "forced_token": token, **values})
        repeated = npu.decoder(
            inputs_embeds=prompt,
            attention_mask=torch.ones(prompt.shape[:2], dtype=torch.int64),
            return_dict=True,
        )
        reset = metrics(repeated.logits[:, -1, : shape.logit_rows], logits)
        if reset["max_abs"] > 1e-6:
            failures.append(
                f"prefill after a prior request does not reset the cache: {reset}"
            )
        result, _, _ = npu.transcribe(waveform, args.language)
        report.update(
            {
                "decoder_conversion": conversion,
                "decoder_cpu_vs_npu": parity,
                "teacher_forced_decode": teacher_steps,
                "repeated_prefill": reset,
                "greedy_transcription": result,
                "reference_first_token": reference.argmax().item(),
                "rotary_relative_error": rotary.relative_error,
            }
        )
    report.update({"passed": not failures, "failures": failures})
    if args.output_json:
        args.output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
