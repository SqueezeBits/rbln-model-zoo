"""Reload the complete artifact and greedily transcribe audio on an RBLN-CA25."""

import argparse
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from artifact import check_toolchain, load_manifest, load_tables, make_prompt, versions
from model_spec import ModelSpec
from scipy.signal import resample_poly


class HostTrace:
    """Optional Chrome/Perfetto host spans; device activity is not inferred from these."""

    def __init__(self, path=None):
        self.path = path
        self.events = []
        self.origin = time.perf_counter_ns()

    @contextmanager
    def span(self, name):
        if self.path is None:
            yield
            return
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            if self.path:
                self.events.append(
                    {
                        "name": name,
                        "ph": "X",
                        "pid": 1,
                        "tid": 1,
                        "ts": (started - self.origin) / 1000,
                        "dur": (time.perf_counter_ns() - started) / 1000,
                    }
                )

    def save(self):
        if self.path:
            Path(self.path).write_text(json.dumps({"traceEvents": self.events}))


def prepare_audio(path):
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // divisor, rate // divisor).astype(
            np.float32
        )
    waveform = torch.from_numpy(np.ascontiguousarray(samples))
    if (
        waveform.numel() < 400
        or waveform.numel() > 640000
        or not waveform.isfinite().all()
    ):
        raise ValueError(
            "audio must be finite and between 25 milliseconds and 40 seconds"
        )
    # Normalize the real waveform before right-padding it into the static bucket.
    return F.layer_norm(waveform, waveform.shape).view(1, -1)


def bucket_inputs(waveform, bucket):
    spec = ModelSpec()
    valid = spec.frames_for_samples(waveform.shape[1])
    if waveform.shape[1] > bucket["samples"] or valid < 1:
        raise ValueError("audio does not fit the selected bucket")
    padded = F.pad(waveform, (0, bucket["samples"] - waveform.shape[1]))
    mask = torch.zeros(1, bucket["frames"])
    mask[:, :valid] = 1
    return padded.contiguous(), mask, valid


class Transcriber:
    def __init__(self, root, device=0, trace=None, waveform=None, language_code=None):
        if device < 0:
            raise ValueError("device must be nonnegative")
        check_toolchain()
        import rebel
        import sentencepiece as spm
        from optimum.rbln import RBLNLlamaForCausalLM

        self.root = Path(root)
        self.manifest = load_manifest(self.root)
        current = versions()
        if any(
            current[name] != self.manifest["versions"][name]
            for name in ("torch", "transformers", "rebel-compiler")
        ):
            raise RuntimeError("runtime toolchain differs from the compiled artifact")
        self.text, self.language, self.lookup = load_tables(self.root, self.manifest)
        self.validate_language(language_code)
        if waveform is not None:
            self.select_bucket(waveform)
        self.tokenizer = spm.SentencePieceProcessor(
            model_file=str(self.root / "tokenizer.model")
        )
        if self.tokenizer.vocab_size() != self.manifest["real_logit_width"]:
            raise ValueError("tokenizer vocabulary differs from decoder logits")
        self.device, self.trace = device, trace or HostTrace()
        # Compiled .rbln binaries are loaded directly. There is no encoder CPU implementation here.
        self.encoder_models, self.encoders = {}, {}
        for bucket in self.manifest["encoder_buckets"]:
            graph = rebel.RBLNCompiledModel(self.root / bucket["file"])
            self.encoder_models[bucket["seconds"]] = graph
            self.encoders[bucket["seconds"]] = graph.create_runtime(
                device=device, tensor_type="pt"
            )
        self.decoder = RBLNLlamaForCausalLM.from_pretrained(
            self.root / "decoder", export=False, rbln_device=device
        )

    def validate_language(self, language):
        if language is not None and language.lower() not in self.lookup:
            raise ValueError(f"unsupported language code: {language}")

    def select_bucket(self, waveform, bucket_seconds=None):
        buckets = self.manifest["encoder_buckets"]
        bucket = next(
            (
                b
                for b in buckets
                if waveform.shape[1] <= b["samples"]
                and (bucket_seconds is None or b["seconds"] == bucket_seconds)
            ),
            None,
        )
        if bucket is None:
            raise ValueError("audio length has no compiled encoder bucket")
        return bucket

    def encode(self, waveform, bucket_seconds=None):
        bucket = self.select_bucket(waveform, bucket_seconds)
        padded, mask, valid = bucket_inputs(waveform, bucket)
        with self.trace.span(f"encoder_{bucket['seconds']}s"):
            encoded = self.encoders[bucket["seconds"]](padded, mask)
        if isinstance(encoded, (list, tuple)):
            encoded = encoded[0]
        if list(encoded.shape) != [1, bucket["frames"], self.manifest["decoder_width"]]:
            raise ValueError("compiled encoder output ABI mismatch")
        return encoded[:, :valid].contiguous(), bucket

    def transcribe(self, waveform, language="eng_Latn", max_new_tokens=256):
        self.validate_language(language)
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        audio, bucket = self.encode(waveform)
        prompt = make_prompt(
            audio, self.text, self.language, self.lookup, self.manifest, language
        )
        budget = min(max_new_tokens, self.manifest["max_seq_len"] - prompt.shape[1])
        if budget < 1:
            raise ValueError("prompt leaves no generation budget")
        with self.trace.span("decoder_prefill"):
            output = self.decoder(
                inputs_embeds=prompt,
                attention_mask=torch.ones(prompt.shape[:2], dtype=torch.int64),
                return_dict=True,
            )
        first_logits = output.logits[:, -1, : self.manifest["real_logit_width"]].clone()
        ids = []
        for _ in range(budget):
            token = int(
                output.logits[0, -1, : self.manifest["real_logit_width"]].argmax()
            )
            if token == self.manifest["eos_token_id"]:
                break
            ids.append(token)
            if len(ids) == budget:
                break
            position = output.generate_idx
            with self.trace.span("decoder_step"):
                output = self.decoder(
                    input_ids=torch.tensor([[token]], dtype=torch.int64),
                    cache_position=position + output.padded_cache_lengths,
                    position_ids=position,
                    generate_idx=position + 1,
                    padded_cache_lengths=output.padded_cache_lengths,
                    return_dict=True,
                )
        text = self.tokenizer.decode(ids)
        return (
            {
                "text": text,
                "token_ids": ids,
                "bucket_seconds": bucket["seconds"],
                "audio_frames": audio.shape[1],
                "prompt_tokens": prompt.shape[1],
                "stopped_on_eos": token == self.manifest["eos_token_id"],
            },
            first_logits,
            prompt,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("omniASR_LLM_1B_v2-rbln")
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument(
        "--language",
        default="eng_Latn",
        help="ISO-script code, or 'none' for unconditional",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--host-trace",
        type=Path,
        help="separate instrumented run; timings are not benchmark results",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if min(args.max_new_tokens, args.repeat, args.threads) < 1 or args.warmup < 0:
        raise ValueError(
            "token, repeat and thread counts must be positive; warmup must be nonnegative"
        )
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    trace = HostTrace(args.host_trace)
    waveform = prepare_audio(args.audio)
    language = None if args.language.lower() == "none" else args.language
    model = Transcriber(args.model_dir, args.device, trace, waveform, language)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model.transcribe(waveform, language, args.max_new_tokens)
        prepared_durations, prepare_durations, e2e_durations = [], [], []
        for _ in range(args.repeat):
            started = time.perf_counter()
            with trace.span("request_e2e"):
                with trace.span("cpu_prepare"):
                    waveform = prepare_audio(args.audio)
                prepared_at = time.perf_counter()
                result, _, _ = model.transcribe(waveform, language, args.max_new_tokens)
            finished = time.perf_counter()
            prepare_durations.append(prepared_at - started)
            prepared_durations.append(finished - prepared_at)
            e2e_durations.append(finished - started)
    trace.save()
    result.update(
        {
            "runtime": "RBLN-CA25 encoder and decoder",
            "device": args.device,
            "audio_seconds": waveform.shape[1] / 16000,
            "host_trace": str(args.host_trace) if args.host_trace else None,
        }
    )
    if not args.host_trace:
        result["timing_definition"] = {
            "cpu_prepare": "local audio file read/decode, mono conversion, resampling and normalization",
            "prepared_inference": "encoder + greedy decoder with prepared waveform",
            "e2e": "cpu_prepare + prepared_inference; excludes artifact load, integrity checks and compilation",
        }
        result["timings"] = {
            name: {
                "samples_seconds": values,
                "median_seconds": float(np.median(values)),
                "p95_seconds": float(np.percentile(values, 95)),
            }
            for name, values in (
                ("cpu_prepare", prepare_durations),
                ("prepared_inference", prepared_durations),
                ("e2e", e2e_durations),
            )
        }
    if args.output_json:
        args.output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
