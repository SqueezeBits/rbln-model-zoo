"""Compile the public 1B-v2 checkpoint into independent encoder and decoder artifacts."""

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch
from artifact import (
    CHECKPOINT_URL,
    LANGUAGE_URL,
    MODEL_NAME,
    OPTIMUM_REVISION,
    SOURCE_HASHES,
    TOKENIZER_URL,
    UPSTREAM_REVISION,
    assert_npu_graph,
    check_toolchain,
    digest,
    versions,
    write_json,
)
from checkpoint import Shape, load_state_dict
from decoder_port import RotaryConventionCheck, convert_decoder, hf_config
from encoder import Encoder
from model_spec import ModelSpec


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--lang-table", type=Path)
    parser.add_argument("--download-dir", type=Path, default=Path(".cache/omniasr"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("omniASR_LLM_1B_v2-rbln")
    )
    parser.add_argument("--audio-buckets-s", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--stage", choices=["encoder", "decoder", "finalize"], help=argparse.SUPPRESS
    )
    parser.add_argument("--bucket", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def fetch(path, default_name, url, download_dir):
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    download_dir.mkdir(parents=True, exist_ok=True)
    path = download_dir / default_name
    if not path.is_file():
        temporary = path.with_suffix(path.suffix + ".partial")
        print(f"Downloading {url}", flush=True)
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)
    return path


def compile_encoder(args):
    import rebel

    spec = ModelSpec()
    samples = args.bucket * spec.sample_rate
    frames = spec.frames_for_samples(samples)
    model = Encoder(load_state_dict(args.checkpoint)).eval()
    graph = rebel.compile_from_torch(
        model,
        input_info=[
            ("waveform", [1, samples], "float32"),
            ("frame_mask", [1, frames], "float32"),
        ],
        npu="RBLN-CA25",
        model_trace_method="export",
    )
    path = args.output_dir / "encoder" / f"{args.bucket}s.rbln"
    path.parent.mkdir(parents=True, exist_ok=True)
    graph.save(path)
    metadata = rebel.RBLNCompiledModel.inspect(path)
    assert_npu_graph(metadata)
    write_json(path.with_suffix(".json"), metadata)
    # Execution is part of compilation verification; inference.py reloads this file independently.
    runtime = graph.create_runtime(device=args.device, tensor_type="pt")
    with torch.inference_mode():
        result = runtime(torch.zeros(1, samples), torch.ones(1, frames))
    if isinstance(result, (list, tuple)):
        result = result[0]
    if (
        list(result.shape) != [1, frames, model.proj.out_features]
        or not result.isfinite().all()
    ):
        raise RuntimeError("invalid compiled encoder output")
    print(f"NPU encoder {args.bucket}s executed: {list(result.shape)}", flush=True)


def compile_decoder(args):
    from optimum.rbln import RBLNLlamaForCausalLM
    from transformers import LlamaForCausalLM

    sd = load_state_dict(args.checkpoint)
    shape = Shape(sd)
    shape.check()
    check = RotaryConventionCheck(shape).run()
    print("\n".join(check.lines()), flush=True)
    if not check.ok():
        raise RuntimeError("rotary convention validation failed")
    config = hf_config(shape)
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    model.load_state_dict(convert_decoder(sd, shape), strict=True, assign=True)
    model.eval()
    compiled = RBLNLlamaForCausalLM.from_model(
        model,
        export=True,
        config=config,
        rbln_config={
            "batch_size": 1,
            "max_seq_len": args.max_seq_len,
            "use_inputs_embeds": True,
            "attn_impl": "eager",
            "use_attention_mask": True,
            "kvcache_block_size": args.max_seq_len,
            "prefill_chunk_size": args.prefill_chunk_size,
            "decoder_batch_sizes": [1],
            "device": args.device,
            "npu": "RBLN-CA25",
        },
    )
    compiled.save_pretrained(args.output_dir / "decoder")
    print("Decoder compiled and saved", flush=True)


def finalize(args):
    import pyarrow.parquet as pq
    import rebel
    import sentencepiece as spm

    sd = load_state_dict(args.checkpoint)
    shape = Shape(sd)
    shape.check()
    shape.require_decoder()
    # Fail for another family/size instead of labelling a different checkpoint 1B-v2.
    if (shape.enc_layers, shape.enc_dim, shape.dec_layers, shape.dec_dim) != (
        48,
        1280,
        12,
        4096,
    ):
        raise ValueError(f"unsupported checkpoint geometry: {shape.describe()}")
    root = args.output_dir
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    if (
        tokenizer.vocab_size(),
        tokenizer.bos_id(),
        tokenizer.pad_id(),
        tokenizer.eos_id(),
    ) != (shape.logit_rows, 0, 1, 2):
        raise ValueError(
            "v2 tokenizer does not match checkpoint vocabulary/special tokens"
        )
    rows = pq.read_table(args.lang_table).to_pylist()
    # The upstream factory offsets the parquet's zero-based index by one.
    lookup = {row["lang"].lower(): row["index"] + 1 for row in rows}
    if len(lookup) != shape.n_langs - 1 or set(lookup.values()) != set(
        range(1, shape.n_langs)
    ):
        raise ValueError("language table does not cover the checkpoint embedding rows")
    write_json(root / "lang_lookup.json", lookup)
    shutil.copyfile(args.tokenizer, root / "tokenizer.model")
    torch.save(
        {
            "text": sd["text_frontend.weight"].clone(),
            "language": sd["lang_embeddings.weight"].clone(),
        },
        root / "prompt_embeddings.pt",
    )
    buckets = []
    for seconds in sorted(args.audio_buckets_s):
        relative = f"encoder/{seconds}s.rbln"
        metadata = rebel.RBLNCompiledModel.inspect(root / relative)
        assert_npu_graph(metadata)
        buckets.append(
            {
                "seconds": seconds,
                "samples": seconds * 16000,
                "frames": shape.spec.frames_for_seconds(seconds),
                "file": relative,
                "metadata": metadata,
            }
        )
    # Every generated decoder file, tokenizer, table and encoder binary is required at reload.
    required = [
        root / "decoder" / n
        for n in (
            "config.json",
            "rbln_config.json",
            "prefill.rbln",
            "decoder_batch_1.rbln",
        )
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"incomplete decoder artifact: {path.name}")
    files = {
        str(p.relative_to(root)): {"sha256": digest(p), "size": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "compile.json"
    }
    source = [
        {
            "name": path.name,
            "url": url,
            "sha256": digest(path),
            "size": path.stat().st_size,
        }
        for path, url in (
            (args.checkpoint, CHECKPOINT_URL),
            (args.tokenizer, TOKENIZER_URL),
            (args.lang_table, LANGUAGE_URL),
        )
    ]
    write_json(
        root / "compile.json",
        {
            "schema_version": 1,
            "model": MODEL_NAME,
            "sources": source,
            "upstream_revision": UPSTREAM_REVISION,
            "optimum_revision": OPTIMUM_REVISION,
            "optimum_source": json.loads(
                importlib.metadata.distribution("optimum-rbln").read_text(
                    "direct_url.json"
                )
            ),
            "recipe_sha256": {
                p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "versions": versions(),
            "target_npu": "RBLN-CA25",
            "num_devices": 1,
            "encoder_buckets": buckets,
            "encoder_key_mask": "boolean",
            "decoder_batch_sizes": [1],
            "decoder_metadata": {
                name: rebel.RBLNCompiledModel.inspect(root / "decoder" / name)
                for name in ("prefill.rbln", "decoder_batch_1.rbln")
            },
            "max_seq_len": args.max_seq_len,
            "prefill_chunk_size": args.prefill_chunk_size,
            "quantization": None,
            "geometry": shape.describe(),
            "decoder_width": shape.dec_dim,
            "embedding_rows": shape.text_rows,
            "real_logit_width": shape.logit_rows,
            "language_rows": shape.n_langs,
            "bos_token_id": 0,
            "pad_token_id": 1,
            "eos_token_id": 2,
            "prompt_layout": "[valid audio frames][lid marker][language embedding][bos]",
            "language_row_zero": "unconditional",
            "files": files,
        },
    )


def main():
    args = parse_args()
    check_toolchain()
    if args.threads < 1 or args.prefill_chunk_size < 1:
        raise ValueError("threads and prefill chunk size must be positive")
    spec = ModelSpec()
    if not args.audio_buckets_s or len(set(args.audio_buckets_s)) != len(
        args.audio_buckets_s
    ):
        raise ValueError("audio buckets must be nonempty and unique")
    if any(b < 1 or b > 40 for b in args.audio_buckets_s):
        raise ValueError("static audio buckets must be 1..40 seconds")
    if args.max_seq_len % args.prefill_chunk_size or args.max_seq_len > 8192:
        raise ValueError(
            "max sequence length must be <=8192 and divisible by prefill chunk size"
        )
    if spec.frames_for_seconds(max(args.audio_buckets_s)) + 3 >= args.max_seq_len:
        raise ValueError("largest audio bucket leaves no generation budget")
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    args.checkpoint = fetch(
        args.checkpoint, "omniASR-LLM-1B-v2.pt", CHECKPOINT_URL, args.download_dir
    )
    args.tokenizer = fetch(
        args.tokenizer, "tokenizer.model", TOKENIZER_URL, args.download_dir
    )
    args.lang_table = fetch(
        args.lang_table, "languages.parquet", LANGUAGE_URL, args.download_dir
    )
    if args.stage:
        {"encoder": compile_encoder, "decoder": compile_decoder, "finalize": finalize}[
            args.stage
        ](args)
        return
    if args.output_dir.exists():
        raise FileExistsError(
            "output directory exists; choose a new directory to prevent mixed artifacts"
        )
    for path, url in (
        (args.checkpoint, CHECKPOINT_URL),
        (args.tokenizer, TOKENIZER_URL),
        (args.lang_table, LANGUAGE_URL),
    ):
        if digest(path) != SOURCE_HASHES[url]:
            raise ValueError(
                f"source checksum differs from the pinned public model: {path.name}"
            )
    args.output_dir.mkdir(parents=True)
    argv = sys.argv[1:]
    # Separate processes release compiler and device memory between static bucket builds.
    for stage, bucket in [
        ("decoder", None),
        *[("encoder", b) for b in sorted(args.audio_buckets_s)],
        ("finalize", None),
    ]:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *argv,
            "--stage",
            stage,
        ]
        if bucket is not None:
            command.extend(["--bucket", str(bucket)])
        started = time.perf_counter()
        subprocess.run(command, check=True)
        print(
            f"{stage} {bucket or ''} finished in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    from artifact import load_manifest

    load_manifest(args.output_dir)
    print(f"Complete artifact: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
