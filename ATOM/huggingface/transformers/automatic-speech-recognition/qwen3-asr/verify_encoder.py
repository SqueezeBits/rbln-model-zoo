"""Verify a saved native Qwen3-ASR encoder against a pinned CPU reference."""

import argparse
import hashlib
import io
import json
import math
import re
import time
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"


def immutable_revision(value):
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise argparse.ArgumentTypeError(
            "source revision must be a full immutable commit SHA"
        )
    return value


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="saved parent artifact directory"
    )
    parser.add_argument(
        "--audio", required=True, help="16 kHz audio file or public HTTP(S) URL"
    )
    parser.add_argument("--device", type=int, default=0, help="RBLN runtime device")
    parser.add_argument("--output", type=Path, default=Path("encoder-validation.json"))
    parser.add_argument(
        "--reference-dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--reference-attention", choices=("eager", "sdpa"), default="eager"
    )
    parser.add_argument("--source-model", default=MODEL_ID)
    parser.add_argument("--source-revision", type=immutable_revision, default=REVISION)
    return parser.parse_args()


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def architecture_config(value):
    """Ignore loader locations, library version and the selected loading precision."""
    if isinstance(value, dict):
        return {
            key: architecture_config(item)
            for key, item in value.items()
            if not (isinstance(key, str) and key.startswith("_"))
            and key not in ("transformers_version", "dtype", "torch_dtype")
        }
    if isinstance(value, list):
        return [architecture_config(item) for item in value]
    return value


def recorded_provenance(model, model_id, revision):
    graph = model / "audio_tower" / "compiled_model.rbln"
    actual = digest(graph)
    record_path = Path(__file__).with_name("validation.json")
    record = json.loads(record_path.read_text()) if record_path.is_file() else {}
    files = record.get("compile", {}).get("artifact_files", [])
    expected = next(
        (
            item["sha256"]
            for item in files
            if item["path"] == "audio_tower/compiled_model.rbln"
        ),
        None,
    )
    bound = (
        actual == expected
        and record.get("model_id") == model_id
        and record.get("model_revision") == revision
    )
    return {
        "encoder_graph_sha256": actual,
        "matches_adjacent_validation_record": bound,
        "artifact_weight_identity": "recorded" if bound else "unverified",
    }


class RuntimeCounter:
    def __init__(self, runtime):
        if (type(runtime).__module__, type(runtime).__name__) != (
            "rebel.sync_runtime",
            "SyncRuntime",
        ):
            raise RuntimeError(
                f"Expected an actual NPU SyncRuntime, found {type(runtime)}"
            )
        self.runtime = runtime
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.runtime(*args, **kwargs)


def verify(args):
    import numpy as np
    import rebel
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from optimum.rbln import RBLNAutoModelForMultimodalLM, RBLNQwen3ASREncoder
    from transformers import (
        AutoConfig,
        AutoModelForMultimodalLM,
        AutoProcessor,
        PreTrainedModel,
    )

    torch.set_num_threads(16)
    torch.manual_seed(42)
    if args.audio.startswith(("https://", "http://")):
        with urlopen(args.audio, timeout=60) as response:
            audio_bytes = response.read()
    else:
        audio_bytes = Path(args.audio).read_bytes()
    pcm, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if pcm.ndim == 2:
        pcm = pcm.mean(axis=-1)
    if sample_rate != 16000 or not len(pcm):
        raise ValueError("audio must be nonempty and sampled at 16000 Hz")

    source = Path(snapshot_download(args.source_model, revision=args.source_revision))
    if source.name != args.source_revision:
        raise ValueError(
            "resolved CPU checkpoint revision differs from the pinned source revision"
        )
    reference = AutoModelForMultimodalLM.from_pretrained(
        source,
        local_files_only=True,
        dtype=getattr(torch, args.reference_dtype),
        attn_implementation=args.reference_attention,
    ).eval()
    saved_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if architecture_config(saved_config.to_dict()) != architecture_config(
        reference.config.to_dict()
    ):
        raise ValueError(
            "saved artifact model configuration differs from the pinned CPU reference"
        )
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    result = {
        "model_id": args.source_model,
        "source_revision": args.source_revision,
        "resolved_source_revision": source.name,
        "source_revision_verified": True,
        "saved_model_architecture_config_matches_reference": True,
        "provenance": recorded_provenance(
            args.model, args.source_model, args.source_revision
        ),
        "versions": {
            name: version(name)
            for name in ("torch", "transformers", "optimum-rbln", "rebel-compiler")
        },
        "reference": {
            "dtype": args.reference_dtype,
            "attention": args.reference_attention,
        },
        "fixture": {
            "sha256": hashlib.sha256(audio_bytes).hexdigest(),
            "sample_rate": sample_rate,
            "duration_seconds": len(pcm) / sample_rate,
        },
        "strict_tolerance": {"atol": 1e-3, "rtol": 1e-3},
        "cases": [],
    }
    denied = {
        "rebel.compile": 0,
        "rebel.compile_from_torch": 0,
        "AutoModelForMultimodalLM.from_pretrained": 0,
        "PreTrainedModel.from_pretrained": 0,
    }

    def forbid(name):
        def fail(*unused_args, **unused_kwargs):
            denied[name] += 1
            raise RuntimeError(f"saved-artifact execution attempted forbidden {name}")

        return fail

    with ExitStack() as stack:
        for owner, attribute, name in (
            (rebel, "compile", "rebel.compile"),
            (rebel, "compile_from_torch", "rebel.compile_from_torch"),
            (
                AutoModelForMultimodalLM,
                "from_pretrained",
                "AutoModelForMultimodalLM.from_pretrained",
            ),
            (PreTrainedModel, "from_pretrained", "PreTrainedModel.from_pretrained"),
        ):
            stack.enter_context(
                patch.object(owner, attribute, side_effect=forbid(name))
            )
        start = time.perf_counter()
        parent = RBLNAutoModelForMultimodalLM.from_pretrained(
            args.model, export=False, rbln_device=args.device, local_files_only=True
        )
        encoder = RBLNQwen3ASREncoder.from_pretrained(
            args.model / "audio_tower",
            export=False,
            rbln_device=args.device,
            local_files_only=True,
        )
        result["reload_seconds"] = time.perf_counter() - start
        parent_counter = RuntimeCounter(parent.audio_tower.model[0])
        standalone_counter = RuntimeCounter(encoder.model[0])
        parent.audio_tower.model[0], encoder.model[0] = (
            parent_counter,
            standalone_counter,
        )
        chunk_length = encoder.chunk_len
        windows = encoder.rbln_config.num_windows
        chunks_per_window = encoder.chunks_per_window
        result["compiled_encoder"] = {
            "num_windows": windows,
            "chunk_frames": chunk_length,
            "window_tokens": encoder.window_size,
            "dtype": str(encoder.dtype),
            "input_info": encoder.rbln_config.compile_cfgs[0].input_info,
            "runtime": "rebel.sync_runtime.SyncRuntime",
        }
        durations = [(1.5,), (4.0,), (7.9,), (8.0,), (8.1,), (12.0,), (1.5, 12.0)]
        with torch.inference_mode():
            for seconds in durations:
                clips = [
                    np.tile(pcm, math.ceil(value * sample_rate / len(pcm)))[
                        : int(value * sample_rate)
                    ]
                    for value in seconds
                ]
                inputs = processor.feature_extractor(
                    clips,
                    sampling_rate=sample_rate,
                    padding="longest",
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                features, mask = inputs["input_features"], inputs["attention_mask"]
                expected = reference.model.get_audio_features(
                    input_features=features.to(reference.dtype),
                    input_features_mask=mask,
                    return_dict=True,
                ).pooler_output
                previous = parent_counter.calls, standalone_counter.calls
                start = time.perf_counter()
                actual = parent.audio_tower(features, mask)
                elapsed = time.perf_counter() - start
                separate = encoder(features, mask)
                valid_rows = mask.reshape(len(clips), -1, chunk_length).sum(-1)
                for _ in range(3):
                    valid_rows = torch.where(valid_rows > 0, (valid_rows + 1) // 2, 0)
                expected_calls = math.ceil(
                    len(clips)
                    * math.ceil(features.shape[-1] / chunk_length / chunks_per_window)
                    / windows
                )
                difference = actual.float() - expected.float()
                result["cases"].append(
                    {
                        "audio_seconds": list(seconds),
                        "feature_shape": list(features.shape),
                        "valid_frames": mask.sum(-1).tolist(),
                        "output_shape": list(actual.shape),
                        "reference_shape": list(expected.shape),
                        "expected_valid_rows": int(valid_rows.sum()),
                        "shape_matches_reference": actual.shape == expected.shape,
                        "expected_runtime_calls": expected_calls,
                        "parent_runtime_calls": parent_counter.calls - previous[0],
                        "standalone_runtime_calls": standalone_counter.calls
                        - previous[1],
                        "parent_standalone_exact_equal": torch.equal(actual, separate),
                        "finite": bool(torch.isfinite(actual).all()),
                        "reference_finite": bool(torch.isfinite(expected).all()),
                        "max_abs": difference.abs().max().item(),
                        "relative_l2": (
                            difference.norm() / expected.float().norm()
                        ).item(),
                        "cosine": torch.nn.functional.cosine_similarity(
                            actual.float().flatten(), expected.float().flatten(), dim=0
                        ).item(),
                        "strict_allclose": torch.allclose(
                            actual.float(), expected.float(), atol=1e-3, rtol=1e-3
                        ),
                        "unprofiled_parent_encoder_host_seconds": elapsed,
                    }
                )
        result["blocked_loader_and_compiler_calls"] = denied
        result["cpu_reference_loaded_before_loader_block"] = True
        result["functional_status"] = (
            "passed"
            if all(
                case["finite"]
                and case["reference_finite"]
                and case["shape_matches_reference"]
                and case["parent_standalone_exact_equal"]
                and case["output_shape"][0] == case["expected_valid_rows"]
                and case["parent_runtime_calls"]
                == case["standalone_runtime_calls"]
                == case["expected_runtime_calls"]
                for case in result["cases"]
            )
            else "failed"
        )
        result["strict_numeric_parity"] = all(
            case["strict_allclose"] for case in result["cases"]
        )
    return result


def main():
    args = arguments()
    # Preserve structured failure output for arbitrary SDK errors at the CLI boundary.
    try:
        result = verify(args)
    except Exception as error:  # noqa: BLE001
        result = {
            "functional_status": "failed",
            "strict_numeric_parity": None,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("functional_status", "strict_numeric_parity")}
        )
    )
    return 0 if result["functional_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
