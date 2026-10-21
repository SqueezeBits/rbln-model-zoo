"""Compile static text, vision and audio encoders and inspect the saved binaries."""

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
DEFAULT_NPU = "RBLN-CR13"
DEFAULT_DEVICES = [0]
DEFAULT_AUDIO_LAYOUT = "row"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("embeddinggemma-2-compiled")
    )
    parser.add_argument("--devices", type=int, nargs="+", default=DEFAULT_DEVICES)
    parser.add_argument(
        "--npu", default=DEFAULT_NPU, choices=["RBLN-CA25", "RBLN-CR13"]
    )
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--text-buckets", type=int, nargs="+", default=[32, 64, 96, 128, 288, 352, 512]
    )
    parser.add_argument(
        "--audio-buckets", type=int, nargs="+", default=[64, 128, 1056, 1536]
    )
    parser.add_argument("--image-soft-tokens", type=int, choices=[70, 280], default=70)
    parser.add_argument(
        "--audio-layout", choices=["row", "selector"], default=DEFAULT_AUDIO_LAYOUT
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Save graphs without creating NPU runtimes",
    )
    return parser


def inspect_artifacts(output, npu, num_devices):
    import rebel

    config = json.loads((output / "rbln_config.json").read_text())
    graphs = []
    for cfg in config["_compile_cfgs"]:
        path = output / (cfg["compiled_model_name"] + ".rbln")
        inspected = rebel.RBLNCompiledModel.inspect(path)
        if inspected["npu"] != npu or inspected["num_devices"] != num_devices:
            raise RuntimeError(f"Unexpected compiled target for {path.name}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        graphs.append(
            {
                "file": path.name,
                "sha256": digest,
                "npu": inspected["npu"],
                "num_devices": inspected["num_devices"],
                "inputs": inspected["inputs"],
            }
        )
    if not graphs:
        raise RuntimeError("No compiled encoder graphs were saved")
    return graphs


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "Use a new output directory; existing artifacts are never overwritten"
        )
    if (
        args.threads < 1
        or not args.devices
        or len(set(args.devices)) != len(args.devices)
        or min(args.devices) < 0
    ):
        parser.error(
            "threads must be positive and devices must be distinct nonnegative IDs"
        )
    if any(not 1 <= n <= 8192 for n in args.text_buckets) or any(
        not 4 <= n <= 3072 for n in args.audio_buckets
    ):
        parser.error(
            "text buckets must be within 1-8192 and audio buckets within 4-3072"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import rebel
    import torch
    from optimum.rbln import RBLNEmbeddingGemma2Model
    from transformers import AutoModel, AutoProcessor

    torch.set_num_threads(args.threads)
    controls = rebel if hasattr(rebel, "set_num_threads") else rebel._C
    controls.set_num_threads(args.threads)
    source = AutoModel.from_pretrained(
        MODEL_ID,
        revision=args.revision,
        dtype=getattr(torch, args.dtype),
        attn_implementation="eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=args.revision)
    model = RBLNEmbeddingGemma2Model.from_model(
        source,
        preprocessors=[processor],
        model_save_dir=args.output,
        rbln_config={
            "npu": args.npu,
            "num_devices": len(args.devices),
            "device": args.devices,
            "dtype": args.dtype,
            "create_runtimes": not args.compile_only,
            "text_buckets": args.text_buckets,
            "audio_buckets": args.audio_buckets,
            "image_soft_tokens": args.image_soft_tokens,
            "audio_layout": args.audio_layout,
        },
    )
    try:
        report = {
            "schema_version": 1,
            "model_id": MODEL_ID,
            "model_revision": args.revision,
            "dtype": args.dtype,
            "devices": args.devices,
            "create_runtimes": not args.compile_only,
            "audio_layout": args.audio_layout,
            "image_soft_tokens": args.image_soft_tokens,
            "versions": {
                name: version(name)
                for name in ("optimum-rbln", "rebel-compiler", "torch", "transformers")
            },
            "graphs": inspect_artifacts(args.output, args.npu, len(args.devices)),
        }
        # This marker is written only after every saved binary passes inspection.
        temporary = args.output / ".compile-manifest.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output / "compile-manifest.json")
        print(
            json.dumps(
                {
                    "saved": str(args.output),
                    "graphs": len(report["graphs"]),
                    "npu": args.npu,
                }
            )
        )
    finally:
        model.close()


if __name__ == "__main__":
    main()
