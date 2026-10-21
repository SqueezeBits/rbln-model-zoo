"""Reload a compiled EmbeddingGemma 2 model and embed text, images or audio."""

import argparse
import hashlib
import json
import os
from pathlib import Path

DEFAULT_DEVICES = [0]


def check_manifest(directory):
    try:
        manifest = json.loads((directory / "compile-manifest.json").read_text())
        config = json.loads((directory / "rbln_config.json").read_text())
        graphs = manifest["graphs"]
        expected = {
            cfg["compiled_model_name"] + ".rbln" for cfg in config["_compile_cfgs"]
        }
        if (
            manifest["schema_version"] != 1
            or not expected
            or len(graphs) != len(expected)
        ):
            raise ValueError("Unsupported or incomplete compile manifest")
        if {item["file"] for item in graphs} != expected:
            raise ValueError(
                "Compile manifest does not match the saved graph inventory"
            )
        for item in graphs:
            path = directory / item["file"]
            if path.name != item["file"]:
                raise ValueError("Graph filenames must be local to the saved model")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != item["sha256"]:
                raise ValueError(f"Graph hash mismatch: {path.name}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Invalid or missing compile-manifest.json: {error}"
        ) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("embeddinggemma-2-compiled"))
    parser.add_argument("--devices", type=int, nargs="+", default=DEFAULT_DEVICES)
    parser.add_argument("--text")
    parser.add_argument("--images", nargs="+", default=[])
    parser.add_argument("--audio", nargs="+", default=[])
    parser.add_argument(
        "--dimensions", type=int, choices=[128, 256, 512, 768], default=768
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--trace",
        type=Path,
        help="Additional host trace pass for Perfetto; no device kernel timeline",
    )
    args = parser.parse_args()
    if (
        args.threads < 1
        or len(set(args.devices)) != len(args.devices)
        or min(args.devices) < 0
    ):
        parser.error(
            "threads must be positive and devices must be distinct nonnegative IDs"
        )
    if not args.model.is_dir():
        parser.error("Compile the model first; --model must be a saved model directory")
    try:
        check_manifest(args.model)
    except ValueError as error:
        parser.error(str(error))
    if args.trace is not None and args.trace.exists():
        parser.error("Use a new trace path")
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import rebel
    import torch
    from optimum.rbln import RBLNAutoModel
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_profiling import (
        TraceSession,
    )
    from transformers import AutoProcessor

    torch.set_num_threads(args.threads)
    controls = rebel if hasattr(rebel, "set_num_threads") else rebel._C
    controls.set_num_threads(args.threads)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = RBLNAutoModel.from_pretrained(
        args.model, export=False, rbln_device=args.devices
    )
    request = {"images": args.images, "audio": args.audio}
    if args.text is not None:
        request["text"] = args.text
    elif not args.images and not args.audio:
        request["text"] = "What causes the northern lights?"
    try:
        embedding = model.encode(
            request, processor=processor, dimensions=args.dimensions
        )
        if args.trace is not None:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            with TraceSession(args.trace):
                model.encode(request, processor=processor, dimensions=args.dimensions)
        print(
            json.dumps(
                {
                    "shape": list(embedding.shape),
                    "norm": float(embedding.norm()),
                    "graphs": model.engine.last_graphs,
                    "stages": model.engine.last_stages,
                    "embedding": embedding.tolist(),
                }
            )
        )
    finally:
        model.close()


if __name__ == "__main__":
    main()
