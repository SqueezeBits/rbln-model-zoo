"""Portable artifact contract shared by compilation, reload and verification."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import torch

UPSTREAM_REVISION = "81f51e224ce9e74b02cc2a3eaf21b2d91d743455"
OPTIMUM_REVISION = "a6a709ed0727f87cabfebf191b6a89b141f80731"
MODEL_NAME = "omniASR_LLM_1B_v2"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/mms/omniASR-LLM-1B-v2.pt"
TOKENIZER_URL = "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_written_v2.model"
LANGUAGE_URL = (
    "https://raw.githubusercontent.com/facebookresearch/omnilingual-asr/"
    f"{UPSTREAM_REVISION}/src/omnilingual_asr/models/wav2vec2_llama/languges_lookup_table.parquet"
)

SOURCE_HASHES = {
    CHECKPOINT_URL: "cceb4d9ebac3d168a6af6b26c62ce11bafc562b38976c6bfa87e7d60422c6da5",
    TOKENIZER_URL: "8aa11a1092142ef472537476ef6e76541123e2f0d789b79f3ebd119008240b1e",
    LANGUAGE_URL: "1f5e1486c455a9ab6e492c0b187e538a5c3736e876e20e7417ae6678a034f5b8",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def versions():
    names = (
        "torch",
        "transformers",
        "optimum-rbln",
        "rebel-compiler",
        "sentencepiece",
        "numpy",
        "soundfile",
    )
    return {
        "python": platform.python_version(),
        **{n: importlib.metadata.version(n) for n in names},
    }


def check_toolchain():
    expected = {
        "torch": "2.14.1+cpu",
        "transformers": "5.20.1",
        "rebel-compiler": "0.11.0.post1",
    }
    for name, version in expected.items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"this recipe requires {name}=={version}")
    direct = importlib.metadata.distribution("optimum-rbln").read_text(
        "direct_url.json"
    )
    revision = json.loads(direct or "{}").get("vcs_info", {}).get("commit_id")
    if revision != OPTIMUM_REVISION:
        raise RuntimeError(
            f"install optimum-rbln from the pinned Git revision {OPTIMUM_REVISION}"
        )


def assert_npu_graph(metadata):
    """Reject compiler graph partitions containing host tensor operations."""
    stats = metadata.get("subgraph", {}).get("stats", [])
    if not stats:
        raise RuntimeError("compiler metadata has no graph-placement statistics")
    for stat in stats:
        reasons = stat.get("host_op_reason", {})
        if any(reasons.values()) or stat.get("device_func_count") != 1:
            raise RuntimeError(
                f"encoder contains host operations or graph breaks: {stat}"
            )


def load_manifest(root, verify_hashes=True):
    root = Path(root)
    manifest = json.loads((root / "compile.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("model") != MODEL_NAME:
        raise ValueError("unsupported Omnilingual artifact contract")
    if manifest.get("encoder_key_mask") != "boolean":
        raise ValueError("encoder contract requires exact boolean key exclusion")
    buckets = manifest["encoder_buckets"]
    seconds = [b["seconds"] for b in buckets]
    if (
        not seconds
        or seconds != sorted(set(seconds))
        or any(s < 1 or s > 40 for s in seconds)
    ):
        raise ValueError(
            "encoder buckets must be unique, ordered clip durations in 1..40 seconds"
        )
    required = {
        "decoder/config.json",
        "decoder/rbln_config.json",
        "decoder/prefill.rbln",
        "decoder/decoder_batch_1.rbln",
        "decoder/torch_artifacts.pth",
        "prompt_embeddings.pt",
        "lang_lookup.json",
        "tokenizer.model",
    }
    required.update(b["file"] for b in manifest["encoder_buckets"])
    if not required.issubset(manifest["files"]):
        raise ValueError(
            "artifact manifest omits required encoder/decoder/tokenizer files"
        )
    for name, record in manifest["files"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise ValueError(f"artifact missing or size mismatch: {name}")
        if verify_hashes and digest(path) != record["sha256"]:
            raise ValueError(f"artifact checksum mismatch: {name}")
    for bucket in manifest["encoder_buckets"]:
        frames, samples = bucket["frames"], bucket["samples"]
        from model_spec import ModelSpec

        if (
            samples != bucket["seconds"] * 16000
            or ModelSpec().frames_for_samples(samples) != frames
        ):
            raise ValueError("encoder bucket sample/frame mismatch")
        import rebel

        info = rebel.RBLNCompiledModel.inspect(root / bucket["file"])
        if info != bucket["metadata"]:
            raise ValueError("encoder binary metadata differs from the saved ABI")
        if info["npu"] != manifest["target_npu"] or info["num_devices"] != 1:
            raise ValueError("encoder device target differs from the decoder contract")
        assert_npu_graph(info)
        if any(i["dtype"] != "float32" for i in info["inputs"] + info["outputs"]):
            raise ValueError(
                "encoder ABI requires float32 waveform, mask and projected embeddings"
            )
        shapes = [i["shape"] for i in info["inputs"]]
        if shapes != [[1, samples], [1, frames]]:
            raise ValueError(f"encoder input ABI mismatch: {shapes}")
        if info["outputs"][0]["shape"] != [1, frames, manifest["decoder_width"]]:
            raise ValueError("encoder output does not match decoder embedding width")
    if manifest["embedding_rows"] != manifest["real_logit_width"] + 1:
        raise ValueError("the LID marker must be an input-only embedding row")
    config = json.loads((root / "decoder/config.json").read_text())
    if (
        config["hidden_size"] != manifest["decoder_width"]
        or config["vocab_size"] != manifest["embedding_rows"]
    ):
        raise ValueError("decoder geometry disagrees with encoder/text embeddings")
    decoder_config = json.loads((root / "decoder/rbln_config.json").read_text())
    expected = {
        "batch_size": 1,
        "max_seq_len": manifest["max_seq_len"],
        "prefill_chunk_size": manifest["prefill_chunk_size"],
        "decoder_batch_sizes": [1],
        "use_inputs_embeds": True,
        "use_attention_mask": True,
        "attn_impl": "eager",
    }
    if any(decoder_config.get(k) != v for k, v in expected.items()):
        raise ValueError("decoder static configuration disagrees with manifest")
    for name, saved in manifest["decoder_metadata"].items():
        info = rebel.RBLNCompiledModel.inspect(root / "decoder" / name)
        if (
            info != saved
            or info["npu"] != manifest["target_npu"]
            or info["num_devices"] != 1
        ):
            raise ValueError("decoder binary target/ABI disagrees with manifest")
        assert_npu_graph(info)
    return manifest


def load_tables(root, manifest):
    root = Path(root)
    tables = torch.load(
        root / "prompt_embeddings.pt", map_location="cpu", weights_only=True
    )
    text, lang = tables["text"], tables["language"]
    if list(text.shape) != [manifest["embedding_rows"], manifest["decoder_width"]]:
        raise ValueError("text embedding table disagrees with manifest")
    if list(lang.shape) != [manifest["language_rows"], manifest["decoder_width"]]:
        raise ValueError("language embedding table disagrees with manifest")
    lookup = json.loads((root / "lang_lookup.json").read_text())
    if set(lookup.values()) != set(range(1, lang.shape[0])):
        raise ValueError("language rows must cover 1..N; row 0 is unconditional")
    return text, lang, lookup


def make_prompt(audio, text, language, lookup, manifest, language_code):
    row = 0 if language_code is None else lookup[language_code.lower()]
    marker = text[manifest["real_logit_width"]].view(1, 1, -1)
    lid = language[row].view(1, 1, -1)
    bos = text[manifest["bos_token_id"]].view(1, 1, -1)
    return torch.cat((audio, marker, lid, bos), dim=1).contiguous()
