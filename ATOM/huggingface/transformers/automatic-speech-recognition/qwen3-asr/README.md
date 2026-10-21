# Qwen3-ASR

`compile.py` exports the native Hugging Face checkpoint `Qwen/Qwen3-ASR-1.7B-hf`.
The parent export compiles both the text decoder and the audio encoder with its
multimodal projector. The saved encoder lives under `audio_tower/`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python /absolute/path/to/rebel_compiler.whl
uv pip install --python .venv/bin/python -r requirements.txt
uv run --no-project --python .venv/bin/python python compile.py
uv run --no-project --python .venv/bin/python python inference.py
```

Use a compiler/library/driver combination from the supported RBLN version matrix.
The SDK wheel comes from the private package index and is not stored here.

The example defaults to four text-decoder devices and four audio attention
windows per encoder call. `audio_tower.num_windows` controls work per call,
independently of decoder batch size. For this checkpoint one window is 104
post-convolution tokens, approximately eight seconds of audio. Longer inputs
run through the same encoder graph in additional calls; samples in a batch keep
separate windows. Use the matching processor and its `input_features_mask`.

The checkpoint chooses BF16 when no loading dtype is specified. Inspect the
saved `rbln_config.json` signatures when recording precision; a requested
configuration value alone does not establish the compiled dtype.

## Single-device smoke configuration

This smaller configuration was used for the validation below. It differs from
the longer-context defaults in `compile.py`.

```python
from optimum.rbln import RBLNAutoModelForMultimodalLM

model = RBLNAutoModelForMultimodalLM.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B-hf",
    revision="bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
    export=True,
    rbln_config={
        "batch_size": 1,
        "max_seq_len": 1024,
        "num_devices": 1,
        "npu": "RBLN-CA25",
        "attn_impl": "eager",
        "prefill_chunk_size": 128,
        "kvcache_block_size": 1024,
        "kvcache_num_blocks": 1,
        "audio_tower": {"num_windows": 1},
    },
)
model.save_pretrained("qwen3-asr-smoke")
```

The saved encoder can also be loaded on its own, without the decoder or source
checkpoint weights:

```python
from optimum.rbln import RBLNQwen3ASREncoder
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("qwen3-asr-smoke", local_files_only=True)
encoder = RBLNQwen3ASREncoder.from_pretrained(
    "qwen3-asr-smoke/audio_tower", export=False, local_files_only=True
)
# inputs = processor.apply_transcription_request(audio=...)
# embeddings = encoder(inputs["input_features"], inputs["input_features_mask"])
```

The result contains projected audio embeddings of width 2048. Unprojected
encoder features and language-identification heads are separate interfaces.

## Reproduce the encoder checks

Use `verify_encoder.py` with a saved parent model and a 16 kHz audio file or
public audio URL. The default CPU reference is **BF16 eager**, matching the
recorded artifact's weight precision. Reference loading uses the immutable model
revision above; `--source-model` and `--source-revision` allow another native
Qwen3-ASR checkpoint, with a full commit SHA required.

```bash
python verify_encoder.py \
  --model qwen3-asr-smoke \
  --audio speech.flac \
  --device 0 \
  --output encoder-validation.json
```

`--reference-dtype float32` and `--reference-attention sdpa` select additional
CPU references. The script loads CPU weights before blocking compiler entry
points and Hugging Face weight loaders. It then reloads the parent and standalone
encoder, checks actual NPU `SyncRuntime` calls, and compares single and mixed
batch inputs around the eight-second window boundary. It records shape checks,
cosine similarity, relative L2 and strict `allclose` without changing tolerance.

Exit code zero means the **functional** reload/runtime/window/shape checks
passed. The separate `strict_numeric_parity` Boolean can still be false and
must be inspected. The output also records whether the encoder graph hash
matches the adjacent validation record. An immutable CPU reference and matching
architecture config alone do not prove an unrecorded artifact's weight identity.

## Recorded validation

[validation.json](validation.json) records the source revision, input and artifact
hashes, exact package versions, graph signatures and measurements from a CA25 run.
It used optimum source commit `a6a709ed0727f87cabfebf191b6a89b141f80731`,
Transformers 5.15.1, PyTorch 2.13.0+cpu and compiler 0.11.0.post1. The library and
compiler emitted a version mismatch warning. This is evidence for the recorded
smoke configuration, rather than a supported-version matrix.

- Parent export saved encoder, prefill and decoder graphs. The encoder signature
  was BF16 `[8, 1, 128, 100]` with a BF16 `[1, 1, 1, 104]` attention mask.
- Reload and execution ran in another process with compiler entry points and
  Hugging Face source-weight loading blocked. Encoder and text execution used
  `rebel.sync_runtime.SyncRuntime`. Loading only `audio_tower/` produced exactly
  the same encoder output as loading the parent artifact.
- Inputs of 1.5, 4, 7.9 and 8 seconds used one encoder call; 8.1 and 12 seconds
  used two. A mixed 1.5/12-second batch used four calls, with 176 valid output
  rows. All tested output shapes matched the CPU reference.
- Strict encoder `allclose(atol=1e-3, rtol=1e-3)` was **false**. Minimum cosine
  similarity was 0.999261 and maximum relative L2 error was 3.85% against CPU
  eager BF16. CPU SDPA BF16 and eager FP32 comparisons are recorded separately.
- Greedy token IDs matched CPU eager BF16 exactly on the 10.435-second fixture.
  CPU SDPA BF16 and eager FP32 omitted one comma that the NPU output included;
  punctuation-insensitive word sequences matched all three references.
- Five unprofiled encoder host calls took approximately 12.25-12.55 ms each,
  excluding feature extraction and text generation. A separate CPU host trace
  was captured; the native `.pb` profiler was not used.

These are functional and numerical smoke checks on one English fixture, clipped
or repeated for the length cases. They do not establish corpus WER/CER,
bit-exact encoder parity, performance of the default four-device configuration,
or compatibility of artifacts across SDK releases.
