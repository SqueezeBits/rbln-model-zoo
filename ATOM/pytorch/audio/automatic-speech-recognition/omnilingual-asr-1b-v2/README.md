# Omnilingual ASR LLM 1B v2

Compile and reload Meta's [`omniASR_LLM_1B_v2`](https://github.com/facebookresearch/omnilingual-asr) on one **RBLN-CA25**. Like the Qwen3-ASR example, `compile.py` exports the audio encoder and language decoder, then `inference.py` reloads both to transcribe audio. Omnilingual uses a raw-waveform wav2vec2 encoder and an embedding-input LLaMA decoder; its binaries and input signatures are specific to this recipe.

The recorded SDK compiles and executes all four encoder buckets on NPU, but the encoder **does not pass the strict FP32 numerical parity check** (relative L2 0.103-0.117, cosine 0.9932-0.9947 on the recorded utterance). This is an experimental recipe; read [VALIDATION.md](VALIDATION.md) for the measured limits before using its output.

## Environment

Use Linux x86-64, Python 3.12, a CA25 SDK wheel and a compatible device driver. The checked toolchain is SDK `0.11.0.post1`, KMD `3.4.1`, PyTorch `2.14.1+cpu`, Transformers `5.20.1`, and optimum-rbln commit `a6a709ed0727f87cabfebf191b6a89b141f80731`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python /path/to/rebel_compiler-0.11.0.post1-cp312-cp312-manylinux_2_34_x86_64.whl
uv pip install --python .venv/bin/python --no-deps 'optimum-rbln @ git+https://github.com/rebel-jingu/optimum-rbln.git@a6a709ed0727f87cabfebf191b6a89b141f80731'
```

The last install deliberately uses `--no-deps`: the pinned optimum commit's package metadata declares an older Torch/Transformers stack. This recipe supplies the versions tested here. That commit can emit a compiler/library base-version warning with SDK 0.11; retain the warning in validation logs.

## Compile and reload

```bash
.venv/bin/python compile.py --device 0 --audio-buckets-s 5 10 --output-dir omniASR_LLM_1B_v2-rbln
.venv/bin/python download_sample.py
.venv/bin/python inference.py --device 0 --model-dir omniASR_LLM_1B_v2-rbln --audio .cache/sample/sample.flac --language eng_Latn
```

Compilation downloads the public checkpoint (8.5 GiB), v2 SentencePiece tokenizer and pinned upstream language table into `.cache/omniasr`. Existing files are checked against SHA-256 values before compilation. To use existing downloads, pass `--checkpoint`, `--tokenizer` and `--lang-table`. Use a new output directory for each build; partial artifacts are retained for diagnosis and cannot be loaded as complete models.

The defaults compile batch 1, 2048 decoder positions, prefill chunks of 128, and static 5 s / 10 s encoder buckets. Choose buckets explicitly with `--audio-buckets-s`; only the compiled buckets can run. Longer buckets up to 40 s are accepted by the compile CLI, subject to decoder capacity and device memory. Only the configurations in `VALIDATION.md` are physically verified. Every loaded bucket owns an encoder weight copy. If memory is limited, compile only the buckets needed.

Audio is decoded, converted to mono, resampled to 16 kHz, normalized **before** padding, and assigned to the smallest fitting bucket. Out-of-range input fails. The sample downloader keeps the LibriSpeech CC BY 4.0 license, attribution and reference text beside the audio. A user-supplied file works with `--audio /path/to/audio.wav`.

## Artifact contract

- `encoder/{seconds}s.rbln`: waveform `[1, seconds * 16000]` and float32 frame mask `[1, frames]` → projected embeddings `[1, frames, 4096]`. All seven convolution lengths use `floor((n - kernel) / stride) + 1`; 5 s emits 249 frames, 10 s emits 499. Masked frontend rows are zeroed before positional convolution, and encoder attention excludes padded keys. Only valid projected frames enter the decoder.
- `decoder/`: the complete optimum-rbln LLaMA prefill, batch-1 decode, configuration, generation configuration and embedding artifact. The public checkpoint has 48 encoder layers of width 1280 and 12 decoder layers of width 4096. Q/K rows are permuted from fairseq2 interleaved RoPE to HF rotate-half order before compiling.
- `prompt_embeddings.pt`, `tokenizer.model`, `lang_lookup.json`: the text/language tables and exact v2 tokenizer. The prompt is `[audio][LID marker][language embedding][BOS]`. Parquet index `i` selects language embedding row `i + 1`; row 0 is unconditional (`--language none`). Unknown language codes fail. The vocabulary has 10288 output tokens and 10289 text embedding rows; greedy decoding slices logits to 10288 before argmax so the input-only marker cannot be emitted.
- `compile.json`: source URLs and SHA-256 values, upstream/optimum revisions, recipe file hashes, toolchain versions, static bucket signatures, device placement metadata and all required artifact hashes. Reload verifies file completeness, integrity and the encoder/decoder contract before creating runtimes.

The encoder is loaded directly from saved `.rbln` files. Reload cannot recompile it or select a CPU encoder. The build and reload guards reject encoder host partitions. Dense per-group positional convolutions, channel-last GELU, and equivalent asymmetric padding keep the encoder kernels on NPU.

## Numerical and runtime verification

```bash
.venv/bin/python verify.py --device 0 --model-dir omniASR_LLM_1B_v2-rbln --checkpoint .cache/omniasr/omniASR-LLM-1B-v2.pt --audio .cache/sample/sample.flac --output-json verification.json
.venv/bin/python inference.py --device 0 --audio .cache/sample/sample.flac --warmup 1 --repeat 5 --output-json timing.json
.venv/bin/python inference.py --device 0 --audio .cache/sample/sample.flac --warmup 0 --repeat 1 --host-trace host-trace.json
```

`verify.py` checks every compiled encoder bucket against CPU output for the same audio, checks masked padding against unpadded CPU execution, compares the converted HF decoder against an independent decoder using the original interleaved weights, compares NPU prefill logits and three teacher-forced decode steps, verifies cache reset by repeating prefill after a request, and runs greedy transcription. Decoder parity uses the **same NPU audio embeddings** for CPU and NPU decoding; it isolates the decoder and does not establish end-to-end parity to the CPU encoder. `compile.json` indicates structural artifact completeness, while numerical validation is reported separately. The command collects all diagnostics and exits with status 1 if any numerical check fails. This is a numerical smoke check, not an accuracy evaluation. A matching transcript on one utterance does not establish accuracy across languages or durations.

Uninstrumented warm runs report separate median, p95 and raw samples for **CPU preparation** (local file read/decode, mono conversion, resampling, normalization), **prepared inference** (encoder + greedy decoder), and **E2E** (CPU preparation + prepared inference). Each measured request reads and prepares the audio again. Artifact loading, checksums and compilation are excluded from every timing. Host trace spans are collected in a separate instrumented run and open in Perfetto. They do not expose device kernels or prove zero bubbles. This recipe does not enable the SDK device profiler.

Meta's [code and models are Apache-2.0](https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/LICENSE). Source geometry and language conditioning follow upstream revision `81f51e224ce9e74b02cc2a3eaf21b2d91d743455`.
