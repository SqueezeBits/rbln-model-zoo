# Recorded validation — 2026-10-21

**Status: experimental. All four encoder buckets compile and execute on NPU, but strict encoder FP32 numerical parity failed.** `verify.py` preserves the numerical failures and exits with status **1**. Structural artifact completeness in `compile.json` is a separate result.

On the single recorded utterance, CPU FP32 encoder → CPU FP32 decoder and NPU encoder → NPU decoder produce identical token IDs and text, including EOS termination. This single utterance does not establish accuracy across languages, speakers, noise or durations.

Machine-readable evidence is in [validation_results.json](validation_results.json).

## Configuration and sources

One RBLN-CA25, KMD 3.4.1, SDK 0.11.0.post1, Python 3.12.3, Torch 2.14.1+cpu and Transformers 5.20.1. Decoder batch 1, maximum sequence length 2048, prefill chunk 128, eager attention, no weight quantization. Encoder and decoder logical graph inputs/outputs are float32.

- Model/code: [Meta Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr/tree/81f51e224ce9e74b02cc2a3eaf21b2d91d743455), revision `81f51e224ce9e74b02cc2a3eaf21b2d91d743455`, Apache-2.0.
- Adapter: [optimum-rbln](https://github.com/rebel-jingu/optimum-rbln/tree/a6a709ed0727f87cabfebf191b6a89b141f80731), revision `a6a709ed0727f87cabfebf191b6a89b141f80731`.
- Model checkpoint SHA-256: `cceb4d9ebac3d168a6af6b26c62ce11bafc562b38976c6bfa87e7d60422c6da5`.
- V2 tokenizer SHA-256: `8aa11a1092142ef472537476ef6e76541123e2f0d789b79f3ebd119008240b1e`.
- Pinned language table SHA-256: `1f5e1486c455a9ab6e492c0b187e538a5c3736e876e20e7417ae6678a034f5b8`.

## Encoder

Each saved `.rbln` binary was inspected and independently reloaded. All have **one device subgraph and no host-op reasons**, target one CA25, and execute with finite full-bucket output. Dense per-group positional convolutions, channel-last GELU and equivalent asymmetric positional padding remove the observed CPU partitions.

For the 3.505 s speech fixture, each bucket consumes 56080 real waveform samples / 175 valid frames. The float32 frame mask zeroes projected frontend padding before positional convolution and supplies a boolean SDPA key mask. Finite `-10000` masking was rejected: this checkpoint has attention scores above 90000. A separate synthetic NPU probe with masked-key score 100623 excluded those keys correctly.

| Static bucket | Waveform samples | Output frames | CPU padded vs unpadded relative L2 | NPU vs CPU FP32 relative L2 | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 s | 80000 | 249 | 1.76e-6 | **0.102869** | **0.994698** |
| 10 s | 160000 | 499 | 1.21e-6 | **0.102993** | **0.994685** |
| 20 s | 320000 | 999 | 1.21e-6 | **0.110528** | **0.993882** |
| 40 s | 640000 | 1999 | 1.21e-6 | **0.116551** | **0.993196** |

Padding invariance passes its 1e-4 threshold. NPU encoder parity fails the unchanged thresholds of relative L2 ≤ 0.02 and cosine ≥ 0.999. The largest differences concentrate in an output projection channel. The cause remains unresolved. On the same 5 s waveform, CPU FP16 differs from CPU FP32 by relative L2 0.01328 and CPU BF16 by 0.12312; these sensitivity diagnostics do not prove the NPU computation correct. Alternate BF16 and key-centering candidates did not resolve the full-encoder discrepancy and are excluded from the recipe.

## Decoder and utterance

The decoder comparisons below use the **same NPU audio embeddings** for the CPU and NPU decoders. They isolate decoder conversion, attention positions and KV-cache behavior; they do not establish end-to-end encoder numerical parity.

| Check | Result |
| --- | --- |
| Interleaved-RoPE conversion proof | Relative error 8.79e-16; unpermuted control fails |
| Original-weight CPU decoder vs converted HF CPU decoder | Relative L2 3.29e-7 |
| NPU vs original-weight CPU decoder prefill logits | Relative L2 0.001204; first token matches |
| Three teacher-forced cached decode steps | Relative L2 0.002089 / 0.001782 / 0.001661; next-token argmax matches |
| Prefill repeated after a prior decode request | Maximum absolute difference 0.0 |
| CPU FP32 full pipeline vs NPU full pipeline | All 46 generated token IDs match; both stop on EOS |

The fixture is LibriSpeech `6930-75918-0000`, supplied by Vassil Panayotov et al. under [CC BY 4.0](https://www.openslr.org/12). Audio SHA-256: `9ce35224156f071ab58eb7feb8a5ceae600f6f9f353da2a6cbf797b6b1ac8a23`. The downloader preserves attribution, license and reference text.

Reference: `CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS`.

CPU and NPU greedy transcription: `concord returned to its place amidst the tents`.

## Warm request timing

One warmup request, then five uninstrumented requests on the same local 3.505 s fixture. Every measured request reads, decodes and prepares the audio again. Artifact loading, integrity checks and compilation are outside the timed interval.

| Scope | Median | p95 |
| --- | ---: | ---: |
| CPU preparation: local read/decode, mono, resample, normalize | 0.970 ms | 1.489 ms |
| Prepared inference: NPU encoder + greedy decoder | 579.643 ms | 579.996 ms |
| Local-file E2E: CPU preparation + prepared inference | 580.688 ms | 581.213 ms |

A separate instrumented request records host `cpu_prepare`, encoder, decoder-prefill and decoder-step spans in Chrome/Perfetto format. Device timelines were not collected; no zero-bubble claim is made. Five requests on one utterance provide a limited timing sample, not a throughput or service-latency evaluation.

## Guards and reproduction

Missing tokenizer, obsolete finite-mask contract, negative device, unknown language and over-bucket audio are rejected. Invalid request preflight creates zero NPU runtimes. Numerical metrics explicitly reject empty, NaN and infinite tensors. Ruff and Python compilation checks pass.

Follow the environment setup in `README.md`, then run:

```bash
.venv/bin/python compile.py --device 0 --audio-buckets-s 5 10 20 40 --output-dir omniASR_LLM_1B_v2-rbln
.venv/bin/python download_sample.py
.venv/bin/python verify.py --device 0 --model-dir omniASR_LLM_1B_v2-rbln --checkpoint .cache/omniasr/omniASR-LLM-1B-v2.pt --audio .cache/sample/sample.flac --output-json verification.json
```

The recorded numerical verification returns exit status 1 for the encoder failures above. Preserve that status when using the recipe in CI. Timing and host trace commands are in `README.md`.
