# EmbeddingGemma 2: text, images and audio

Compile all three encoders with the native `RBLNEmbeddingGemma2Model` adapter.
This recipe defaults to **RBLN-CR13**, logical NPU IDs **0**, BF16, batch 1,
70 image soft tokens and the **row** audio layout. Each encoder graph uses
the selected number of NPUs; the SDK chooses its distribution.

The output is one normalized embedding with 768, 512, 256 or 128 dimensions,
returned as `.embeddings`. It is an embedding-only API. Calls are sequential;
this is not a vLLM or SentenceTransformers serving integration.

## Install in a separate uv environment

Use Linux x86-64 and Python 3.12. From this example directory:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock
uv pip install --python .venv/bin/python /absolute/path/to/rebel_compiler.whl
uv pip install --python .venv/bin/python --no-deps \
  'git+https://github.com/rebel-jingu/optimum-rbln.git@a6a709ed0727f87cabfebf191b6a89b141f80731'
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 RBLN_NUM_THREADS=4
export RBLN_PROFILER=0 RBLN_RUNTIME_TIMER=0
```

`requirements.txt` lists the direct dependencies; `requirements.lock` pins the
68 public packages used by the clean validation environment.

The private SDK is installed with its dependencies and is not committed here.
CR13 requires SDK >=0.12. CA25 validation uses SDK 0.11.0.post1.
The pinned optimum feature commit contains the adapter; the public release does
not. `--no-deps` applies only to that final optimum installation: its shared
metadata pins an older Transformers/Torch stack than this model requires.
The model-specific requirements provide the newer stack used for this recipe;
other optimum model families have not been certified with it.

## Compile and inspect

```bash
uv run --no-project --python .venv/bin/python python compile.py \
  --output .cache/embeddinggemma-2 --devices 0
```

The default builds seven text buckets (32, 64, 96, 128, 288, 352, 512),
one vision bucket (630 patches), and four audio buckets (64, 128, 1056, 1536).
`--audio-layout selector` applies the constant-selection formulation to every
audio bucket. `row` selects the validated CR13 formulation.

`compile-manifest.json` is written atomically and last, after inspecting every saved `.rbln`
binary for the actual NPU target and device count. It records hashes, input
signatures, source revision, precision and toolchain versions. Existing output
directories are rejected. An interrupted directory without this manifest is not
a completed recipe build; use a fresh directory for a retry. Inference checks
the completion manifest, graph inventory and every graph hash before loading
the processor or creating runtimes.

Normally compilation creates runtimes and requires the selected NPUs to be free.
`--compile-only` skips runtime creation; it does not validate physical execution.
A smaller compile smoke bundle can be built with
`--text-buckets 128 --audio-buckets 64`; use short inputs that fit those buckets.
Static capacities are fixed at compilation and overflow is rejected at inference.

The source revision defaults to
`914f7f89142e33e77833254d9c9b90c3cef7303b`.
Access to the source checkpoint and acceptance of its Hugging Face terms may be
required in the executing user's environment.

## Reload and run

```bash
uv run --no-project --python .venv/bin/python python inference.py \
  --model .cache/embeddinggemma-2 --devices 0 --text 'What causes the northern lights?'

uv run --no-project --python .venv/bin/python python inference.py \
  --model .cache/embeddinggemma-2 --devices 0 \
  --text 'Picture <|image|> and sound <|audio|>' \
  --images /absolute/path/photo.png --audio /absolute/path/speech.flac \
  --dimensions 256 --trace .cache/mixed-host-trace.json
```

Reload uses the saved processor, token embeddings and RBLN graphs. It does not
load the original encoder checkpoint. File audio is decoded to mono and resampled
to 16 kHz. Image and audio CPU preparation can overlap for a mixed request.

`--trace` captures a separate additional request for [Perfetto](https://ui.perfetto.dev).
These are **host and SDK Runtime spans**, including transfer and possible SDK
host work. They do not measure device-internal kernel bubbles. On the measured
CA25 SDK 0.11 environment, the native device profiler aborts; zero bubbles have
not been established.

For full numerical checks, E2E timing and uninstrumented median/p95 measurements,
use the [pinned adapter's validator and benchmark](https://github.com/rebel-jingu/optimum-rbln/tree/a6a709ed0727f87cabfebf191b6a89b141f80731/examples/multimodal-embedding).
Those scripts distinguish preparation, prepared forward and E2E and exclude
trace collection from latency samples.

## Verification scope

The adapter graphs were previously validated in a standalone CR13 implementation.
This new recipe has not been physically recompiled on CR13 in this run.
CA25 validation does not establish CR13 performance or chiplet placement.

Compiled binaries, model weights, private SDK wheels, local environments and
traces are not included in this repository.
