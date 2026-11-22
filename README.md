# Micro Wake Word Model: Okay GLaDOS

## Usage

1. Fetch all submodules:
   ```
   git submodule update --init --recursive`
   ```

2. Install all python dependencies:
   ```
   uv sync
   ```

3. Create a Jupyter kernel:
   ```
   uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=micro_wake_word_training
   ```
3. Execute the steps in the notebook.
4. Profit.
   

## Near-miss negatives

Everyday phrases that sound a bit like the wake word ("okay thanks", "okay class", "glad to hear it")
are trained on as negatives, so the model doesn't learn "okay" plus anything.

- `near_miss_phrases.txt` is what the notebook trains on.
- `near_miss_holdout.txt` is never trained on. `uv run python tools/near_miss_check.py` runs the
  deployed model and the latest run over it and reports how often each one fires, by group. It
  uses synthetic speech only, so treat it as a comparison between models, not a real-world false
  positive rate. The script refuses to run if the two files share a phrase.

## Voice models

Wake word and near-miss clips come from the `voice_models` list in the notebook. Each entry has
a unique `name` (its folder under `data/generated_samples/`), a `type` (`piper` or `kokoro`) and a
`samples` count. The same `type` can be used more than once, for example two Piper voices. Piper (VITS) and Kokoro-82M (StyleTTS2) are
different architectures, so the model doesn't learn one engine's voice artifacts.

## Notes

**Here are a few random notes on how I got it to work:**

- Ran in WSL v2
- Had to install CUDA in WSL (Microsoft has documentation on this)
- TensorFlow doesn't find the pip-installed `libcusolver` on its own, so the training cell prepends
  `.venv/.../nvidia/cusolver/lib` to `LD_LIBRARY_PATH`. Without it, TF silently falls back to CPU.
- RTX 50-series (compute capability 12.0) isn't in TensorFlow's prebuilt kernels, so CUDA kernels
  are JIT-compiled from PTX the first time. Expect a slow first run; later runs use the cache in `~/.nv`.
- piper-sample-generator runs in its own uv project, [tools/piper](tools/piper), because it pins
  `audiomentations==0.33.0`, which conflicts with microWakeWord. It's pinned to the commit from
  upstream PR #21, because the v3.2.0 release imports `piper_train` without packaging it.
