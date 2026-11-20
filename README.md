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
- Samples are generated with the PyTorch generator model (`en_US-libritts_r-medium.pt`), which
  runs batched on the GPU (~250 samples/s on an RTX 5070 Ti vs ~20/s for the .onnx voice on CPU).
  Running the .onnx voice with `onnxruntime-gpu` was tried and was slower than CPU.
