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
- Had to install tensorflow~=2.16.0 for training step to work, otherwise it failed silently with
  status code 139 due to CUDA library issues.
