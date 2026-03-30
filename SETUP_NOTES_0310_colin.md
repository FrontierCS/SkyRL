# Running SkyRL on hyperbolic-14

Machine: `hyperbolic-14` — 8x NVIDIA B200 (183GB each), NFS home at `/mnt/nfs/`, fast local storage at `/data/users/zwcolin/`.

## Key Issues & Solutions

### 1. NFS is too slow for everything

`/mnt/nfs/` has severe performance issues for I/O-heavy workloads (venv creation, package installs, model downloads). All project files, caches, and virtual environments should live on `/data/users/zwcolin/`.

**Project location:** `/data/users/zwcolin/med_agent/`

### 2. Cache directories must point to `/data`

All cache env vars in `~/.bashrc` are togglable via a `CACHE_STORAGE` variable:

```bash
# In ~/.bashrc — set to "local" for /data, "nfs" for /mnt
CACHE_STORAGE="local"
```

This controls `UV_CACHE_DIR`, `HF_HOME`, `TRANSFORMERS_CACHE`, `TMPDIR`, `TRITON_CACHE_DIR`, `FLASHINFER_CACHE_DIR`, and others. After changing, run `source ~/.bashrc`.

**Important:** New shell sessions spawned by Cursor do NOT source `.bashrc`. You must explicitly export all cache vars in any training script or command. See the `train_evolve.sh` zwcolin block for a complete list.

### 3. NFS does not support Unix domain sockets (Ray crash)

Ray creates Unix domain sockets in `TMPDIR`. If `TMPDIR` points to NFS, Ray crashes with:

```
Failed to connect to socket at address:/mnt/nfs/home/zwcolin/.cache_tmp/tmp/ray/session_.../sockets/raylet
```

**Fix:** `TMPDIR` and `RAY_TMPDIR` must point to a local filesystem:
```bash
export TMPDIR=/data/users/zwcolin/.cache/tmp
export RAY_TMPDIR=/data/users/zwcolin/.cache/tmp
```

### 4. B200 GPUs require CUDA 13.0 (cu130) stack

B200 (Blackwell) GPUs have new tensor core architectures. PyTorch/vLLM wheels compiled with CUDA 12.8 (cu128) cause cuBLAS errors:

```
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling cublasGemmEx(...)
```

This manifests during vLLM's model profile run at `qwen3_5.py:164 in_proj_qkvz(hidden_states)`.

**Fix:** All CUDA packages must use cu130 builds. In `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[[tool.uv.index]]
name = "flashinfer-cu130"
url = "https://flashinfer.ai/whl/cu130"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cu130", marker = "sys_platform == 'linux'" },
    { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
]
flash-attn = { url = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu130torch2.10-cp312-cp312-linux_x86_64.whl", marker = "sys_platform == 'linux'" }
flashinfer-jit-cache = { index = "flashinfer-cu130", marker = "sys_platform == 'linux'" }
```

Verify after install:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# Expected: 2.10.0+cu130 13.0
```

### 5. Other users' GPU processes

Other users may have long-running processes (e.g. `sglang::scheduler`) occupying GPU memory. Check with `nvidia-smi` before running. These processes are owned by other users and require `sudo` to kill:

```bash
sudo pkill -9 -f sglang
```

### 6. Script defaults point to NFS `$HOME`

`run_gsm8k.sh` and `run_gsm8k_qwen3_5.sh` default several paths to `$HOME/...` which resolves to `/mnt/nfs/home/zwcolin/`. Override them:

| Variable | Default (NFS) | Override (local) |
|---|---|---|
| `DATA_DIR` | `$HOME/data/gsm8k` | `/data/users/zwcolin/med_agent/data/gsm8k` |
| `trainer.ckpt_path` | `$HOME/ckpts/...` | `/data/users/zwcolin/med_agent/ckpts/...` |
| `trainer.export_path` | `$HOME/exports/` | `/data/users/zwcolin/med_agent/exports/` |

### 7. Wandb not configured

The script defaults to `LOGGER=wandb`, which requires `WANDB_API_KEY`. For quick testing, override with `LOGGER=console`.

---

## Qwen3.5 Support

### Required dependency changes (`pyproject.toml`)

The cherry-picked `update` commit bumps versions for Qwen3.5 compatibility. The full set of changes:

| Package | Before | After | Reason |
|---------|--------|-------|--------|
| vllm | 0.16.0 | 0.17.0 | Native Qwen3.5 model support |
| torch | 2.9.1 | 2.10.0 | Required by vllm 0.17.0 |
| transformers | >=4.56.1,<5 | >=5.3.0 | Native `qwen3_5` model type for FSDP |
| flashinfer-python | 0.6.3 | 0.6.4 | Required by vllm 0.17.0 |
| flashinfer-jit-cache | 0.6.3 | 0.6.4 | Matches flashinfer-python |
| accelerate | (unpinned) | >=1.13.0 | Fixes `_is_hf_initialized` TypeError with transformers 5.x |
| requires-python | >=3.11 | >=3.12,<3.13 | flash-attn wheel is cp312-only |

Both `fsdp` AND `megatron` extras must be updated (they each independently pin torch, vllm, flashinfer).

The `flashrl` extra also pins torch — must be bumped to 2.10.0 to match the cu130 index.

An override is needed to force transformers >=5.3.0 past vllm's cap:
```toml
# In override-dependencies:
"transformers>=5.3.0",
```

### flash-attn pre-built wheel

flash-attn has no official wheel for torch 2.10.0. Use community wheels from [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels):

```toml
# In [tool.uv.sources]:
flash-attn = { url = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu130torch2.10-cp312-cp312-linux_x86_64.whl", marker = "sys_platform == 'linux'" }
```

### flash-linear-attention (fla) — fast path for Qwen3.5

Qwen3.5 has hybrid attention: standard attention layers + **gated delta rule** linear attention layers. Without fast kernels, transformers falls back to pure-Python loops that are ~10x slower.

The fast path requires both:
1. **`flash-linear-attention`** (fla) — Triton kernels for `chunk_gated_delta_rule`
2. **`causal-conv1d`** — CUDA kernels for causal 1D convolution

**Installing flash-linear-attention:**
```bash
# Pure Python/Triton — installs cleanly
uv pip install flash-linear-attention
```

Add to `fsdp` extra in pyproject.toml:
```toml
"flash-linear-attention>=0.4.1; sys_platform == 'linux'",
```

**Installing causal-conv1d:**

`causal-conv1d` requires CUDA compilation (~8 min build). It was originally blocked by a `sys_platform == 'never'` override in `pyproject.toml` to suppress transitive resolution issues.

To enable it:
1. Change the override from `"causal-conv1d; sys_platform == 'never'"` to `"causal-conv1d>=1.4.0; sys_platform == 'linux'"`
2. Add to `fsdp` extra: `"causal-conv1d>=1.4.0; sys_platform == 'linux'"`
3. Run `uv lock && uv sync --extra fsdp` (the CUDA build takes ~8 minutes)

Verify fast path:
```python
python -c "from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available; print(is_fast_path_available)"
# Expected: True
```

The `model_wrapper.py` will also log:
```
INFO | Qwen3.5 fast path is ENABLED (causal-conv1d + flash-linear-attention)
```

### Code patches (already applied in cherry-pick)

1. **`return_dict=False`** on `apply_chat_template(tokenize=True)` calls — transformers 5.x returns `BatchEncoding` instead of `list[int]`
2. **`vllm_worker.py`** — weight sync layer naming fix for `ForConditionalGeneration` → adds `language_model.` prefix
3. **`model_wrapper.py`** — Qwen3.5 monkey-patches:
   - 3D `position_ids` fix (upstream: huggingface/transformers#44399)
   - CPU tensor creation fix for `torch_chunk_gated_delta_rule` (commented out since fast path avoids it)

### Debug `import fla` in `main_base.py`

The cherry-picked commit left a debug `import fla` at the top of `main_base.py`. This was removed since `fla` is optional and causes `ModuleNotFoundError` if not installed.

---

## Running GSM8K with Qwen3.5

```bash
cd /data/users/zwcolin/med_agent/radiology/SkyRL

# Prepare data (one-time)
uv run python examples/train/gsm8k/gsm8k_dataset.py \
  --output_dir /data/users/zwcolin/med_agent/data/gsm8k

# Run training (all env vars explicit — don't rely on .bashrc)
export UV_CACHE_DIR=/data/users/zwcolin/.cache/uv
export HF_HOME=/data/users/zwcolin/.cache/huggingface
export TMPDIR=/data/users/zwcolin/.cache/tmp
export RAY_TMPDIR=/data/users/zwcolin/.cache/tmp
export TRITON_CACHE_DIR=/data/users/zwcolin/.cache/triton
export FLASHINFER_CACHE_DIR=/data/users/zwcolin/.cache/flashinfer
export TORCH_EXTENSIONS_DIR=/data/users/zwcolin/.cache/torch_extensions
export DEEP_GEMM_CACHE_DIR=/data/users/zwcolin/.cache/deep_gemm

DATA_DIR=/data/users/zwcolin/med_agent/data/gsm8k \
NUM_GPUS=8 \
LOGGER=console \
VLLM_USE_V1=1 \
bash examples/train/gsm8k/run_gsm8k_qwen3_5.sh \
  trainer.ckpt_path=/data/users/zwcolin/med_agent/ckpts/gsm8k_qwen3.5_2B_ckpt \
  trainer.epochs=1
```

Key differences from the Qwen2.5 `run_gsm8k.sh`:
- Uses `Qwen/Qwen3.5-2B` model
- Sets `Qwen3_5DecoderLayer` for FSDP wrap policy
- Sets `language_model_only=true` (Qwen3.5 uses VL wrapper architecture in vLLM)
- `VLLM_USE_V1=1` works with cu130 stack (was broken with cu128)

---

## Running Evolve Training

The `train_evolve.sh` script has user-specific env var blocks. The zwcolin block (lines 90-105) exports all cache dirs to `/data/users/zwcolin/.cache/...`.

```bash
cd /data/users/zwcolin/med_agent/radiology
bash SkyRL/examples/train/evolve/train_evolve.sh
```

The script:
1. Starts a frozen solver vLLM server on GPUs 4-7
2. Waits for it to be healthy (up to 3 min)
3. Runs advisor FSDP training on GPUs 0-3 with colocated vLLM

---

## Earlier Results (Qwen2.5-1.5B GSM8K)

Training ran successfully with GRPO on Qwen2.5-1.5B-Instruct:

| Metric | Step 1 | Step 2 |
|---|---|---|
| `reward/avg_raw_reward` | 0.137 | 0.193 |
| `reward/avg_pass_at_2` | 0.250 | 0.345 |
| `policy/policy_kl` | 0.0004 | 0.0029 |
| `timing/step` | 46.5s | 22.7s |

Eval at step 0 (pre-training baseline): `pass_at_1 = 0.091`.
