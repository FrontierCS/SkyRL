#!/usr/bin/env bash
# Train the advisor model (Qwen3.5-9B) with EvolveGenerator on problem 0.
#
# Prerequisites:
#   uv run python scripts/build_solution_pool.py --problem-id 0
#   uv run python scripts/build_training_dataset.py --problem-id 0
#
# Usage (from project root):
#   bash SkyRL/examples/train/evolve/train_evolve.sh
set -euo pipefail

# Charlie's dump directory
# DUMP_DIR="/mnt/local_storage"
DUMP_DIR="/data/qmang"

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The path to Frontier-CS-Evolve
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

TRAIN_DATA="['$PROJECT_ROOT/data/train_p0.jsonl']"
SOLUTION_POOL_PATH="$PROJECT_ROOT/data/solution_pool_p0.json"
SNAPSHOTS_ROOT="$PROJECT_ROOT/snapshots"

RUN_NAME="evolve_p0_$(date +%Y%m%d_%H%M%S)"
CKPTS_DIR="$DUMP_DIR/outputs/rl_training/$RUN_NAME/ckpts"
EXPORTS_DIR="$DUMP_DIR/outputs/rl_training/$RUN_NAME/exports"
export LOG_DIR="$DUMP_DIR/outputs/rl_training/$RUN_NAME/logs"
ROLLOUTS_DIR="$DUMP_DIR/outputs/rl_training/$RUN_NAME/rollouts"

# ── Model ────────────────────────────────────────────────────────────────────
# MODEL_PATH="Qwen/Qwen3-4B"
# SERVED_MODEL_NAME="Qwen3-4B"
MODEL_PATH="/data/qmang/hf_cache/hub/models--Qwen--Qwen3.5-9B"
SERVED_MODEL_NAME="Qwen3.5-9B"

# ── Infrastructure ───────────────────────────────────────────────────────────
# GPU layout: GPU 0 → advisor vLLM + FSDP training (colocated)
#             GPUs 1-3 → frozen solver vLLM
ADVISOR_GPUS="0"
SOLVER_GPUS="1,2,3"
NUM_GPUS=1           # advisor + training use 1 GPU
SOLVER_NUM_GPUS=3    # solver uses 3 GPUs (data parallel)
MAX_MODEL_LEN=262144
# MAX_MODEL_LEN=32000  # For Qwen3
N_SAMPLES_PER_PROMPT=2
MINI_BATCH_SIZE=1    # must be a multiple of N_SAMPLES_PER_PROMPT

# ── Solver (frozen) vLLM server ───────────────────────────────────────────────
SOLVER_PORT=8001
SOLVER_BASE_URL="http://127.0.0.1:${SOLVER_PORT}/v1"

# ── EvolveAgent config ───────────────────────────────────────────────────────
NUM_TURNS=2
MAX_SOLVER_CALLS=5
MAX_ADVISOR_CONTEXT_ITERS=10
LANG=cpp

# ── Dr. GRPO ─────────────────────────────────────────────────────────────────
LOSS_REDUCTION="seq_mean_token_sum_norm"
GRPO_NORM_BY_STD=false
USE_KL_LOSS=false

cd "$PROJECT_ROOT/SkyRL"

# scaleevolve lives in the project root — add it to PYTHONPATH so it's importable
# from within the SkyRL venv
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/vendor/frontier-cs-internal/src:${PYTHONPATH:-}"

# Export the PYTHONPATH to the SkyRL venv (needed by Charlie, not QMANG)
# export SKYRL_PYTHONPATH_EXPORT=1 

# QMANG's environment variables
export UV_CACHE_DIR="/data/qmang/uv_cache"
export UV_PROJECT_ENVIRONMENT="/data/qmang/Frontier-CS-Evolve-venv/skyrl-train"
export HF_HOME="/data/qmang/hf_cache"
export TRITON_CACHE_DIR="/data/qmang/triton_cache"
export TORCH_HOME="/data/qmang/torch_cache"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_WORKSPACE_DIR="/data/qmang/flashinfer_cache"

/data/qmang/Frontier-CS-Evolve-venv/skyrl-train/bin/python -c "import vllm; print(f'vllm version: {vllm.__version__}')"

# ── Start frozen solver vLLM server ──────────────────────────────────────────
mkdir -p "$LOG_DIR"
echo "Starting frozen solver vLLM on port ${SOLVER_PORT} (GPUs ${SOLVER_GPUS})..."

# QMANG's vllm serve command
PREFIX_VLLM_SERVE="HF_HUB_OFFLINE=1 FLASHINFER_WORKSPACE_DIR=/data/qmang/flashinfer_cache PATH=/data/qmang/.venv/bin:$PATH /data/qmang/.venv/bin/vllm serve"

# Charlie's vllm serve command
# PREFIX_VLLM_SERVE="uv run --isolated --extra fsdp vllm serve"

# NOTE(Charlie): Remove --language-model-only \ for Qwen3 / lower vllm version
CUDA_VISIBLE_DEVICES="$SOLVER_GPUS" $PREFIX_VLLM_SERVE \
    "$MODEL_PATH" \
    --port "$SOLVER_PORT" \
    --host 127.0.0.1 \
    -dp "$SOLVER_NUM_GPUS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --served-model-name "$SERVED_MODEL_NAME" \
    --language-model-only \
    --gpu-memory-utilization 0.8 \
    --attention-backend FLASH_ATTN \
    >> "$LOG_DIR/solver-vllm.log" 2>&1 &
SOLVER_PID=$!

# Kill solver on exit
trap 'echo "Stopping solver vLLM (pid $SOLVER_PID)..."; kill "$SOLVER_PID" 2>/dev/null; wait "$SOLVER_PID" 2>/dev/null' EXIT

# Wait for solver server to be ready (up to 3 minutes)
echo "Waiting for solver vLLM server to be ready..."
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${SOLVER_PORT}/health" > /dev/null 2>&1; then
    echo "Solver vLLM ready after ${i}s"
    break
  fi
  if ! kill -0 "$SOLVER_PID" 2>/dev/null; then
    echo "ERROR: Solver vLLM process died. Check $LOG_DIR/solver-vllm.log"
    exit 1
  fi
  sleep 1
done

# QMANG's python command
PREFIX_SKYRL_PYTHON="/data/qmang/Frontier-CS-Evolve-venv/skyrl-train/bin/python"
# Charlie's python command
# PREFIX_SKYRL_PYTHON="uv run --isolated --extra fsdp --extra frontier-cs python"

# NOTE(Charlie): Remove `fsdp_config.wrap_policy.transformer_layer_cls_to_wrap` for Qwen3
# it is for Qwen3.5

CUDA_VISIBLE_DEVICES="$ADVISOR_GPUS" \
$PREFIX_SKYRL_PYTHON -m examples.train.evolve.main_evolve \
  data.train_data="$TRAIN_DATA" \
  data.val_data="[]" \
  trainer.policy.model.path="$MODEL_PATH" \
  generator.inference_engine.model_dtype=bfloat16 \
  generator.inference_engine.served_model_name="$SERVED_MODEL_NAME" \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.pipeline_parallel_size=1 \
  generator.inference_engine.expert_parallel_size=1 \
  generator.inference_engine.data_parallel_size=1 \
  generator.inference_engine.enable_http_endpoint=true \
  generator.inference_engine.http_endpoint_host="127.0.0.1" \
  generator.inference_engine.http_endpoint_port=8002 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.vllm_v1_disable_multiproc=true \
  generator.inference_engine.enable_prefix_caching=true \
  generator.inference_engine.enable_chunked_prefill=true \
  generator.inference_engine.max_num_batched_tokens=8192 \
  generator.inference_engine.max_num_seqs=1024 \
  generator.inference_engine.enforce_eager=true \
  generator.inference_engine.fully_sharded_loras=false \
  generator.inference_engine.enable_ray_prometheus_stats=false \
  generator.inference_engine.override_existing_update_group=auto \
  generator.inference_engine.weight_transfer_threshold_cuda_ipc_GB=1.0 \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.sampling_params.max_generate_length=32768 \
  generator.sampling_params.temperature=0.6 \
  generator.sampling_params.top_p=0.95 \
  generator.sampling_params.top_k=20 \
  generator.sampling_params.min_p=0.0 \
  generator.sampling_params.repetition_penalty=1.1 \
  generator.sampling_params.presence_penalty=0.6 \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser=qwen3_coder \
  generator.inference_engine.engine_init_kwargs.reasoning_parser=qwen3 \
  generator.inference_engine.engine_init_kwargs.attention_backend=FLASH_ATTN \
  generator.problem_id=0 \
  generator.snapshots_root="$SNAPSHOTS_ROOT" \
  generator.solution_pool_path="$SOLUTION_POOL_PATH" \
  generator.num_turns=$NUM_TURNS \
  generator.max_solver_calls=$MAX_SOLVER_CALLS \
  generator.max_advisor_context_iters=$MAX_ADVISOR_CONTEXT_ITERS \
  generator.lang=$LANG \
  generator.max_seq_len=$MAX_MODEL_LEN \
  generator.solver_base_url="$SOLVER_BASE_URL" \
  generator.rl_rollouts_dir="$ROLLOUTS_DIR" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.grpo_norm_by_std=$GRPO_NORM_BY_STD \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.temperature=1.0 \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.policy.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  trainer.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.train_batch_size=$MINI_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.eval_before_train=false \
  trainer.export_path="$EXPORTS_DIR" \
  trainer.ckpt_path="$CKPTS_DIR" \
  trainer.log_path="$LOG_DIR" \
  trainer.logger=console \
  "$@"
