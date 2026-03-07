#!/usr/bin/env bash
# Train the advisor model (Qwen3.5-27B) with EvolveGenerator on problem 0.
#
# Prerequisites:
#   uv run python scripts/build_solution_pool.py --problem-id 0
#   uv run python scripts/build_training_dataset.py --problem-id 0
#
# Usage (from project root):
#   bash SkyRL/skyrl-train/examples/evolve/train_evolve.sh
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

TRAIN_DATA="['$PROJECT_ROOT/data/train_p0.jsonl']"
SOLUTION_POOL_PATH="$PROJECT_ROOT/data/solution_pool_p0.json"
SNAPSHOTS_ROOT="$PROJECT_ROOT/snapshots"

RUN_NAME="evolve_p0_$(date +%Y%m%d_%H%M%S)"
CKPTS_DIR="$PROJECT_ROOT/outputs/rl_training/$RUN_NAME/ckpts"
EXPORTS_DIR="$PROJECT_ROOT/outputs/rl_training/$RUN_NAME/exports"
LOG_DIR="/tmp/skyrl-logs/$RUN_NAME"

CHAT_TEMPLATE_PATH="$SCRIPT_DIR/../../skyrl_train/utils/templates/qwen3_acc_thinking.jinja2"

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH="/data/qmang/hf_cache/hub/models--Qwen--Qwen3.5-9B"
SERVED_MODEL_NAME="Qwen3.5-9B"

# ── Infrastructure ───────────────────────────────────────────────────────────
NUM_GPUS=4
MAX_MODEL_LEN=32768
N_SAMPLES_PER_PROMPT=8
MINI_BATCH_SIZE=8    # must be a multiple of N_SAMPLES_PER_PROMPT

# ── EvolveAgent config ───────────────────────────────────────────────────────
NUM_TURNS=5
MAX_SOLVER_CALLS=6
MAX_ADVISOR_CONTEXT_ITERS=10
LANG=cpp

# ── Dr. GRPO ─────────────────────────────────────────────────────────────────
LOSS_REDUCTION="seq_mean_token_sum_norm"
GRPO_NORM_BY_STD=false
USE_KL_LOSS=false

cd "$PROJECT_ROOT/SkyRL/skyrl-train"

# scaleevolve lives in the project root — add it to PYTHONPATH so it's importable
# from within the SkyRL venv
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export UV_CACHE_DIR="/data/qmang/uv_cache"

uv run --extra vllm -m examples.evolve.main_evolve \
  data.train_data="$TRAIN_DATA" \
  trainer.policy.model.path="$MODEL_PATH" \
  generator.served_model_name="$SERVED_MODEL_NAME" \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.enable_http_endpoint=true \
  generator.http_endpoint_host="127.0.0.1" \
  generator.http_endpoint_port=8002 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.gpu_memory_utilization=0.8 \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.sampling_params.max_generate_length=8192 \
  +generator.engine_init_kwargs.chat_template="$CHAT_TEMPLATE_PATH" \
  +generator.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  +generator.engine_init_kwargs.enable_log_requests=false \
  +generator.engine_init_kwargs.enable_auto_tool_choice=true \
  +generator.engine_init_kwargs.tool_call_parser=qwen3_coder \
  +generator.engine_init_kwargs.language_model_only=true \
  +generator.engine_init_kwargs.attention_backend=FLASH_ATTN \
  +evolve.problem_id=0 \
  +evolve.snapshots_root="$SNAPSHOTS_ROOT" \
  +evolve.solution_pool_path="$SOLUTION_POOL_PATH" \
  +evolve.num_turns=$NUM_TURNS \
  +evolve.max_solver_calls=$MAX_SOLVER_CALLS \
  +evolve.max_advisor_context_iters=$MAX_ADVISOR_CONTEXT_ITERS \
  +evolve.lang=$LANG \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.grpo_norm_by_std=$GRPO_NORM_BY_STD \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.train_batch_size=$MINI_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.export_path="$EXPORTS_DIR" \
  trainer.ckpt_path="$CKPTS_DIR" \
  trainer.log_path="$LOG_DIR" \
  trainer.logger=console \
  "$@"
