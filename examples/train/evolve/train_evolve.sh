#!/usr/bin/env bash
# Train the advisor model (Qwen3.5-9B) with EvolveGenerator on problem 0.
# Frozen solver variant: uses GPT-5 via OpenAI API as the solver.
# All 8 GPUs are used for the advisor (vLLM + FSDP training).
#
# Prerequisites:
#   uv run python scripts/build_solution_pool.py --problem-id 0
#   uv run python scripts/build_training_dataset.py --problem-id 0
#
# Usage (from project root):
#   bash SkyRL/examples/train/evolve/train_evolve.sh
set -euo pipefail

# DUMP_DIR="/data"
DUMP_DIR="/data_pool"

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The path to Frontier-CS-Evolve
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

TRAIN_DATA="['$PROJECT_ROOT/data/train_p0.jsonl']"
SOLUTION_POOL_PATH="$PROJECT_ROOT/data/solution_pool_p0.json"
SNAPSHOTS_ROOT="$PROJECT_ROOT/snapshots"

RUN_NAME="evolve_p0_$(date +%Y%m%d_%H%M%S)"
CKPTS_DIR="$DUMP_DIR/rl_ckpts/$RUN_NAME"
EXPORTS_DIR="$DUMP_DIR/outputs/$RUN_NAME/exports"
export LOG_DIR="$DUMP_DIR/outputs/$RUN_NAME/logs"
ROLLOUTS_DIR="$DUMP_DIR/outputs/$RUN_NAME/rollouts"

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH="Qwen/Qwen3.5-9B"
SERVED_MODEL_NAME="Qwen3.5-9B"

# ── Infrastructure ────────────────────────────────────────────────────────────
# All 8 GPUs for the advisor vLLM + FSDP training (solver is GPT-5 via API)
ADVISOR_GPUS="0,1,2,3,4,5,6,7"
NUM_GPUS=8
TP_SIZE=1          # tensor parallel across 1 GPU
NUM_ENGINES=8      # 8 GPUs / TP=1 = 8 engines
MAX_TRAIN_SEQ_LEN=43000  # training sequence budget — bounds memory; 99%+ sequences are under 30K
LOSS_NORM_SEQ_LEN=262144 # loss normalization constant for seq_mean_token_sum_norm (not memory-related)
N_SAMPLES_PER_PROMPT=8
TRAIN_BATCH_SIZE=8
MINI_BATCH_SIZE=2

# ── Solver (frozen GPT-5 via OpenAI API) ────────────────────────────────────
SOLVER_MODEL="gpt-5.4"
SOLVER_REASONING_EFFORT="low"

# ── EvolveAgent config ───────────────────────────────────────────────────────
NUM_TURNS=1
MAX_SOLVER_CALLS=5
MAX_ADVISOR_CONTEXT_ITERS=3
LANG=cpp

# ── Dr. GRPO ─────────────────────────────────────────────────────────────────x
LOSS_REDUCTION="seq_mean_token_sum_norm"
GRPO_NORM_BY_STD=false
USE_KL_LOSS=false

cd "$PROJECT_ROOT/SkyRL"

# Load OPENAI_API_KEY, WANDB_API_KEY (and any other secrets) from .env
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

# scaleevolve lives in the project root — add it to PYTHONPATH so it's importable
# from within the SkyRL venv
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/vendor/frontier-cs-internal/src:${PYTHONPATH:-}"

# Cache directories on /data (large NVMe)
export UV_CACHE_DIR="/data/cache/uv"
export HF_HOME="/data/cache/huggingface"
# Use locally-cached model weights — avoid unauthenticated HF Hub API calls
# that fail with "file not found" for sharded models.
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR="/data/cache/triton"
export TORCH_HOME="/data/cache/torch"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_WORKSPACE_DIR="/data/cache/flashinfer"
# Pre-set VLLM_USE_V1 so that SkyRL's prepare_runtime_environment() does NOT
# set VLLM_ENABLE_V1_MULTIPROCESSING=0, which breaks the async engine core
# process spawning inside Ray actors with TP > 1.
export VLLM_USE_V1=1
# Dump infra logs to stdout for debugging (skip log redirection in actors)
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1
# Point Ray temp dir to /data to avoid filling up root filesystem
export RAY_TMPDIR="$DUMP_DIR/ray_tmp"
# Disable PyArrow's bundled jemalloc background thread — it segfaults
# (SIGSEGV at 0x350 in jemalloc_bg_thd) in multiprocessing.spawn child
# processes when running inside Ray actors.  PyArrow's jemalloc uses the
# symbol prefix "je_arrow_", so the config env var is JE_ARROW_MALLOC_CONF.
export JE_ARROW_MALLOC_CONF="background_thread:false"
# Also tell Arrow to prefer the system allocator (belt-and-suspenders).
export ARROW_DEFAULT_MEMORY_POOL=system

mkdir -p "$LOG_DIR"

PREFIX_SKYRL_PYTHON="uv run --isolated --extra fsdp --extra frontier-cs python"

# NOTE(Charlie): Remove `fsdp_config.wrap_policy.transformer_layer_cls_to_wrap` for Qwen3
# it is for Qwen3.5

CUDA_VISIBLE_DEVICES="$ADVISOR_GPUS" \
$PREFIX_SKYRL_PYTHON -m examples.train.evolve.main_evolve \
  data.train_data="$TRAIN_DATA" \
  data.val_data="['$PROJECT_ROOT/data/val_p0.jsonl']" \
  trainer.policy.model.path="$MODEL_PATH" \
  generator.inference_engine.model_dtype=bfloat16 \
  generator.inference_engine.served_model_name="$SERVED_MODEL_NAME" \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
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
  generator.inference_engine.gpu_memory_utilization=0.7 \
  generator.inference_engine.distributed_executor_backend=mp \
  generator.inference_engine.vllm_v1_disable_multiproc=false \
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
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_TRAIN_SEQ_LEN \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser=qwen3_coder \
  generator.inference_engine.engine_init_kwargs.chat_template="$PROJECT_ROOT/SkyRL/skyrl/train/utils/templates/qwen3_5_acc_thinking.jinja2" \
  generator.inference_engine.engine_init_kwargs.attention_backend=FLASH_ATTN \
  generator.inference_engine.engine_init_kwargs.language_model_only=true \
  generator.problem_id=0 \
  generator.snapshots_root="$SNAPSHOTS_ROOT" \
  generator.solution_pool_path="$SOLUTION_POOL_PATH" \
  generator.num_turns=$NUM_TURNS \
  generator.max_solver_calls=$MAX_SOLVER_CALLS \
  generator.max_advisor_context_iters=$MAX_ADVISOR_CONTEXT_ITERS \
  generator.max_advisor_code_lookups=1 \
  generator.lang=$LANG \
  generator.solver_model="$SOLVER_MODEL" \
  generator.solver_reasoning_effort="$SOLVER_REASONING_EFFORT" \
  generator.max_seq_len=$MAX_TRAIN_SEQ_LEN \
  generator.step_wise_trajectories=true \
  generator.rl_rollouts_dir="$ROLLOUTS_DIR" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.grpo_norm_by_std=$GRPO_NORM_BY_STD \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.temperature=1.0 \
  trainer.algorithm.max_seq_len=$LOSS_NORM_SEQ_LEN \
  trainer.policy.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  trainer.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.use_sample_packing=false \
  trainer.eval_before_train=false \
  trainer.export_path="$EXPORTS_DIR" \
  trainer.ckpt_path="$CKPTS_DIR" \
  trainer.log_path="$LOG_DIR" \
  trainer.logger=wandb \
  trainer.project_name="frontier-cs-evolve" \
  trainer.run_name="$RUN_NAME" \
  "$@"
