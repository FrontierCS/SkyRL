#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
SKYRL_DIR="$PROJECT_ROOT/SkyRL"
DUMP_DIR="/data"
LOG_BASE="$DUMP_DIR/outputs/full_ctx_search"
mkdir -p "$LOG_BASE"
RESULTS_FILE="$LOG_BASE/results.csv"
echo "context_len,status" > "$RESULTS_FILE"

# Env
export UV_CACHE_DIR="/data/cache/uv" HF_HOME="/data/cache/huggingface" HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR="/data/cache/triton" TORCH_HOME="/data/cache/torch"
export FLASHINFER_DISABLE_VERSION_CHECK=1 FLASHINFER_WORKSPACE_DIR="/data/cache/flashinfer"
export VLLM_USE_V1=1 SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1 RAY_TMPDIR="/data/ray_tmp"
export JE_ARROW_MALLOC_CONF="background_thread:false" ARROW_DEFAULT_MEMORY_POOL=system
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/vendor/frontier-cs-internal/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
[ -f "$PROJECT_ROOT/.env" ] && { set -a; source "$PROJECT_ROOT/.env"; set +a; }

PY="uv run --isolated --extra fsdp --extra frontier-cs python"

run_test() {
    local CTX=$1 GEN=$(($1 - 1024))
    export LOG_DIR="$LOG_BASE/logs_${CTX}"; mkdir -p "$LOG_DIR"
    echo "=== Testing context=$CTX (prompt=1024 gen=$GEN) ==="
    cd "$SKYRL_DIR"
    set +e
    timeout 600 $PY -m examples.train_scripts.full_context.main_full_ctx \
      data.train_data="['$PROJECT_ROOT/data/train_p0.jsonl']" \
      data.val_data="['$PROJECT_ROOT/data/val_p0.jsonl']" \
      trainer.policy.model.path=Qwen/Qwen3.5-9B \
      generator.inference_engine.model_dtype=bfloat16 \
      generator.inference_engine.served_model_name=Qwen3.5-9B \
      generator.inference_engine.num_engines=8 \
      generator.inference_engine.tensor_parallel_size=1 \
      generator.inference_engine.pipeline_parallel_size=1 \
      generator.inference_engine.expert_parallel_size=1 \
      generator.inference_engine.data_parallel_size=1 \
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
      generator.inference_engine.engine_init_kwargs.max_model_len=65536 \
      generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
      generator.inference_engine.engine_init_kwargs.attention_backend=FLASH_ATTN \
      generator.inference_engine.engine_init_kwargs.language_model_only=true \
      generator.n_samples_per_prompt=8 \
      generator.sampling_params.max_generate_length=$GEN \
      generator.sampling_params.temperature=0.6 \
      generator.sampling_params.top_p=0.95 \
      generator.sampling_params.top_k=20 \
      generator.sampling_params.min_p=0.0 \
      generator.sampling_params.repetition_penalty=1.1 \
      generator.sampling_params.presence_penalty=0.6 \
      trainer.algorithm.advantage_estimator=grpo \
      trainer.algorithm.loss_reduction=seq_mean_token_sum_norm \
      trainer.algorithm.grpo_norm_by_std=false \
      trainer.algorithm.use_kl_loss=false \
      trainer.algorithm.temperature=1.0 \
      trainer.algorithm.max_seq_len=262144 \
      trainer.policy.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
      trainer.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
      trainer.placement.colocate_all=true \
      trainer.strategy=fsdp2 \
      trainer.placement.policy_num_nodes=1 \
      trainer.placement.ref_num_nodes=1 \
      trainer.placement.policy_num_gpus_per_node=8 \
      trainer.placement.ref_num_gpus_per_node=8 \
      trainer.train_batch_size=8 \
      trainer.policy_mini_batch_size=2 \
      trainer.micro_train_batch_size_per_gpu=1 \
      trainer.micro_forward_batch_size_per_gpu=1 \
      trainer.use_sample_packing=false \
      trainer.eval_before_train=false \
      trainer.max_prompt_length=1024 \
      trainer.logger=console \
      trainer.project_name=full_ctx_search \
      trainer.run_name="ctx_${CTX}" \
      trainer.num_dummy_steps=1 \
      trainer.ckpt_path="$LOG_BASE/ckpts_${CTX}" \
      trainer.export_path="$LOG_BASE/exports_${CTX}" \
      trainer.log_path="$LOG_DIR" \
      2>&1 | tee "$LOG_BASE/ctx_${CTX}.log"
    local rc=$?; set -e
    if [ $rc -eq 0 ]; then
        echo "$CTX,PASS" >> "$RESULTS_FILE"; echo ">>> PASS ctx=$CTX"; return 0
    else
        echo "$CTX,OOM" >> "$RESULTS_FILE"; echo ">>> FAIL ctx=$CTX"; return 1
    fi
}

LO=32768; HI=200000; BEST=0
echo "Binary search [$LO, $HI] for Qwen3.5-9B on 8xH100 FSDP2 micro_batch=1"

while [ $((HI - LO)) -gt 4096 ]; do
    MID=$(( ((LO + HI) / 2 / 1024) * 1024 ))
    echo ">>> LO=$LO MID=$MID HI=$HI"
    if run_test $MID; then BEST=$MID; LO=$MID; else HI=$MID; fi
    ray stop --force 2>/dev/null || true; sleep 5
done

echo "=== DONE: max context=$BEST tokens ==="
cat "$RESULTS_FILE"
