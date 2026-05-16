BASE_MODEL_PATH="/home/weidu/ty/GraduationProject/AIForMath/Deepseek-Prover-V2"
LORA_ADAPTER_PATH="/home/weidu/ty/GraduationProject/AIForMath/outputs/deepseekprover_v2_highschool_tactic"

PORT=11451

if [ -f "$BASE_MODEL_PATH/refs/main" ]; then
    SNAPSHOT_HASH=$(cat "$BASE_MODEL_PATH/refs/main")
    ACTUAL_MODEL_PATH="$BASE_MODEL_PATH/snapshots/$SNAPSHOT_HASH"
    echo "🔍 检测到 HuggingFace 缓存结构，自动重定向到真实路径:"
    echo "   -> $ACTUAL_MODEL_PATH"
else
    ACTUAL_MODEL_PATH="$BASE_MODEL_PATH"
fi

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=DEBUG

/home/weidu/anaconda3/envs/vllm_ty/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$ACTUAL_MODEL_PATH" \
    --tensor-parallel-size 2 \
    --distributed-executor-backend ray \
    --trust-remote-code \
    --gpu-memory-utilization 0.75 \
    --max-model-len 4096 \
    --enable-lora \
    --lora-modules my-tactic-lora="$LORA_ADAPTER_PATH" \
    --port $PORT \
    --disable-custom-all-reduce \
    --enforce-eager \
    --guided-decoding-backend lm-format-enforcer \
    --max_lora_rank 64