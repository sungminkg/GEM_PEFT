#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${SCRIPT_DIR}"

MODEL=${MODEL:-facebook/opt-1.3b}
TASK=${TASK:-RTE}
EPOCH=${EPOCH:-5}
BS=${BS:-2}
LR=${LR:-1e-5}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}

if [[ -n "${LOCAL_HOST:-}" ]]; then
  DEEPSPEED_INCLUDE=${DEEPSPEED_INCLUDE:-"localhost:${LOCAL_HOST}"}
else
  DEEPSPEED_INCLUDE=${DEEPSPEED_INCLUDE:-"localhost:0"}
fi

DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"${REPO_ROOT}/ds_config_zero2.json"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${REPO_ROOT}/outputs"}
RESULT_DIR=${RESULT_DIR:-"${REPO_ROOT}/results"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 25000-30000 -n 1)}
mkdir -p "${OUTPUT_ROOT}" "${RESULT_DIR}"
export RESULT_DIR

MODEL_NAME=${MODEL##*/}
MODE=${MODE:-entropy_gradweight_masking}
RUN_MODE_NAME="${MODE}"
EXTRA_ARGS=()

case "${MODE}" in
  lora)
    EXTRA_ARGS+=(--lora)
    ;;
  adapter)
    EXTRA_ARGS+=(--adapter)
    ;;
  random_masking)
    MASKING_PROB=${MASKING_PROB:-0.0}
    RUN_MODE_NAME="random_masking_${MASKING_PROB}"
    EXTRA_ARGS+=(--random_masking --masking_prob "${MASKING_PROB}")
    ;;
  bitfit)
    EXTRA_ARGS+=(--bitfit)
    ;;
  gradient_masking)
    MASKING_PROB=${MASKING_PROB:-0.0}
    RUN_MODE_NAME="gradient_masking_${MASKING_PROB}"
    EXTRA_ARGS+=(--gradient_masking --masking_prob "${MASKING_PROB}")
    ;;
  gradweight_masking)
    MASKING_PROB=${MASKING_PROB:-0.0}
    RUN_MODE_NAME="gradweight_masking_${MASKING_PROB}"
    EXTRA_ARGS+=(--gradweight_masking --masking_prob "${MASKING_PROB}")
    ;;
  entropy_gradweight_masking)
    MASKING_PROB=${MASKING_PROB:-0.0}
    RUN_MODE_NAME="entropy_gradweight_masking_${MASKING_PROB}"
    # Historical run name kept for compatibility. This mode is GEM.
    EXTRA_ARGS+=(--entropy_gradweight_masking --masking_prob "${MASKING_PROB}")
    ;;
  fft)
    EXTRA_ARGS+=(--fft)
    ;;
  *)
    echo "Unsupported MODE: ${MODE}" >&2
    exit 1
    ;;
esac

OLD_BS=${BS}
TASK_ARGS=()

case "${TASK}" in
  MultiRC)
    BS=2
    GA=$((OLD_BS / BS))
    TASK_ARGS+=(--gradient_accumulation_steps "${GA}")
    ;;
esac

TRAIN_ARGS=()
case "${TASK}" in
  RTE|SST2|WIC|BoolQ|MultiRC|Copa)
    TRAIN_ARGS+=(--train_as_classification)
    ;;
esac

TAG=${TAG:-"${RUN_MODE_NAME}-${LR}-${SEED}"}
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/${TASK}-${MODEL_NAME}-${TAG}"}

deepspeed \
  --master_port "${MASTER_PORT}" \
  --include "${DEEPSPEED_INCLUDE}" \
  "${REPO_ROOT}/run.py" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --overwrite_output_dir \
  --model_name "${MODEL}" \
  --task_name "${TASK}" \
  --output_dir "${OUTPUT_DIR}" \
  --tag "${TAG}" \
  --train_set_seed "${SEED}" \
  --num_train "${TRAIN}" \
  --num_dev "${DEV}" \
  --logging_steps 10 \
  --learning_rate "${LR}" \
  --num_train_epochs "${EPOCH}" \
  --per_device_train_batch_size "${BS}" \
  --load_best_model_at_end \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 1 \
  "${TRAIN_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "${TASK_ARGS[@]}" \
  "$@"
