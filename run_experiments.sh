#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

MODEL=${MODEL:-microsoft/phi-2}
EPOCH=${EPOCH:-10}
MASKING_PROB=${MASKING_PROB:-0.999}
DEEPSPEED_INCLUDE=${DEEPSPEED_INCLUDE:-localhost:0}

TASKS=(${TASKS:-RTE WIC MultiRC})
MODES=(${MODES:-entropy_gradweight_masking})
SEEDS=(${SEEDS:-28 210 10283})

for TASK in "${TASKS[@]}"; do
  for MODE in "${MODES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      if [[ -n "${LR:-}" ]]; then
        RUN_LR="${LR}"
      else
        case "${MODE}" in
          random_masking)
            RUN_LR="1e-3"
            ;;
          fft)
            RUN_LR="1e-6"
            ;;
          *)
            RUN_LR="1e-5"
            ;;
        esac
      fi

      echo "Running MODE=${MODE} TASK=${TASK} SEED=${SEED} LR=${RUN_LR}"
      MODEL="${MODEL}" \
      TASK="${TASK}" \
      EPOCH="${EPOCH}" \
      MODE="${MODE}" \
      LR="${RUN_LR}" \
      MASKING_PROB="${MASKING_PROB}" \
      DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE}" \
      SEED="${SEED}" \
      bash "${SCRIPT_DIR}/run.sh"
    done
  done
done
