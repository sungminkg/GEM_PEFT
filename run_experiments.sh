#!/bin/bash

# batch=8로 실험!!
# RTE: batch=8

MODEL=microsoft/phi-2    # facebook/opt-125m, facebook/opt-1.3b, microsoft/deberta-v3-base
EPOCH=7
MASKING_PROB=0.999
LOCAL_HOST=0

  # for MODE in fft entropy_gradweight_masking lora bitfit adapter random_masking gradient_masking; do   gradient_ln_select_masking
  #   for SEED in 28 210 8274 10283; do

# for TASK in RTE WIC BoolQ MultiRC; do
#   for MODE in fft; do
#     for SEED in 28 10283; do 

for TASK in SST2; do
  for MODE in fft entropy_gradweight_masking lora bitfit adalora random_masking gradient_masking; do
    for SEED in 28 10283; do 

      if [ "$MODE" = "entropy_gradweight_masking" ]; then
        for LR in 1e-5 1e-4; do   # 5e-5 1e-5 5e-4 1e-4
          echo "Running $MODE | Task=$TASK | Seed=$SEED | LR=$LR"
          MODEL=$MODEL TASK=$TASK EPOCH=$EPOCH MODE=$MODE LR=$LR MASKING_PROB=$MASKING_PROB LOCAL_HOST=$LOCAL_HOST SEED=$SEED bash run2.sh
        done

      elif [ "$MODE" = "random_masking" ]; then
        for LR in 1e-3; do   # 1e-3 1e-2
          echo "Running $MODE | Task=$TASK | Seed=$SEED | LR=$LR"
          MODEL=$MODEL TASK=$TASK EPOCH=$EPOCH MODE=$MODE LR=$LR MASKING_PROB=$MASKING_PROB LOCAL_HOST=$LOCAL_HOST SEED=$SEED bash run2.sh
        done

      elif [ "$MODE" = "gradient_masking" ]; then
        for LR in 1e-5; do   # 1e-5 1e-4
          echo "Running $MODE | Task=$TASK | Seed=$SEED | LR=$LR"
          MODEL=$MODEL TASK=$TASK EPOCH=$EPOCH MODE=$MODE LR=$LR MASKING_PROB=$MASKING_PROB LOCAL_HOST=$LOCAL_HOST SEED=$SEED bash run2.sh
        done

      elif [ "$MODE" = "bitfit" ]; then
        for LR in 1e-5; do    # 1e-5 1e-4
          echo "Running $MODE | Task=$TASK | Seed=$SEED | LR=$LR"
          MODEL=$MODEL TASK=$TASK EPOCH=$EPOCH MODE=$MODE LR=$LR LOCAL_HOST=$LOCAL_HOST SEED=$SEED bash run2.sh
        done

      elif [ "$MODE" = "adapter" ]; then
        for LR in 1e-5; do  # 1e-5 1e-4
          echo "Running $MODE | Task=$TASK | Seed=$SEED | LR=$LR"
          MODEL=$MODEL TASK=$TASK EPOCH=$EPOCH MODE=$MODE LR=$LR LOCAL_HOST=$LOCAL_HOST SEED=$SEED bash run2.sh
        done

      elif [ "$MODE" = "lora" ]; then
        for LR in 1e-5; do  # 1e-5 1e-4
          echo "Running $MODE | Task=$TASK | Seed=$SEED | LR=$LR"
          MODEL=$MODEL TASK=$TASK EPOCH=$EPOCH MODE=$MODE LR=$LR LOCAL_HOST=$LOCAL_HOST SEED=$SEED bash run2.sh
        done
      
      elif [ "$MODE" = "fft" ]; then
        for LR in 1e-6 1e-5; do
          echo "Running $MODE | Task=$TASK | Seed=$SEED | LR=$LR"
          MODEL=$MODEL TASK=$TASK EPOCH=$EPOCH MODE=$MODE LR=$LR LOCAL_HOST=$LOCAL_HOST SEED=$SEED bash run2.sh
        done

      else
        for LR in 1e-5 1e-4; do
          echo "Running $MODE | Task=$TASK | Seed=$SEED | LR=$LR"
          MODEL=$MODEL TASK=$TASK EPOCH=$EPOCH MODE=$MODE LR=$LR MASKING_PROB=$MASKING_PROB LOCAL_HOST=$LOCAL_HOST SEED=$SEED bash run2.sh
        done
      fi

    done
  done
done