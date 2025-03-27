# export WANDB_MODE=online
# export WANDB_PROJECT="PEFT_Masking"
# export WANDB_MODE="online"
# export WANDB_ENTITY="kangsung"
# export WANDB_NAME="$TASK-${MODEL_NAME}-$TAG"

MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

EPOCH=${EPOCH:-5}
BS=${BS:-8}
LR=${LR:-1e-5}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
LOCAL_HOST=${LOCAL_HOST:-3}
DS_CONFIG=${DS_CONFIG:-"ds_config_zero2.json"}

MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "lora" ]; then
  EXTRA_ARGS="--lora"
elif [ "$MODE" == "adapter" ]; then
  EXTRA_ARGS="--adapter"
elif [ "$MODE" == "random_masking" ]; then
  MASKING_PROB=${MASKING_PROB:-0.0}
  MODE="random_masking_${MASKING_PROB}"
  EXTRA_ARGS="--random_masking --masking_prob $MASKING_PROB"
elif [ "$MODE" == "bitfit" ]; then
  EXTRA_ARGS="--bitfit"
elif [ "$MODE" == "adalora" ]; then
  EXTRA_ARGS="--adalora"
elif [ "$MODE" == "gradient_masking" ]; then
  MASKING_PROB=${MASKING_PROB:-0.0}
  MODE="gradient_masking_${MASKING_PROB}"
  EXTRA_ARGS="--gradient_masking --masking_prob $MASKING_PROB"
elif [ "$MODE" == "gradweight_masking" ]; then
  MASKING_PROB=${MASKING_PROB:-0.0}
  MODE="gradweight_masking_${MASKING_PROB}"
  EXTRA_ARGS="--gradweight_masking --masking_prob $MASKING_PROB"
elif [ "$MODE" == "gradweight_ln_masking" ]; then
  MASKING_PROB=${MASKING_PROB:-0.0}
  MODE="gradweight_ln_masking_${MASKING_PROB}"
  EXTRA_ARGS="--gradweight_ln_masking --masking_prob $MASKING_PROB"
elif [ "$MODE" == "gradweight_select_masking" ]; then
  MASKING_PROB=${MASKING_PROB:-0.0}
  MODE="gradweight_select_masking_${MASKING_PROB}"
  EXTRA_ARGS="--gradweight_select_masking --masking_prob $MASKING_PROB"
elif [ "$MODE" == "gradweight_ln_select_masking" ]; then
  MASKING_PROB=${MASKING_PROB:-0.0}
  MODE="gradweight_ln_select_masking_${MASKING_PROB}"
  EXTRA_ARGS="--gradweight_ln_select_masking --masking_prob $MASKING_PROB"
elif [ "$MODE" == "entropy_gradient_masking" ]; then
  MASKING_PROB=${MASKING_PROB:-0.0}
  MODE="entropy_gradient_masking_${MASKING_PROB}"
  EXTRA_ARGS="--entropy_gradient_masking --masking_prob $MASKING_PROB"
elif [ "$MODE" == "entropy_gradweight_masking" ]; then
  MASKING_PROB=${MASKING_PROB:-0.0}
  MODE="entropy_gradweight_masking_${MASKING_PROB}"
  EXTRA_ARGS="--entropy_gradweight_masking --masking_prob $MASKING_PROB"
elif [ "$MODE" == "gradweight_whole_masking" ]; then
  MASKING_PROB=${MASKING_PROB:-0.0}
  MODE="gradweight_whole_masking_${MASKING_PROB}"
  EXTRA_ARGS="--gradweight_whole_masking --masking_prob $MASKING_PROB"
elif [ "$MODE" == "gradient_entropy_masking" ]; then
  MASKING_PROB=${MASKING_PROB:-0.0}
  MODE="gradient_entropy_masking_${MASKING_PROB}"
  EXTRA_ARGS="--gradient_entropy_masking --masking_prob $MASKING_PROB"
elif [ "$MODE" == "fft" ]; then
  EXTRA_ARGS="--fft"
fi

port=$(shuf -i25000-30000 -n1)
OLD_BS=$BS

TASK_ARGS=""
case $TASK in
CB)
  DEV=50
  BS=4
  GA=$(expr $OLD_BS / $BS)
  echo "Gradient accumulation: $GA"
  TASK_ARGS="--gradient_accumulation_steps $GA"
  ;;
WSC)
  DEV=50
  ;;
MultiRC)
  BS=2
  GA=$(expr $OLD_BS / $BS)
  echo "Gradient accumulation: $GA"
  TASK_ARGS="--gradient_accumulation_steps $GA"
  ;;
ReCoRD)
  BS=1
  GA=$(expr $OLD_BS / $BS)
  echo "Gradient accumulation: $GA"
  TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False --eval_batch_size 1"
  ;;
DROP)
  BS=1
  GA=$(expr $OLD_BS / $BS)
  echo "Gradient accumulation: $GA"
  TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
  ;;
esac

TAG="$MODE-$LR-$SEED"

deepspeed --master_port $port --include localhost:$LOCAL_HOST run2.py --deepspeed "$DS_CONFIG" \
  --overwrite_output_dir \
  --model_name $MODEL \
  --task_name $TASK \
  --output_dir ./saved_models/$TASK-${MODEL_NAME}-$TAG\
  --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
  --learning_rate $LR --num_train_epochs $EPOCH --per_device_train_batch_size $BS \
  --load_best_model_at_end --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1 \
  --train_as_classification \
  $EXTRA_ARGS \
  $TASK_ARGS \
  "$@"

# --bf16 \