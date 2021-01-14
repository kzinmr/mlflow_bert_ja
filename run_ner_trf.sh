export BERT_MODEL=cl-tohoku/bert-base-japanese
export DATA_DIR=${WORK_DIR}/data/
export OUTPUT_DIR=${WORK_DIR}/outputs/
export SEED=42
mkdir -p $OUTPUT_DIR
export NUM_WORKERS=8

# https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py
# export GPUS=1
# export MAX_LENGTH=128
export BATCH_SIZE=16
export LEARNING_RATE=1e-3
export WEIGHT_DECAY=0.01
# export PATIENCE=3
# export ANNEAL_FACTOR=0.5

export NUM_EPOCHS=3
# export NUM_SAMPLES=20000

python ner_trf.py \
  --model_name_or_path ${BERT_MODEL} \
  --train_file ${DATA_DIR}/train.csv \
  --validation_file ${DATA_DIR}/dev.csv \
  --test_file ${DATA_DIR}/test.csv \
  --output_dir ${OUTPUT_DIR} \
  --preprocessing_num_workers ${NUM_WORKERS} \
#   --pad_to_max_length \
#   --label_all_tokens \
  --return_entity_level_metrics \
  --per_device_train_batch_size=${BATCH_SIZE} \
  --per_device_eval_batch_size=${BATCH_SIZE} \
  --gradient_accumulation_steps=1 \
  --learning_rate=${LEARNING_RATE} \
  --weight_decay=${WEIGHT_DECAY} \
  --adam_epsilon=1e-8 \
  --num_train_epochs=${NUM_EPOCHS} \
  --lr_scheduler_type="linear" \
  --seed=${SEED} \
  --load_best_model_at_end \
  --do_train \
  --do_eval