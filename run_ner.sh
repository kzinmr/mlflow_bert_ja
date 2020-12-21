export BERT_MODEL=cl-tohoku/bert-base-japanese
# export WORK_DIR=${PWD}
export DATA_DIR=${WORK_DIR}/data/
export OUTPUT_DIR=${WORK_DIR}/outputs/
# export CACHE=${WORK_DIR}/cache/
export LABEL_PATH=$DATA_DIR/label_types.txt
export SEED=42
mkdir -p $OUTPUT_DIR
# In Docker, the following error occurs due to not big enough memory:
# `RuntimeError: DataLoader worker is killed by signal: Killed.`
# Try to reduce NUM_WORKERS or MAX_LENGTH or BATCH_SIZE or increase docker memory
export NUM_WORKERS=8
export GPUS=1

export MAX_LENGTH=128
export BATCH_SIZE=16
export LEARNING_RATE=1e-3
export WEIGHT_DECAY=0.01
export PATIENCE=3
export ANNEAL_FACTOR=0.5

export NUM_EPOCHS=10
export NUM_SAMPLES=15000

python3 ner.py \
--model_name_or_path=$BERT_MODEL \
--output_dir=$OUTPUT_DIR \
--accumulate_grad_batches=1 \
--max_epochs=$NUM_EPOCHS \
--seed=$SEED \
--do_train \
--do_predict \
--gpus=$GPUS \
--data_dir=$DATA_DIR \
--labels=$LABEL_PATH \
--num_workers=$NUM_WORKERS \
--max_seq_length=$MAX_LENGTH \
--train_batch_size=$BATCH_SIZE \
--eval_batch_size=$BATCH_SIZE \
--learning_rate=$LEARNING_RATE \
--patience=$PATIENCE \
--anneal_factor=$ANNEAL_FACTOR \
--adam_epsilon=1e-8 \
--weight_decay==$WEIGHT_DECAY \
--num_samples=$NUM_SAMPLES
