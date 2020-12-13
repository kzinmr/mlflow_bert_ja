export DATA_DIR=${PWD}/ner_data/
export OUTPUT_DIR=${PWD}/outputs/
export CACHE=${PWD}/cache/
export SEED=42
export BERT_MODEL=cl-tohoku/bert-base-japanese
export MAX_LENGTH=128
export LEARNING_RATE=5e-5

export BATCH_SIZE=32

export LABEL_PATH=$DATA_DIR/label_types.txt
export NUM_EPOCHS=1
export NUM_SAMPLES=100
mkdir -p $OUTPUT_DIR

python3 ner_refactor.py \
--data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR \
--seed $SEED \
--num_samples=$NUM_SAMPLES \
--gradient_accumulation_steps=1 \
--do_train \
--do_predict \
--model_name_or_path=$BERT_MODEL \
--cache_dir=$CACHE \
--labels $LABEL_PATH \
--max_seq_length  $MAX_LENGTH \
--train_batch_size $BATCH_SIZE \
--eval_batch_size $BATCH_SIZE \
--num_workers=4 \
--learning_rate=$LEARNING_RATE \
--adam_epsilon=1e-8 \
--weight_decay=0.0 \
--warmup_steps=0 \
--num_train_epochs $NUM_EPOCHS \
--gpus=0

# --adafactor \ else AdamW
# --tokenizer_name=""
# --config_name=""
# --encoder_layerdrop
# --decoder_layerdrop
# --dropout
# --attention_dropout