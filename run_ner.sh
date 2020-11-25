export DATA_DIR=./ner_data/
export MAX_LENGTH=128
export LEARNING_RATE=2e-5
export BERT_MODEL=bert-base-multilingual-cased
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SEED=1
export OUTPUT_DIR_NAME=germeval-model
export OUTPUT_DIR=${PWD}/${OUTPUT_DIR_NAME}
mkdir -p $OUTPUT_DIR

python3 download_ner_data.py $DATA_DIR $BERT_MODEL $MAX_LENGTH

python3 ner.py --gpus 1 --data_dir $DATA_DIR \
--labels ./labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--learning_rate $LEARNING_RATE \
--num_train_epochs $NUM_EPOCHS \
--train_batch_size $BATCH_SIZE \
--seed $SEED \
--do_predict