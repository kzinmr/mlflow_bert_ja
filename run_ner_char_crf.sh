export DATA_DIR=${WORK_DIR}/data/
export OUTPUT_DIR=${WORK_DIR}/outputs/
export SEED=42
mkdir -p $OUTPUT_DIR

export NUM_EPOCHS=1
export NUM_SAMPLES=100

python3 ner_char_crf.py \
--output_dir=$OUTPUT_DIR \
--data_dir=$DATA_DIR \
--seed=$SEED \
--num_samples=$NUM_SAMPLES \
--max_iterations=$NUM_EPOCHS \
--c1=0.1 \
--c2=0.1 \
--window_size=20 \
--all_possible_transitions
