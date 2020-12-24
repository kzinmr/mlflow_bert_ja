export DATA_DIR=${WORK_DIR}/data/
export OUTPUT_DIR=${WORK_DIR}/outputs/
export SEED=42
mkdir -p $OUTPUT_DIR

export NUM_EPOCHS=10
export NUM_SAMPLES=20000

python3 ner_crf.py \
--output_dir=$OUTPUT_DIR \
--data_dir=$DATA_DIR \
--seed=$SEED \
--num_samples=$NUM_SAMPLES \
--max_iterations=$NUM_EPOCHS \
--c1=1.0 \
--c2=1.0 \
--all_possible_transitions
