export BERT_MODEL=cl-tohoku/bert-base-japanese
export DATA_DIR=${PWD}/ner_data/
export OUTPUT_DIR=${PWD}/outputs/
export CACHE=${PWD}/cache/
export SEED=42
export LABEL_PATH=$DATA_DIR/label_types.txt
export NUM_WORKERS=4
export GPUS=0

export MAX_LENGTH=128
export LEARNING_RATE=5e-5
export BATCH_SIZE=32
export NUM_EPOCHS=1
export NUM_SAMPLES=100
mkdir -p $OUTPUT_DIR

python3 ner_ja.py \
--model_name_or_path=$BERT_MODEL \
--output_dir=$OUTPUT_DIR \
--accumulate_grad_batches=1 \
--max_epochs=$NUM_EPOCHS \
--seed=$SEED \
--do_train \
--do_predict \
--cache_dir=$CACHE \
--gpus=$GPUS \
--data_dir=$DATA_DIR \
--labels=$LABEL_PATH \
--num_workers=$NUM_WORKERS \
--max_seq_length=$MAX_LENGTH \
--train_batch_size=$BATCH_SIZE \
--eval_batch_size=$BATCH_SIZE \
--learning_rate=$LEARNING_RATE \
--adam_epsilon=1e-8 \
--weight_decay=0.0 \
--num_samples=$NUM_SAMPLES

# --adafactor \ else AdamW
# --tokenizer_name=""
# --config_name=""
# --encoder_layerdrop
# --decoder_layerdrop
# --dropout
# --attention_dropout

# usage: ner_refactor.py [-h] 
# --output_dir OUTPUT_DIR 
# --model_name_or_path MODEL_NAME_OR_PATH
# [--logger [LOGGER]]
# [--checkpoint_callback [CHECKPOINT_CALLBACK]]
# [--default_root_dir DEFAULT_ROOT_DIR]
# [--gradient_clip_val GRADIENT_CLIP_VAL]
# [--process_position PROCESS_POSITION]
# [--num_nodes NUM_NODES] [--num_processes NUM_PROCESSES]
# [--gpus GPUS] [--auto_select_gpus [AUTO_SELECT_GPUS]]
# [--tpu_cores TPU_CORES]
# [--log_gpu_memory LOG_GPU_MEMORY]
# [--progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE]
# [--overfit_batches OVERFIT_BATCHES]
# [--track_grad_norm TRACK_GRAD_NORM]
# [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH]
# [--fast_dev_run [FAST_DEV_RUN]]
# [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES]
# [--max_epochs MAX_EPOCHS] [--min_epochs MIN_EPOCHS]
# [--max_steps MAX_STEPS] [--min_steps MIN_STEPS]
# [--limit_train_batches LIMIT_TRAIN_BATCHES]
# [--limit_val_batches LIMIT_VAL_BATCHES]
# [--limit_test_batches LIMIT_TEST_BATCHES]
# [--val_check_interval VAL_CHECK_INTERVAL]
# [--flush_logs_every_n_steps FLUSH_LOGS_EVERY_N_STEPS]
# [--log_every_n_steps LOG_EVERY_N_STEPS]
# [--accelerator ACCELERATOR]
# [--sync_batchnorm [SYNC_BATCHNORM]]
# [--precision PRECISION]
# [--weights_summary WEIGHTS_SUMMARY]
# [--weights_save_path WEIGHTS_SAVE_PATH]
# [--num_sanity_val_steps NUM_SANITY_VAL_STEPS]
# [--truncated_bptt_steps TRUNCATED_BPTT_STEPS]
# [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
# [--profiler [PROFILER]] [--benchmark [BENCHMARK]]
# [--deterministic [DETERMINISTIC]]
# [--reload_dataloaders_every_epoch [RELOAD_DATALOADERS_EVERY_EPOCH]]
# [--auto_lr_find [AUTO_LR_FIND]]
# [--replace_sampler_ddp [REPLACE_SAMPLER_DDP]]
# [--terminate_on_nan [TERMINATE_ON_NAN]]
# [--auto_scale_batch_size [AUTO_SCALE_BATCH_SIZE]]
# [--prepare_data_per_node [PREPARE_DATA_PER_NODE]]
# [--plugins PLUGINS] [--amp_backend AMP_BACKEND]
# [--amp_level AMP_LEVEL]
# [--distributed_backend DISTRIBUTED_BACKEND]
# [--automatic_optimization [AUTOMATIC_OPTIMIZATION]]
# [--move_metrics_to_cpu [MOVE_METRICS_TO_CPU]]
# [--enable_pl_optimizer [ENABLE_PL_OPTIMIZER]]
# [--seed SEED]
# [--do_train] [--do_predict]
# [--gradient_accumulation_steps ACCUMULATE_GRAD_BATCHES]
# [--cache_dir CACHE_DIR]