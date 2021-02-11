python3 train_ssd300.py \
	--exp_name  sample \
	--reload_data_path NUMPY_DATA_PATH \
	--data_def_dir PATH_TO_CSV \
	--model_type tooth-id \
	--output_path sample_output \
	--epochs 1 \
	--steps_per_epoch 1 \
	--batch_size 1 \
	--data_aug_params_json ../input_files/data_aug_params.json &> log.txt && \
python3 eval_ssd300.py \
	--train_output_path sample_output \
	--output_path sample_output &> score_log.txt
