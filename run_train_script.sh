# from scratch
# provide a placeholder e.g. 'none' to train from scratch
bash ./train_example.sh none desired_savedir_name dataset_name 1

# finetune
bash ./train_example.sh pretrained_dir_name desired_savedir_name dataset_name 1

# Note: The script will look for a dataset directory at ~/Human2LocoMan/demonstrations/${dataset_name}.
# You can change this by modifying dataset_generator_func.dataset_dir in locoman.yaml
# You may change other parameters in train_example.sh as needed.