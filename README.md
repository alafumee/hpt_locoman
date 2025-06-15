# HPT Usage for LocoMan:
1. Install HPT dependencies following [the instructions](#️-setup)

2. Prepare your datasets in the same format in our main repo and update the dataset name and save directory in `run_train_script.sh`. The script will look for a dataset directory at `~/Human2LocoMan/demonstrations/${dataset_name}`， while the results will be saved at `output/${DATE}_${save_dir}`. If you wish, you can change these by modifying `dataset_generator_func.dataset_dir` in `locoman.yaml`, and `output_dir` in `experiments/scripts/locoman/train_example.sh`. Then you can execute the script to train the model. 

```bash
# from scratch
# provide a placeholder e.g. 'none' to train from scratch
bash ./train_example.sh none desired_savedir_name dataset_name 1

# finetune
bash ./train_example.sh pretrained_dir_name desired_savedir_name dataset_name 1
```

You can adjust the parameters as needed in `experiments/scripts/locoman/train_example.sh`.

```bash
set -x
set -e
DATE="`date +'%d_%m_%Y_%H_%M_%S'`_$$" 
STAT_DIR=${0}
STAT_DIR="${STAT_DIR##*/}"
STAT_DIR="${STAT_DIR%.sh}"

echo "RUNNING $STAT_DIR!"
PRETRAINED=${1}
PRETRAINEDCMD=${2}
NUM_RUN=${4-"1"}
DATASET_NAME=${3-"locoman"}


# train
ADD_ARGUMENT=${5-""}

# Loop through the arguments starting from the 5th
for arg in "${@:6}"; do
  ADD_ARGUMENT+=" $arg"  # Concatenate each argument
done


CMD="CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 time python -m hpt.run  \
		script_name=$STAT_DIR \
		env=locoman  \
		train.pretrained_dir=output/$PRETRAINED  \
		dataset.episode_cnt=100 \
		train.total_iters=160000 \
		dataloader.batch_size=32 \
		val_dataloader.batch_size=32 \
		optimizer.lr=1e-4 \
		train.freeze_trunk=False \
		domains=${DATASET_NAME} \
		output_dir=output/${DATE}_${PRETRAINEDCMD} \
		$ADD_ARGUMENT"

eval $CMD
```

For more information, you may refer to the original instructions [here](https://github.com/liruiw/HPT) or below. You should be able to train on LocoMan datasets with the above instructions and not worry about the original usage instructions.
<hr style="border: 2px solid gray;"></hr>


# 🦾 Heterogenous Pre-trained Transformers
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=flat-square)](https://huggingface.co/liruiw/hpt-base)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Paper](https://badgen.net/badge/icon/arXiv?icon=awesome&label&color=red&style=flat-square)](http://arxiv.org/abs/2409.20537)
[![Website](https://img.shields.io/badge/Website-hpt-blue?style=flat-square)](https://liruiw.github.io/hpt)
[![Python](https://img.shields.io/badge/Python-%3E=3.8-blue?style=flat-square)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E=2.0-orange?style=flat-square)]()

[Lirui Wang](https://liruiw.github.io/), [Xinlei Chen](https://xinleic.xyz/), [Jialiang Zhao](https://alanz.info/), [Kaiming He](https://people.csail.mit.edu/kaiming/)

Neural Information Processing Systems (Spotlight), 2024



<hr style="border: 2px solid gray;"></hr>


This is a pytorch implementation for pre-training Heterogenous Pre-trained Transformers (HPTs). The pre-training procedure train on mixture of embodiment datasets with a supervised learning objective. The pre-training process can take some time, so we also provide pre-trained checkpoints below. You can find more details on our [project page](https://liruiw.github.io/hpt). An alternative clean implementation of HPT in Hugging Face can also be found [here](https://github.com/liruiw/lerobot/).



## ⚙️ Setup
1. ```pip install -e .```


<details>
<summary><span style="font-weight: bold;">Install (old-version) Mujoco</span></summary>

```
mkdir ~/.mujoco
cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz  -O mujoco210.tar.gz --no-check-certificate
tar -xvzf mujoco210.tar.gz

# add the following line to ~/.bashrc if needed
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export MUJOCO_GL=egl
```

</details>

## 🚶 Usage
0. Check out ``quickstart.ipynb`` for how to use the pretrained HPTs.
1. ```python -m hpt.run``` train policies on each environment. Add `+mode=debug`  for debugging.
2. ```bash experiments/scripts/metaworld/train_test_metaworld_1task.sh test test 1 +mode=debug``` for example script.
3. Change ``train.pretrained_dir`` for loading pre-trained trunk transformer. The model can be loaded either from local checkpoint folder or huggingface [repository](https://huggingface.co/liruiw/hpt-xlarge).


## 🤖 Try this On Your Own Dataset
0. For training, it requires a dataset conversion  `convert_dataset` function for packing your own datasets. Check [this](env/realworld) for example.
1. For evaluation, it requires a `rollout_runner.py` file for each benchmark and  a ``learner_trajectory_generator`` evaluation function that provides rollouts.
2. If needed, modify the [config](experiments/configs/config.yaml) for changing the perception stem networks and action head networks in the models. Take a look at [`realrobot_image.yaml`](experiments/configs/env/realrobot_image.yaml) for example script in the real world.
3. Add `dataset.use_disk=True` for saving and loading the dataset in disk.



---
## 💾 File Structure
```angular2html
├── ...
├── HPT
|   ├── data            # cached datasets
|   ├── output          # trained models and figures
|   ├── env             # environment wrappers
|   ├── hpt             # model training and dataset source code
|   |   ├── models      # network models
|   |   ├── datasets    # dataset related
|   |   ├── run         # transfer learning main loop
|   |   ├── run_eval    # evaluation main loop
|   |   └── ...
|   ├── experiments     # training configs
|   |   ├── configs     # modular configs
└── ...
```

### 🕹️ Citation
If you find HPT useful in your research, please consider citing:
```
@inproceedings{wang2024hpt,
author    = {Lirui Wang, Xinlei Chen, Jialiang Zhao, Kaiming He},
title     = {Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers},
booktitle = {Neurips},
year      = {2024}
}
```

