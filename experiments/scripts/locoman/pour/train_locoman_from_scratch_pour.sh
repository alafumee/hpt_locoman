set -x
set -e
DATE="`date +'%d_%m_%Y_%H_%M_%S'`_$$" 
STAT_DIR=${0}
STAT_DIR="${STAT_DIR##*/}"
STAT_DIR="${STAT_DIR%.sh}"

echo "RUNNING $STAT_DIR!"
PRETRAINED=${1}
PRETRAINEDCMD=${2}
NUM_RUN=${3-"1"}
size=${4-"small"}
VAL_RATIO=${5-"0.05"}


# train
ADD_ARGUMENT=${6-""}

# Loop through the arguments starting from the 6th
for arg in "${@:7}"; do
  ADD_ARGUMENT+=" $arg"  # Concatenate each argument
done


CMD="CUDA_VISIBLE_DEVICES=5 HYDRA_FULL_ERROR=1 time python -m hpt.run  \
		script_name=$STAT_DIR \
		env=real_toy_collection  \
		train.pretrained_dir=output/$PRETRAINED  \
		dataset.val_ratio=$VAL_RATIO \
		dataset.episode_cnt=1000 \
		dataset.action_horizon=180 \
		train.total_iters=100000 \
		dataloader.batch_size=24 \
		val_dataloader.batch_size=24 \
		optimizer.lr=5e-5 \
		train.freeze_trunk=False \
		domains=train_pour_locoman_bimanual_${size} \
		output_dir=output/${DATE}_${PRETRAINEDCMD} \
		$ADD_ARGUMENT"

eval $CMD
# eval
# HYDRA_FULL_ERROR=1  time  python   -m hpt.run_eval --multirun \
# 		--config-name=config \
# 		--config-path=../output/${DATE}_${PRETRAINEDCMD}   \
# 		train.pretrained_dir="'output/${DATE}_${PRETRAINEDCMD}'" \
# 	  seed="range(3)" \
# 		hydra.sweep.dir=output/${DATE}_${PRETRAINEDCMD}  \
#  		hydra/launcher=joblib \
# 		hydra.launcher.n_jobs=3
