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

# train
ADD_ARGUMENT=${5-""}

# Loop through the arguments starting from the 6th
for arg in "${@:6}"; do
  ADD_ARGUMENT+=" $arg"  # Concatenate each argument
done


CMD="CUDA_VISIBLE_DEVICES=7 HYDRA_FULL_ERROR=1 time python -m hpt.run  \
		script_name=$STAT_DIR \
		env=real_toy_collection  \
		train.pretrained_dir=output/$PRETRAINED  \
		dataset.episode_cnt=100 \
		train.freeze_trunk=False \
		domains=train_toy_collect_locoman_${size} \
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
