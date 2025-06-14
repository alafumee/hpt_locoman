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
