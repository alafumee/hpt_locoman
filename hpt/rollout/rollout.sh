set -x
set -e
# DATE="`date +'%d_%m_%Y_%H_%M_%S'`_$$" 
STAT_DIR=${0}
STAT_DIR="${STAT_DIR##*/}"
STAT_DIR="${STAT_DIR%.sh}"

echo "RUNNING $STAT_DIR!"
DIR_STR=${1}
# PRETRAINEDCMD=${2}
# NUM_RUN=${3-"1"}


# train
ADD_ARGUMENT=${4-""}

# Loop through the arguments starting from the 5th
for arg in "${@:5}"; do
  ADD_ARGUMENT+=" $arg"  # Concatenate each argument
done


# CMD="CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 time python -m hpt.run  \
# 		script_name=$STAT_DIR \
# 		env=real_toy_collection  \
# 		train.pretrained_dir=output/$PRETRAINED  \
# 		dataset.episode_cnt=100 \
# 		domains=toy_collect_human \
# 		output_dir=output/${DATE}_${PRETRAINEDCMD} \
# 		$ADD_ARGUMENT"

# eval $CMD
# eval
# multirun
# HYDRA_FULL_ERROR=1  time  python   -m hpt.rollout.hpt_rollout --multirun \
# 		--config-name=config \
# 		--config-path=../output/${DIR_STR}   \
# 		train.pretrained_dir="'output/${DIR_STR}'" \
# 	  seed="range(3)" \
# 		hydra.sweep.dir=output/${DIR_STR}  \
#  		hydra/launcher=joblib \
# 		hydra.launcher.n_jobs=3

HYDRA_FULL_ERROR=1  time  python -m hpt.rollout.hpt_rollout \
		--config-name=config \
		--config-path=../output/${DIR_STR}   \
		train.pretrained_dir="'output/${DIR_STR}'" \
		+use_real_robot=True \
        +head_camera_type=1 \
        +use_wrist_camera=True \
        +desired_stream_fps=60 \
        +control_freq=60 \
        +inference_interval=10 \
        +action_chunk_size=60 \
        +device="'cuda:0'" \
        +exp_name="'test'" \
        +delta_action=False \
