for size in smaller small mid full; do
  bash ./experiments/scripts/real_toy_collection/train_locoman_from_scratch_smallmodel.sh hf://liruiw/hpt-small from_hpt-small_${size} 1 $size
done