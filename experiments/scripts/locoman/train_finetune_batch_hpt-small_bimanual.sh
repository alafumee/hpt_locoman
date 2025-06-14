for size in small mid full; do
  bash ./experiments/scripts/real_toy_collection/train_locoman_from_scratch_bimanual.sh hf://liruiw/hpt-small from_hpt-small_${size}_bimanual 1 $size
done