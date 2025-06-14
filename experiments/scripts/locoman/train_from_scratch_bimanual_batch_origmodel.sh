for size in small mid full; do
  bash ./experiments/scripts/real_toy_collection/train_locoman_from_scratch_bimanual.sh none from_scratch_bimanual_${size} 1 $size
done
