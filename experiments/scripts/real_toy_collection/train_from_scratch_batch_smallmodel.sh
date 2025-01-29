for size in smaller small mid full; do
  bash ./experiments/scripts/real_toy_collection/train_locoman_from_scratch_smallmodel.sh none from_scratch_${size}_smallmodel 1 $size
done
