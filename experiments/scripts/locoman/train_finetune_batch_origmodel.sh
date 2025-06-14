for size in smaller small mid full; do
  bash ./experiments/scripts/real_toy_collection/train_locoman_finetune.sh 10_02_2025_09_35_11_250032_humanpretrain_single_fixedimage finetune_fixedimage_origmodel_${size} 1 $size
done