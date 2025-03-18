for size in small mid full; do
  bash ./experiments/scripts/real_toy_collection/train_locoman_finetune_bimanual.sh 10_02_2025_09_33_50_248490_humanpretrain_bimanual finetune_bimanual_origmodel_${size} 1 $size
done