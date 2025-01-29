for size in smaller small mid full; do
  bash ./experiments/scripts/real_toy_collection/train_locoman_finetune_smallmodel.sh 26_01_2025_17_14_04_4006662_pretrain_gaussiannorm_smallmodel finetune_smallmodel_${size} 1 $size
done