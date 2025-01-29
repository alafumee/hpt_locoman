for size in smaller small mid full; do
  bash ./experiments/scripts/real_toy_collection/train_locoman_finetune.sh 25_01_2025_06_02_03_3263847_pretrain_gaussiannorm realfinetune_origmodel_${size} 1 $size
done