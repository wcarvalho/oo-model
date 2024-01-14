CUDA_VISIBLE_DEVICES=0 \
python -m ipdb -c continue experiments/minihack_online_trainer.py \
  --train_single=True \
  --folder="/home/jhsansom/results/" \
  --use_wandb=True \
  --wandb_entity="jhsansom" \
  --wandb_project="Factored MuZero" \
  --search="factored1" \
  --num_gpus=1