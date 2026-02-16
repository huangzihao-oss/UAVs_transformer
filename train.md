python3 train.py \
  --M 2 --N 1 --K 20 --T 1800 --map_size 1000 \
  --speed_levels 6-20 --history_horizon 20 --init_uav_energy 300000 \
  --transformer_dim 128 --transformer_heads 4 --transformer_layers 2 --transformer_dropout 0.1 \
  --max_ep_len 1000 --max_training_timesteps 3000000
