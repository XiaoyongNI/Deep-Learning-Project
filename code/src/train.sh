#!/usr/bin/env bash
python train.py \
  --saved_fn 'test' \
  --arch 'darknet' \
  --batch_size 16 \
  --num_workers 4 \
  --gpu_idx 0 \
  --detector_type 'fine' \
