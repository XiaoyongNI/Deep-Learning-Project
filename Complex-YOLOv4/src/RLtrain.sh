#!/usr/bin/env bash
python3 RLtrain.py \
  --saved_fn 'complex_yolov4' \
  --arch 'darknet' \
  --batch_size 4 \
  --num_workers 4 \
  --no-val \
  --gpu_idx 0
