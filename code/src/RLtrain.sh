#!/usr/bin/env bash
python3 RLtrain.py \
 --cv_dir 'RLsave/reg_1' \
 --model_type 'resnet200mf' \
 --batch_size 16
