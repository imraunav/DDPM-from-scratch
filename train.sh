#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python -u train.py \
--batch_size 32 \
--learning_rate 0.0003 \
--dataset_path /storage/raunav/imagenet/ILSVRC/Data/CLS-LOC/train \
--max_updates 1000000  \
