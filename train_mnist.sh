#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python -u train_mnist.py \
--batch_size 64 \
--learning_rate 0.0003 \
--max_epoch 1000  \
