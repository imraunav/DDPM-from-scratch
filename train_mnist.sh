#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python -u train_mnist.py \
--batch_size 32 \
--learning_rate 0.0003 \
--n_resblock 8 \
--model_channel 16 \
--groups 16 \
--dropout 0.5 \
--n_head 8 \
--max_epoch 100  \
