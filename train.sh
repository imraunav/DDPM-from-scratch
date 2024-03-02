#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python -u train.py --batch_size 64 --learning_rate 0.0003 --dataset_path /storage/public_datasets/DIV2K --max_epoch 1000  
