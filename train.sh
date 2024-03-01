#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python -u train.py --batch_size 64 --learning_rate 0.0003 --dataset_path /storage/public_datasets/imagenet/ILSVRC/Data/CLS-LOC/train --max_epoch 1000  
