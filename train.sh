#!/bin/bash

CUDA_VISIBLE_DEVICE=6 python train.py --batch_size 64 --learning_rate 0.0003 --dataset_path /storage/public_datasets/imagenet/ILSVRC/Data/CLS-LOC/train --max_epoch 1000  
