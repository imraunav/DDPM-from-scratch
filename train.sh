#!/bin/bash

accelerate launch train.py \
--batch_size 16 \
--learning_rate 0.0003 \
--image_size 256 \
--dataset_path /storage/public_datasets/celeba_hq/CelebAMask-HQ/CelebA-HQ-img/ \
--max_epoch 1000  \

