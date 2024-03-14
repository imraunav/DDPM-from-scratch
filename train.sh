#!/bin/bash

accelerate launch train.py \
--batch_size 2 \
--learning_rate 0.00002 \
--image_size 256 \
--dataset_path /storage/public_datasets/celeba_hq/CelebAMask-HQ/CelebA-HQ-img/ \
--max_epoch 1000  \

