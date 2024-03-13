#!/bin/bash

accelerate launch train.py \
--batch_size 8 \
--learning_rate 0.0003 \
--image_size 256 \
--load_checkpoint /storage/raunav/DDPM-from-scratch/checkpoint_celeb.pt
--dataset_path /storage/public_datasets/celeba_hq/CelebAMask-HQ/CelebA-HQ-img/ \
--max_epoch 1000  \

