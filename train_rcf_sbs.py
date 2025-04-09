import os
import argparse

os.system("python3 -m src.train_RCF\
        --batch_size 1 \
        --learning_rate 1e-6 \
        --momentum 0.9 \
        --weight_decay 2e-4 \
        --stepsize 3 \
        --gamma 0.1 \
        --maxepoch 30 \
        --itersize 10 \
        --start_epoch 0 \
        --print_freq 1000 \
        --gpu 0 \
        --dataset_name HED-BSDS\
 ")
