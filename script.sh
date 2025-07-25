#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python3 experiments/cshrec11_encode/train.py --in "data/SHREC_11/processed/" --out "./cshrec11_encode/weights/" && python3 experiments/cshrec11_pred/train.py --in "data/SHREC_11/processed/" --weights "./cshrec11_encode/weights/0/checkpoints-0/" --out "./cshrec11_pred/weights/"