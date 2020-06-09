#!/usr/bin/env bash

mkdir embeddings
cd embeddings
# CUB 2010-2011 train embeddings
wget https://www.dropbox.com/s/qgacaoo7urh35j9/BigBiGAN_CUB_train_z.npy
# DUTS
wget https://www.dropbox.com/s/b77in0vuc8jy1yf/BigBiGAN_DUTS-TR_z.npy
# Flowers
wget https://www.dropbox.com/s/fmw2g54xqli59ck/BigBiGAN_Flowers_train_z.npy
# ImageNet
wget https://www.dropbox.com/s/y5qr9wwtop7ot6b/BigBiGAN_ImageNet_z.npy

cd ../BigGAN
mkdir weights
cd weights
# BigBiGAN-pytorch weights
wget https://www.dropbox.com/s/9w2i45h455k3b4p/BigBiGAN_x1.pth
# background darkening and foreground lightening direction
wget https://www.dropbox.com/s/np74kmkohkbx76t/bg_direction.pth
