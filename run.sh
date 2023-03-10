#! /bin/bash

# source ~/hailing.env

# Configure input data 
filePrefix="/lustre/collider/mocen/project/darkshine/track/data/"
fileList=( "${filePrefix}1/" "${filePrefix}2/" "${filePrefix}3/")
# fileList=( "${filePrefix}signal_16Jan2023/xyz_domGT20/" )

nChannels=3
nClasses=3

data_size=8
num_epochs=20
num_slices_train=60
num_slices_test=20
num_slices_apply=20

num_slices_train=1
num_slices_test=1
num_slices_apply=1

lr=0.01
batch_size=8
# Apply only
apply_only=0

# Pre-train
pre_train=0
pre_net="./net.pt"
pre_log="./train-result.json"


python gnn/main.py --fileList "${fileList[@]}" \
                --data_size $data_size  --num_slices_train $num_slices_train --num_slices_test $num_slices_test \
                --num_slices_apply $num_slices_apply --num_classes $nClasses --num_channels $nChannels \
                --apply_only $apply_only  --pre_train $pre_train  --pre_net $pre_net  --pre_log $pre_log --num_epochs=$num_epochs \
                --lr $lr --batch_size $batch_size
