#!/bin/bash
#!/usr/bin/env python
PATH=$PWD:$PATH

# Config to terminate on error
set -e

# Directory paths
yolo_dir=./yolov7
data_path=./data/vedai.yaml
hyperparams_path=./data/hyp.scratch.custom.yaml
cfg_path=./cfg/training/yolov7-vedai-cfg.yaml
output_name=VEDAI
default_yolo_weight=./yolov7/yolov7.pt

# Configs for YOLOv7
workers=1
device=0 #Do not change if you only have 1 gpu, 0-based indexing
batch_size=8
epochs=100
img_res=1024
output_name=VEDAI
weights=$yolo_dir/yolov7.pt


echo "Checking if yolov7 repo exists..."
if [ ! -d $yolo_dir ];then
	echo "Cloning YOLOv7 repo from Github"
	git clone https://github.com/WongKinYiu/yolov7.git
	echo "Clone completed"
else
	echo "YOLOv7 repo exists. Not doing anything..."
fi

echo "Checking for weights file..."
if [ ! -f $default_yolo_weight ];then
	echo "Downloading trained weights, $weights online"
	cd $yolo_dir
	wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
	cd ..
	echo "Weights downloaded completed"
else
	echo "yolov7.pt already downloaded.Nothing to do here."
fi

if [ ! -d $yolo_dir/data/train ];then
	echo "Moving train folder into yolov7's data subdirectory"
	mv train $yolo_dir/data
else
	echo "Train folder already exists in yolov7"
fi


if [ ! -d $yolo_dir/data/val ];then
	echo "Moving test folder into yolov7's data subdirectory"
	mv val $yolo_dir/data
else
	echo "Test folder already exists in yolov7"
fi

echo "Copying vedai.yaml over to $yolo_dir/data"
cp vedai.yaml $yolo_dir/data

echo "Moving over config file to yolov7's cfg's training subdirectory"
cp yolov7-vedai-cfg.yaml $yolo_dir/cfg/training

echo "Training yolo model"
python $yolo_dir/train.py --workers $workers --device $device --batch-size $batch_size --epochs $epochs --img $img_res $img_res --data $data_path --hyp $hyperparams_path --cfg $cfg_path --name $output_name --weights $weights
