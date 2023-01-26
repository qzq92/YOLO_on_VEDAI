#!/bin/bash
#!/usr/bin/env python
PATH=$PWD:$PATH

# Config to terminate on error
set -e

# Ratio of original data for validation purpose
validation_ratio=0.1

echo "Removing existing train folder in yolov7 if any"
if [ -d $PWD/train ];then
	rm -rf $PWD/train
	echo "Done."
fi

echo "Removing existing val folder in yolov7 if any"
if [ -d $PWD/val ];then
	rm -rf $PWD/val
	echo "Done."
fi

echo "Creating training and testing datasets with YOLO standard annotations. 'Train' and 'test' folders would be generated to store them..."
python process_annotation_to_yolo.py --validation_ratio $validation_ratio 
echo "Completed"