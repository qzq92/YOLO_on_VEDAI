#!/bin/bash
#!/usr/bin/env python
PATH=$PWD:$PATH

# Config to terminate on error
set -e

# Ratio of original data for validation purpose
validation_ratio=0.1
testing_ratio=0.2

echo "Creating training and testing datasets with YOLO standard annotations. 'Train' and 'test' folders would be generated to store them..."
python process_annotation_to_yolo.py --validation_ratio $validation_ratio --testing_ratio $testing_ratio
echo "Completed"