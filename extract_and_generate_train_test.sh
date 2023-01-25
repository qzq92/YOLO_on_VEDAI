#!/bin/bash
#!/usr/bin/env python
PATH=$PWD:$PATH

# Config to terminate on error
set -e

# Directory paths
CO=./Vehicles/CO
IR=./Vehicles/IR
veh_directory_to_check=./Vehicles
annot_directory_to_check=./Annotations
yolo_dir=./yolov7

# Tar archives parts for image
tar1=Vehicules1024.tar.001
tar2=Vehicules1024.tar.002
tar3=Vehicules1024.tar.003
tar4=Vehicules1024.tar.004
tar5=Vehicules1024.tar.005


# Tar files for annotation
annot_tar=Annotations1024.tar

# Download annotations
#wget https://downloads.greyc.fr/vedai/Annotations1024.tar


echo "Retrieving VEDAI images of 1024x1024 resolution online.."
# Download higher resolution (1024x1024) images

echo "Checking for tar archive files containing to VEDAI dataset and downloading if required"
if [ ! -f $tar1 ]; then
	wget https://downloads.greyc.fr/vedai/Vehicules1024.tar.001
fi
if [ ! -f $tar2 ]; then
	wget https://downloads.greyc.fr/vedai/Vehicules1024.tar.002
fi

if [ ! -f $tar3 ]; then
	wget https://downloads.greyc.fr/vedai/Vehicules1024.tar.003
fi

if [ ! -f $tar4 ]; then
	wget https://downloads.greyc.fr/vedai/Vehicules1024.tar.004
fi

if [ ! -f $tar5 ]; then
	wget https://downloads.greyc.fr/vedai/Vehicules1024.tar.005
fi
echo "Check and download completed"

if [ -d $veh_directory_to_check ] 
then
    echo "Directory $veh_directory_to_check exists. Deleting and recreating new extraction of the Vedai dataset downloaded online."
	rm -rf $veh_directory_to_check
	echo "Completed"
fi



echo "Extracting splitted tar folders and renaming the folder..."
cat  Vehicules1024.tar.[0-9][0-9][0-9] | tar -xvf - 
mv Vehicules1024 $veh_directory_to_check
echo "Complete extraction of tar files"


echo "Checking if yolov7 repo exists"
if [ ! -d $yolo_dir ];then
	echo "Cloning YOLOv7 repo from Github"
	git clone https://github.com/WongKinYiu/yolov7.git
	echo "Clone completed"
else
	echo "YOLOv7 repo exists. Not doing anything..."
fi


#if [ -d $annot_directory_to_check ] 
#then
#    echo "Directory $annot_directory_to_check exists. Deleting and recreating for new extraction of the Vedai dataset annotations downloaded online."
#	rm -rf $annot_directory_to_check
#	echo "Completed"
#fi


#echo "Extracting annotation tar folder and renaming the folder..."
#tar -xvf Annotations1024.tar 
#mv Annotations1024 $annot_directory_to_check
#echo "Completed"
#echo ""


echo "Creating subfolders and moving relevant images accordingly.."
mkdir $CO
mkdir $IR
mv $veh_directory_to_check/*_co.png $CO
mv $veh_directory_to_check/*_ir.png $IR
echo "Completed"

echo "Creating training and testing datasets with YOLO standard annotations"
python process_annotation_to_yolo.py
echo "Completed"