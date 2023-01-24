#!/bin/bash

# Config to terminate on error
set -e

# Directory paths
CO=./Vehicles/CO
IR=./Vehicles/IR
veh_directory_to_check=./Vehicles
annot_directory_to_check=./Annotations

# Tar files
annot_tar=Annotations1024.tar

# Download annotations
#wget https://downloads.greyc.fr/vedai/Annotations1024.tar


echo "Retrieving VEDAI images of 1024x1024 resolution online.."
# Download higher resolution (1024x1024) images
wget https://downloads.greyc.fr/vedai/Vehicules1024.tar.001
wget https://downloads.greyc.fr/vedai/Vehicules1024.tar.002
wget https://downloads.greyc.fr/vedai/Vehicules1024.tar.003
wget https://downloads.greyc.fr/vedai/Vehicules1024.tar.004
wget https://downloads.greyc.fr/vedai/Vehicules1024.tar.005
echo "Download completed"


if [ -d $veh_directory_to_check ] 
then
    echo "Directory $veh_directory_to_check exists. Deleting and recreating new extraction of the Vedai dataset downloaded online."
	rm -rf $veh_directory_to_check
	echo "Completed"
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

echo "Extracting splitted tar folders and renaming the folder..."
cat  Vehicules1024.tar.[0-9][0-9][0-9] | tar -xvf - 
mv Vehicules1024 $veh_directory_to_check
echo "Complete extraction of tar files"

echo "Creating subfolders and moving relevant images accordingly.."
mkdir $CO
mkdir $IR
mv $veh_directory_to_check/*_co.png $CO
mv $veh_directory_to_check/*_ir.png $IR
