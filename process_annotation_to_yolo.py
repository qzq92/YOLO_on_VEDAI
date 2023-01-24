from PIL import Image

import argparse
import os
import sys
import logging

import numpy as np
import pandas as pd

def normalise_bounding_box_val(df):
    """Function that generates the normalised bounding boxes' centre coordinates as well as its normalised width and height sizes with respect to image size.

    Args:
      df: Dataframe considered.
    Raises:
      KeyError: When invalid column is referenced..
    Returns:
      Dataframe with additional columns containing normalised x,y centre coordinates and bounding boxes' width and height.
    """
    
    x_coords = ['corner1_x',
                'corner2_x',
                'corner3_x',
                'corner4_x'
    ]

    y_coords = ['corner1_y',
                'corner2_y',
                'corner3_y',
                'corner4_y'
    ]

    try:
        df['max_width_norm'] = (df[x_coords].max(axis=1)-df[x_coords].min(axis=1))\
            /df['width']

        df['max_height_norm'] = (df[y_coords].max(axis=1)-df[y_coords].min(axis=1))\
            /df['height']

        # Create normalised centre point coordinates of horizontal bounding box with regards to image width and height
        df['x_centre_norm'] = df['x_centre']/df['width']

        df['y_centre_norm'] = df['y_centre']/df['height']
        return df
    except KeyError:
        logging.error("Invalid column: class referenced")


def get_img_size(filename):
    """Function that opens a provided file and retrieves its size in terms of width and height.
    
    Args:
      filename: File to be opened.
    Raises:
      IOError: When invalid file path is referenced.
    Returns:
      width, height: Image size in width and height.
    """
    try:
        with Image.open(filename.values[0]) as img:
            width, height = img.size
        
        return width, height
    except IOError:
        logging.error("Invalid file path referenced.")

def apply_class_mapping(df):
    """Function that re-maps/combines some of the class labels to for the purpose of having a continuous labelling value limited to 9 classes(represented by 1,2,3,...,9).

    In particular, the following mapping are as follows:
    1:1; (Car- No change)
    2:2; (Truck- No change)
    4:4; (Tractor- No change)
    5:5; (Camping Van- No change)
    7:8; (Motorbike mapped to others)
    8:8; (Bus to others)
    9:9; (Vans-No change)
    10:8; (Others-represented as label 8 instead of 10)
    11:6; (Pickup )
    23:3; (Boat represented as label 3 instead of 23)
    31:7; (Plane represented as label 7 instead of 31)

    Args:
      df: Dataframe considered.
    Raises:
      KeyError: When invalid column is referenced for data mapping.
    Returns:
      Dataframe with new class labels.
    """
    # Apply new mapping with defined dictionary of label mappings
    new_data = { 1:1,
                2:2,
                4:4,
                5:5,
                7:8,
                8:8,
                9:9,
                10:8,
                11:6,
                23:3,
                31:7
            }
    
    try:
        df['class'] = df['class'].replace(new_data)
        return df
    except KeyError:
        logging.error("Invalid column: class referenced")

def generate_annotation_per_file(df, annot_output_dir):
    """Function that generates the required YOLO annotation format for each image considered in a text file.

    Args:
      df: Dataframe considered.
      annot_output_dir: Directory where generated text file containing YOLO annotations for each image are stored
    Raises:
      KeyError: When invalid column is referenced.
    Returns:
      Dataframe with new class labels.
    """


    col_interest = ['annot_for_img_file',
                    'class',
                    'x_centre_norm',
                    'y_centre_norm',
                    'max_width_norm',
                    'max_height_norm']

    df = df[col_interest]

    # Get a set of unique annotation files, for which based on it, relevant annotations are being filtered from dataframe and save as text content in a text file 
    annot_files_for_img = set(df['annot_for_img_file'])
    for img_id in annot_files_for_img :
        try:
            temp_df = df[df['annot_for_img_file']==img_id].drop('annot_for_img_file', axis=1)
            txt_save_path = os.path.join(annot_output_dir, img_id)
            txt_format = ['%d', '%f', '%f', '%f', '%f']
            np.savetxt(txt_save_path, temp_df.values, fmt=txt_format, delimiter=" ")
        except IOError:
            logging.error("Unable to save annotations into %s", txt_save_path)

    return None

def process_annotation_to_yolo(annotation_file, annot_output_dir):
    """ This function processes the annotation files into a YOLO format.

    The annotation file are to be interpreted as follows:
    For each target and from left to right (one target per line), the image ID, the coordinates of the center in the image, the orientation of the vehicle, the 4 coordinates of the 4 corners, the class name, a flag stating if the target is entirely contained in the image, a flag stating if the vehicle is occluded.
    
    MAPPINGS:
    1: car, 2:trucks, 4: tractors, 5: camping cars, 7: motorcycles, 8:buses, 9: vans, 10: others, 11: pickup, 23: boats , 201: Small Land Vehicles, 31: Large land Vehicles
    """
    columns = ['Image_ID',
        'x_centre',
        'y_centre',
        'orientation',
        'corner1_x',
        'corner2_x',
        'corner3_x',
        'corner4_x',
        'corner1_y',
        'corner2_y',
        'corner3_y',
        'corner4_y',
        'class',
        'is_contained',
        'is_occluded']

    data = pd.read_table(annotation_file, delimiter=' ', names=columns)

    # Filter out annotations for vehicles which are located within image boundary
    contained_df = data[data['is_contained']==1].copy()

    # Generate filepath for each Image ID and its corresponding annotation txt file name. This is to facilitate the extraction of image sizing and also the subsequent annotation file that is to be generated for each image that contains the necessary annotation in YOLO format 
    contained_df['filepath'] = contained_df['Image_ID'].map(lambda x: os.path.join(os.getcwd(), 'Vehicles', 'CO', str(x).zfill(8) + '_co.png'))

    contained_df['annot_for_img_file'] = contained_df['Image_ID'].map(lambda x: str(x).zfill(8) + '.txt')

    # Get image resolution information
    contained_df['width'], contained_df['height'] = get_img_size(contained_df['filepath'])

    # Get normalised bounding box parameters and apply new class label mapping
    contained_df = normalise_bounding_box_val(contained_df)
    
    contained_df = apply_class_mapping(contained_df)
    
    # Create a folder and generate annotation for each image file
    if not os.path.exists(annot_output_dir):
        os.makedirs(annot_output_dir)

    generate_annotation_per_file(contained_df, annot_output_dir)
        
    return None

if __name__ == "__main__":
    # Define log file and logging configuration
    logname = "./process_annotations_errors.log"
    logging.basicConfig(filename = logname,
                        filemode = 'a',
                        format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt = '%H:%M:%S',
                        level = logging.DEBUG)
    # StreamHandler writes to std output
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("Running main function")

    #Parse in arguments from terminal
    #Argparser to read in command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_annotation_file', default='annotation1024_cleaned.txt',help="directory storing VEDAI default annotations")
    parser.add_argument('--output_annotation_dir',  default='Annotations_processed', help="directory to store processed annotations")
    args = parser.parse_args()
    
    process_annotation_to_yolo(args.input_annotation_file,args.output_annotation_dir)
