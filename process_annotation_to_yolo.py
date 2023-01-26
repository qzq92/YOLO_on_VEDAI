import argparse
import os
import sys
import logging


import numpy as np
import pandas as pd

from PIL import Image
from sklearn.model_selection import train_test_split


# DEFAULT PATH SETTINGS FOR TRAIN/VAL/TEST IMAGE/ANNOTATIONS
INPUT_ANNOTATION_FILE = os.path.join(os.getcwd(),\
                                    'annotation1024_cleaned.txt')
OUTPUT_ANNOT_TRAIN_FOLDER = os.path.join(os.getcwd(), 'train', 'labels')
OUTPUT_ANNOT_VAL_FOLDER = os.path.join(os.getcwd(), 'val', 'labels')

INPUT_VEHICLE_IMG_FOLDER = os.path.join(os.getcwd(), 'Vehicles', 'CO')
OUTPUT_IMG_TRAIN_FOLDER = os.path.join(os.getcwd(), 'train', 'images')
OUTPUT_IMG_VAL_FOLDER = os.path.join(os.getcwd(), 'val', 'images')
RESIZE_RES = 640

def normalise_bounding_box_val(df):
    """Function that generates the normalised bounding boxes' centre coordinates as well as its normalised width and height sizes with respect to image size. Should the normalised value exceed 0 or 1, it will be reset to within 0 and 1.

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
        # Ensure bounding box corner label coordinates does not exceed the resolution

        for x in x_coords:
            df[x] = np.where(df[x] > df['width'], df['width'], df[x])
            df[x] = np.where(df[x] < 0, 0, df[x])

        for y in y_coords:
            df[y] = np.where(df[y] > df['height'], df['height'], df[y])
            df[y] = np.where(df[y] < 0, 0, df[y])
       
        # Create normalised centre point coordinates/bbox width/height of horizontal bounding box with regards to image width and height

        df['max_bbox_width_norm'] = (df[x_coords].max(axis=1)-df[x_coords].min(axis=1))\
            /df['width']
        
        df['max_bbox_height_norm'] = (df[y_coords].max(axis=1)-df[y_coords].min(axis=1))\
            /df['height']

        # For x-centre 
        df['x_centre'] = np.where(df['x_centre'] > df['width'],
                                  df['width'],
                                  df['x_centre'])

        df['x_centre'] = np.where(df['x_centre'] < 0,
                                  0,
                                  df['x_centre'])

        df['x_centre_norm'] = df['x_centre']/df['width']
        
        # For y-centre 
        df['y_centre'] = np.where(df['y_centre'] > df['height'],
                          df['height'],
                          df['y_centre'])

        df['y_centre'] = np.where(df['y_centre'] < 0,
                                  0,
                                  df['y_centre'])


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
    # Apply new mapping with defined dictionary of label mappings (must be 0-based mapping)
    new_data = { 1:0,
                2:1,
                4:3,
                5:4,
                7:7,
                8:7,
                9:8,
                10:7,
                11:5,
                23:2,
                31:6
    }
  
    try:
        df['class'] = df['class'].replace(new_data)
        return df
    except KeyError:
        logging.error("Invalid column: class referenced")

def generate_annotation_per_image(df, img_file_list, annot_output_dir):
    """Function that generates the required YOLO annotation format for each image considered in a text file.

    Args:
      df: Dataframe considered.
      img_file_list: List of image file info which annotations are to be generated
      annot_output_dir: Directory where generated text file containing YOLO annotations for each image are stored.
    Raises:
      KeyError: When invalid column is referenced.
    Returns:
      Dataframe with new class labels.
    """
    col_interest = ['annot_for_img_file',
                    'class',
                    'x_centre_norm',
                    'y_centre_norm',
                    'max_bbox_width_norm',
                    'max_bbox_height_norm']

    df = df[col_interest]

    # Get a set of unique annotation files, for which based on it, relevant annotations are being filtered from dataframe and save as text content in a text file.
    
    for img_id in img_file_list:
        try:
            temp_df = df[df['annot_for_img_file']==img_id].drop\
              ('annot_for_img_file', axis=1)
            txt_save_path = os.path.join(annot_output_dir, img_id)
            txt_format = ['%d', '%f', '%f', '%f', '%f']
            np.savetxt(txt_save_path, temp_df.values, fmt=txt_format, delimiter=" ")
        except IOError:
            logging.error("Unable to save annotations into %s", txt_save_path)

    return None

def split_data_train_val_test(df, val_split_ratio):
    """This function splits the input dataframe into training, validation and testing datasets via label stratification identified by 'class' column.
    
    Args:
      df: Dataframe considered for splitting.
      val_split_ratio: Proportion of training data annotations to be used for validation (testing) purpose.

    Raises:
      KeyError: When invalid column is referenced.

    Returns:
      Two dataframe representing training, validation and testing sets with all columns retained.
    """

    # Segment dataframe into features and labels to allow sklearn library to do a stratification split based on class labels. Despite the fact that there are images with more than 1 annotation in it, the treatment treat each annotation as unique to each other and stratification only applies towards the class labels.
    try:
        logging.info("Spltting %s annotations based on %s validation split ratio", len(df), val_split_ratio)
        
        # Segregate features and labels
        X = df.drop(['class'], axis = 1)
        y = df['class']

        # Split training set into training and validation set
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split_ratio, random_state=42, stratify=y)

        # Concatenates features and class as output for annotation generation.
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)

        logging.info("Total annotation for training: %s", len(train_df))
        logging.info("Total annotation for validation: %s", len(val_df))

        return train_df, val_df
    except KeyError:
        logging.error("Invalid column: class referenced")

def copy_images_for_training(annot_file_list, src_img_folder, dest_img_folder):
    """This function copies each file from a given img_file_list(source) from src_img_folder to dest_img_folder(destination)

    Args:
      annot_file_list: set of annotation file information(.txt) to be referenced to guide which image are to be copied.
      src_img_folder: Source directory path where VEDAI images are to be copied from.
      dest_img_folder: Destination directory path where VEDAI images are to be copied to.
    Raises:
      PermissionError: When permission is denied or insufficient for copying.
    Returns:
      None
    """

    # String replacement of .txt to _co for pointing to the relevant image for copying.
    img_file_list = [filename.replace('.txt','_co.png') for filename in annot_file_list]

    for img_file in img_file_list:
        source_img_path = os.path.join(src_img_folder, img_file)
        renamed_img_file = img_file.replace('_co.png','.png')
        destination_path = os.path.join(dest_img_folder, renamed_img_file)
        try:
            logging.info("Resizing and copying image from %s to %s", source_img_path, destination_path)
            image = Image.open(source_img_path)
            image.thumbnail((RESIZE_RES, RESIZE_RES))
            image.save(destination_path)
        except PermissionError:
            logging.error("Permission denied. You could have open a file or directory without knowing. Please check and rerun the program again")
    return None


def main_process_annotation_to_yolo(sys_args):
    """This function processes the annotation files that would be splitted into train/test sets that adhere to YOLO format which is stored in created 'train' and 'test' folders.

    Args:
      sys_args: Arguments parsed during execution of python script.
    Raises:
      KeyError: When invalid column is referenced.
    Returns:
      Dataframe with new class labels.
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

    data = pd.read_table(sys_args.input_annotation_file, delimiter=' ', names=columns)

    # Filter out annotations for vehicles which are located within image boundary
    contained_df = data[data['is_contained']==1].copy()

    # Generate filepath for each Image ID and its corresponding annotation txt file name. This is to facilitate the extraction of image sizing and also the subsequent annotation file that is to be generated for each image that contains the necessary annotation in YOLO format 
    contained_df['filepath'] = \
      contained_df['Image_ID'].map(lambda x: os.path.join(os.getcwd(), 'Vehicles', 'CO', str(x).zfill(8) + '_co.png'))

    # ensure the suffix matches with the corresponding image file
    contained_df['annot_for_img_file'] = \
      contained_df['Image_ID'].map(lambda x: str(x).zfill(8) + '.txt')

    # Get image resolution information
    contained_df['width'], contained_df['height'] = \
      get_img_size(contained_df['filepath'])

    # Get normalised bounding box parameters and apply new class label mapping
    contained_df = normalise_bounding_box_val(contained_df)
    contained_df = apply_class_mapping(contained_df)

    # Split data into train/test sets and get a set of file names for each set which annotations are to be generated respectively. A temporary dataframe is used as a dummy to point to dataframe used for training and testing, simplify code processing.
    train_df, val_df = split_data_train_val_test(contained_df,
                                                sys_args.validation_ratio)
    train_test_state = ['train', 'val']

    for state in train_test_state:
        temp_df = pd.DataFrame()

        # Training case
        if state == 'train':
            annot_output_dir = sys_args.output_annotation_training_folder
            image_output_dir = sys_args.output_image_training_folder
            temp_df = train_df

        # Validation case (Testing case)
        else:
            annot_output_dir = sys_args.output_annotation_validation_folder
            image_output_dir = sys_args.output_image_validation_folder
            temp_df = val_df

        # Create a directory for storing images/annotations that would be used for YOLO model training
        if not os.path.exists(annot_output_dir):
            os.makedirs(annot_output_dir)

        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)

        # Generate YOLO format annotation file for each image list from training/testing dataset.
        annot_file_list = set(temp_df['annot_for_img_file'])
        logging.info("Number of image files to be copied for %s state: %s", state, len(annot_file_list))
        
        generate_annotation_per_image(temp_df, annot_file_list, annot_output_dir)
        
        # Copy over corresponding image from source destination to specified output directory as part of YOLO object detection model training requirement.
        copy_images_for_training(annot_file_list,
                                 sys_args.input_image_folder,
                                 image_output_dir)

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


    # Parse in arguments from terminal. Argparser to read in command line inputs
    parser = argparse.ArgumentParser()

    # For annotations
    parser.add_argument('--input_annotation_file',
                        default=INPUT_ANNOTATION_FILE,
                        help="file storing VEDAI default annotations")
    parser.add_argument('--output_annotation_training_folder',
                        default=OUTPUT_ANNOT_TRAIN_FOLDER,
                        help="YOLO annotations for each image used for training")
    parser.add_argument('--output_annotation_validation_folder',
                        default=OUTPUT_ANNOT_VAL_FOLDER, 
                        help="YOLO annotations for each image used for validation")

    # For images
    parser.add_argument('--input_image_folder',
                         default=INPUT_VEHICLE_IMG_FOLDER,
                         help="file directory storing VEDAI images")
    parser.add_argument('--output_image_training_folder',
                         default=OUTPUT_IMG_TRAIN_FOLDER,
                         help="file directory storing VEDAI images for model training")
    parser.add_argument('--output_image_validation_folder',
                        default=OUTPUT_IMG_VAL_FOLDER,
                        help="file directory storing VEDAI images for model validation")

    # For splitting
    parser.add_argument('--validation_ratio',
                         default=0.2, type=float,
                         help="Proportion of annotation to be used for validation")

    args = parser.parse_args()
    
    main_process_annotation_to_yolo(args)
