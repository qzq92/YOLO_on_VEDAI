
# Exploration of YOLOv7 object detection model on VEDAI dataset

As AI advances are progressingly at a rapid pace, the purpose of this work is to explore the use of open-sourced YOLOv7 object detection model that was introduced in 2022 to detect objects in images. 

Preliminary exploration of data is available in the attached Jupyter notebook *Data exploration.ipynb*

## YOLOv7 reference
- [YOLOv7 Github](https://github.com/WongKinYiu/yolov7/blob/main/README.md)
- [Research paper](https://arxiv.org/abs/2207.02696)
- [Official YOLO v7 Custom Object Detection Tutorial | Windows & Linux (Youtube)](https://www.youtube.com/watch?v=-QWxJ0j9EY8)

## Dataset download website
- [VEDAI](https://downloads.greyc.fr/vedai/)  #Under Download and Copyrights section of the webpage

### Statistics of data representation:
1246 coloured or infra-red images when resolution of 512 x 512 are used.
1268 coloured or infra-red images when resolution of 1024 x 1024 are used.
3757 annotations for images regardless of which resolution

Since there is no actual information on the class labels mappings from the paper itself, other than the number of class labels representation mentioned in their 10-fold validation protocol, the following are the likely mappings between vehicle type and labels based on annotation counts and visualisation made in the notebook.

|Given vehicle type|Total(paper from cross-val)|Closest annotation counts(unaccounted)|Deduced class number(s) from closest count|
|---|---|---|---|
|Boat|170|171(1)|23| 
|Camping (Van)|390|397(7)|5|
|Car|1340|1377(37)|1|
|Others|200|204(4)|10|
|Pickup|950|955(5)|11|
|Plane|47|48(1)|31|
|Tractor|190|190(0)|4|
|Truck|300|307(7)|2|
|Vans|100|101(1)|9|
|Bus(Not stated)|0|3(3)|8|
|Motorbike(Not stated)|0|4(4)|7|
|**Total**|3687|3757(70)|-|

### List of images in table without any annotation based on existence of file (for case of 1024 x 1024 resolution)

|S/N|Colored image|Infrared image|
|---|---|---|
|1.| 00000024_co.txt|00000024_ir.txt|
|2.| 00000028_co.txt|00000028_ir.txt|
|3.| 00000034_co.txt|00000034_ir.txt|
|4.| 00000039_co.txt|00000039_ir.txt|
|5.| 00000341_co.txt|00000341_ir.txt|
|6.| 00000365_co.txt|00000365_ir.txt|
|7.| 00000369_co.txt|00000369_ir.txt|
|8.| 00000411_co.txt|00000411_ir.txt|
|9.| 00000424_co.txt|00000424_ir.txt|
|10.| 00000425_co.txt|00000425_ir.txt|
|11.| 00000522_co.txt|00000522_ir.txt|
|12.| 00000560_co.txt|00000560_ir.txt|
|13.| 00000600_co.txt|00000600_ir.txt|
|14.| 00000606_co.txt|00000606_ir.txt|
|15.| 00000717_co.txt|00000717_ir.txt|
|16.| 00000878_co.txt|00000878_ir.txt|
|17.| 00000887_co.txt|00000887_ir.txt|
|18.| 00001143_co.txt|00001143_ir.txt|
|19.| 00001145_co.txt|00001145_ir.txt|
|20.| 00001185_co.txt|00001185_ir.txt|
|21.| 00001244_co.txt|00001244_ir.txt|
|22.| 00001248_co.txt|00001248_ir.txt|

## Research paper titled *Vehicle Detection in Aerial Imagery: A small target detection benchmark* by Sébastien Razakarivony and Frédéric Jurie link
- [Research Paper link](https://hal.archives-ouvertes.fr/hal-01122605v2/document)

## Notes:
The images are split across compressed tar files as indicated via part1, part2, ... on the page itself. Upon download, you would see the files extening with numeric extensions such as VehiculesXXX.tar.001, VehiculesXXX.tar.002, etc.... (XXX would be 512/1024 depending on which resolution option you choose). 

An *dl_extract_tar.sh* script is available which would execute necessary linux command to download and extract VEDAI images and annotations in the current folder. Subsequently, extracted images would be categorised belonging to coloured or infra-red into 'CO' or 'IR' subfolders under created *Vehicles* folder. Please extract them using necessary file extraction software such as 7zip after downloading them. 

By default, 1246 images are provided (for 512x512 resolution) or 1268 images (for 1024x1024 resolution) either in coloured or infrared versions as represented by a "co" or "ir" in the images' names. Despite the additional 22 images that were provided for the higher resolution option, there is no presence of any vehicles in these images, and hence would not be used for any object detection training

Index range of images used for training: 00000000 to 00000999 (total 979) 
Index range of images used for validation: 00001000 to 00001271 (total 267)

## Interpretation of annotations
Using annotations1024.txt as reference
```
00000000 290.348971 504.611640 3.012318 277 303 304 279 502 498 508 511 2 1 0
00000001 172.413736 406.184469 -0.013888 163 182 181 164 403 403 410 410 1 1 0
00000001 206.608929 405.621843 -0.011363 196 218 218 195 402 402 409 410 9 1 0
```

As stated on page 16 of the research paper, the original annotation file should be interpreted as follows, for each target and from left to right (one target per line), the image ID, the coordinates of the center in the image, the orientation of the vehicle, the 4 coordinates of the 4 corners, the class name, a flag stating if the target is entirely contained in the image (1 or 0), a flag stating if the vehicle is occluded (1 or 0).

In particular, the coordinates should be interpreted as follows (using the first entry as illustration):

```
 'x_center', 'y_center', 'orientation', 'corner1_x', 'corner2_x', 'corner3_x', 'corner4_x', 'corner1_y', 'corner2_y', 'corner3_y', 'corner4_y' 'class', 'is_contained', 'is_occluded'
 290.348971 504.611640 3.012318 277 303 304 279 502 498 508 511 2 1 0
```

## YOLOv7 utilisation for the purpose of model training

In progress...

All annotations are standardized to `<object-class> <x> <y> <width> <height>`, where:  

* `<object-class>` - integer number of object from 0 to (classes-1)  
* `<x> <y> <width> <height>` - float values relative to width and height of image, it can be set from 0.0 to 1.0  
* for example: `<x> = <absolute_x> / <image_width> or <height> = <absolute_height> / <image_height>`  
* **attention:** `<x> <y>` - are center of rectangle (are not top-left corner)  

