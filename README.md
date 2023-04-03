# Automated object detection and tracking in Python
### This repo is for automated object detection and tracking across several tiffs in the tiff file. These functions will created binary masks for bright spots or "objects" in each image. 
### The x and y coordinates of each object is then calculated by averaging the x and y positions of the pixels in each object mask. The coordinates are expected to be fairly precise, (plus or minus 3 to 4 pixels)

#### This script detects objects by a series of image processing step (segmentation). This includes, binarization, dilation of pixels, filling holes in objects, and eroding the final masks to undo the dilation. These steps robustly lead to identification of bright objects in each image.
#### Binarization of the image is done by calclating the background signal intensity and reming any signal that is lower than 2x the background intensity. The user can provide a selected region for calculating the background intensity, or manually define the binzation threshold. 
#### Dilation and hole filling is then conducted to ensure object masks are full/complete. With out this step, objects masks can be non uniform and fail to capture the entire object. 
#### The masks are then eroded to undo the dilation step to return the object to its representative size. 

#### Once each image is segmented. The scrip will then link similar objects together across frames. There are several, more sophisticated object tracking algorithms available, however, this is a simple method that links objects by searching the previous frame for another object similar in proximity. This proximity search can be optionally increased or decreased by the user.   

### Pros: Object masks can be used to calculate pixel intensities for each object in the image. This data is typically collected from microscopy imaging experiments.
### Improvements: If time was not a constaint, object tracking might be improved for new experiments that might contain more dense spots, spots that move faster, spots that divide (i.e. human cells), and other cases where more sophisticated tracking techniques are required (i.e. Kalman filter)
 

#### Getting started: navigate to the local folder location and create a new environment
```
conda env create -f environment.yml
```

#### Activate environment
```
conda activate Eikon_take_home
```

#### Process the tiffs, rhis will begin the processing for the tiff file that is included in this repo
```
example.py
```


