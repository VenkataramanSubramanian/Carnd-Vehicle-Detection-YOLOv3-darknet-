# Carnd-Vehicle-Detection-YOLOv3-darknet-
This is the yolov3 darknet implementation of fifth project of Udacity term-1 carnd nano-degree program

## **OVERVIEW**

Uses the lastest version of yolo in darknet to completed the project.

## **REQUIREMENTS**

Pre-install the environment of carnd nanaodegree program( mainly moviepy and cv for this code)

## **INSTALLING DARKNET**

If you have both 

NUMPY  in  -I/usr/include/python2.7/ -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy/

CUDA   in  -I/usr/local/cuda/include/ 


present in the above location in your gpu system u can use the libdarknet.so already present in the folder

If you dont have a GPU system or numpy and cuda in specified locations, follow the below steps

  1. Download the file by git clone https://github.com/pjreddie/darknet
  2. Implement the changes from the issue in https://github.com/pjreddie/darknet/issues/289 to load the image 
     using moviepy or numpy and predict
  3. make the code
  4. Now a new libdarknet.so will be present in the folder copy that and replace in this folder
  
## **EXECUTION OF THE CODE**

**python abstract.py input_video_file_name output_video_file_name**
