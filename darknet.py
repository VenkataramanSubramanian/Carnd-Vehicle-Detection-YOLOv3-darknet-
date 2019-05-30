from ctypes import *
import math
import random
import cv2
import time
from PIL import Image
from moviepy.editor import VideoFileClip
import os
import numpy as np

colour_dct={0:'STOP',1:'GO'}
color = { 'person':(255,0,0), 'car':(255,255,0), 'bicycle':(0,255,0),'truck':(0,0,255),'bus':(0,255,255),'motorbike':(255,0,0),'traffic light':(128,128,128)}

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    
path=os.getcwd()
lib = CDLL(path+"/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def traffic_light(traf_img):
    
    img_hsv=cv2.cvtColor(traf_img, cv2.COLOR_BGR2HSV)
  
    #Finding the low saturation value based on image 
    sum_saturation = np.sum(img_hsv[:,:,1]) 
    area = traf_img.shape[0]*traf_img.shape[1]
    sat_low = int(sum_saturation / area * 1.3)
      
    #Getting the red pixels in an image using red mask
    lower_red = np.array([150,sat_low,140])
    upper_red = np.array([180,255,255])
    mask_red = cv2.inRange(img_hsv, lower_red, upper_red)
    
    mask_red = mask_red
    
    #Getting the yellow pixels in an image using red mask
    lower_yellow = np.array([10,sat_low,140])
    upper_yellow = np.array([20,255,255])
    mask_yel = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    mask_yellow = mask_yel
    
    mask_stop=mask_red+mask_yellow

    output_hsv_stop = img_hsv.copy()
    output_hsv_stop[np.where(mask_stop==0)] = 0
    

    #Getting the green pixels in an image using red mask
    lower_green = np.array([30,sat_low,140])
    upper_green = np.array([80,255,255])
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

    mask_green = mask_green

    output_hsv_go = img_hsv.copy()
    output_hsv_go[np.where(mask_green==0)] = 0

    #Getting the output colour of the traffic light
    value_stop=np.count_nonzero(output_hsv_stop)
    value_go=np.count_nonzero(output_hsv_go)
    value=np.argmax([value_stop,value_go])

    return colour_dct[value]
    
def detect(image, thresh=.51, hier_thresh=.5, nms=.45):
    data = image.ctypes.data_as(POINTER(c_ubyte))
    im = ndarray_image(data, image.ctypes.shape, image.ctypes.strides)
    
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])

    traffic_img=image.copy()
    for i in res:
        detection=i[0].decode('utf-8')
        if(detection in ('person','bicycle','car','truck','bus','motorbike','traffic light')):
           color_local = color[detection]
           upper = (int(i[2][0]-i[2][2]/2),int(i[2][1]-i[2][3]/2))
           lower = (int(i[2][0]+i[2][2]/2),int(i[2][1]+i[2][3]/2))
           cv2.rectangle(image, upper, lower , color_local, thickness = 4)
           cv2.rectangle(image, (int(i[2][0]-i[2][2]/2),int(i[2][1]-i[2][3]/2-20)), 
                                (int(i[2][0]-i[2][2]/2 + 120),int(i[2][1]-i[2][3]/2)), color_local, thickness = -1)

           if(detection=='traffic light'):
              traf_img=traffic_img[int(i[2][1]-i[2][3]/2):int(i[2][1]+i[2][3]/2), int(i[2][0]-i[2][2]/2):int(i[2][0]+i[2][2]/2)]
              try:
                 colour= traffic_light(traf_img)                 
              except:
                 continue
              cv2.putText(image, '{0}'.format(colour),(int(i[2][0]-i[2][2]/2), 
                       int(i[2][1]-i[2][3]/2)  -6),cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 0, 0),1,cv2.LINE_AA)
           else:        
              cv2.putText(image, '{0}'.format(i[0].decode('utf-8')),(int(i[2][0]-i[2][2]/2), 
                       int(i[2][1]-i[2][3]/2)  -6),cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 0, 0),1,cv2.LINE_AA)              
     

    free_image(im) 
    free_detections(dets, num)
    return image
    

	
net = load_net(b"cfg/yolov3.cfg",b"yolov3.weights", 0)
meta = load_meta(b"cfg/coco.data")



    
    

