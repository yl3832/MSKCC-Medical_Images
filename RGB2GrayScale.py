from PIL import Image
import cv2
import numpy as np
import os

dir_in="./deblur_generate/Deblur_1531955259"
dir_out=dir_in+"_gray"

def rgb2grayscale(dir_in,dir_out):
    if not os.path.exists(dir_out):
            os.makedirs(dir_out)
    for files in os.listdir(dir_in):
        current_img_path = os.path.join(dir_in, files)
        img = cv2.imread(current_img_path)
        img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY)
        save_path=os.path.join(dir_out, files)
        print save_path
        cv2.imwrite(save_path, img )

rgb2grayscale(dir_in,dir_out)
