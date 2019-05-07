# imports
import cv2 # the computer vision library, from http://opencv.org/
import numpy as np # wasn't needed but is always imported in examples
import matplotlib.pyplot as plt
# make it easy to modify the program to use other pictures
img_path = './'
img_base_name = 'pic' 

# ===== delete blue =====
# import the picture
img_array = cv2.imread(img_path+img_base_name+'.jpg') 
# note that [:,:,0] is blue, [:,:,1] is green, [:,:,2] is red 

print(img_array.shape)
img_array[:,:,0] = 0 
# write the image
cv2.imwrite(img_path+img_base_name+'_no_blue.jpg',img_array) 

# ===== delete green =====
img_array = cv2.imread(img_path+img_base_name+'.jpg') 
img_array[:,:,1] = 0
cv2.imwrite(img_path+img_base_name+'_no_green.jpg', img_array) 

# ===== delete red =====
img_array = cv2.imread(img_path+img_base_name+'.jpg')
img_array[:,:,2] = 0
cv2.imwrite(img_path+img_base_name+'_no_red.jpg', img_array)

#==== delete 
img_array = cv2.imread(img_path+img_base_name+'.jpg')

img_array= np.where( img_array == [255,255,255] , 0)

cv2.imwrite(img_path+img_base_name+'_no_white.jpg', img_array)