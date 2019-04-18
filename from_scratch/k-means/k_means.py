import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

def euclidian(a, b):
    return np.linalg.norm(a-b)

def kmeans(k=3,dataset=None,epsilon=0,distance=enclidian):
    history_centroids=[]
    num_rows , num_features = dataset.shape
    
    

    
