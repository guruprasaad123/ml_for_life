# imports
import numpy as np
from sklearn.cluster import KMeans
import cv2 # the computer vision library, from http://opencv.org/
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters=5)

img_array = cv2.imread('pic07.jpg') 

img_reshape = img_array.reshape(-1,3)

kmeans.fit(img_reshape)

print("Colors => {}".format(kmeans.cluster_centers_.astype(np.uint8)))

print("Labels => {}".format(np.unique(kmeans.labels_)))

def plot_colors(hist,centroids):
    bar = np.zeros((50,300,3),dtype=np.uint8)
    start_x = 0
    remove_list = np.array([[255,255,255],[0,0,0],[128,128,128]])
    centroids = centroids.astype(np.uint8)
    centroids = [x for x in centroids if x not in remove_list]

    for (percent,color) in zip(hist,centroids):
        end_x = start_x + (percent * 300)
        cv2.rectangle(bar,
                    (int(start_x),0),
                    (int(end_x),50),
                    color.astype(np.uint8).tolist(),
                    -1
                     )
        start_x = end_x
        
    return bar


def histogram(k_means):
    unique_length = len(np.unique(k_means.labels_))
    labels =  np.arange(0,unique_length+1)
    (hist,_bins )= np.histogram(k_means.labels_,bins=labels)
    hist = hist.astype(np.float)
    hist = hist/hist.sum()
    return hist

hist = histogram(kmeans)
print(" hist {}".format(hist.sum()))
bar = plot_colors(hist,kmeans.cluster_centers_)

plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
