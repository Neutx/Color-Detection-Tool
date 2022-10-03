#Image color analysis with K-Means Clustering algorithm

#required libraires
#scikit-learn, matplotlib, opencv-python, numpy
#pip install scikit-learn, matplotlib, opencv-python, numpy

#What is K-Mean clustering
#K-Mean clustering Algorithm
#Color Analysis with K-Mean Clustering







from collections import Counter
from configparser import Interpolation
from sklearn.cluster import KMeans
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import cv2

img_name = 'img2.png'
raw_img = cv2.imread(img_name)
raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
print(raw_img.shape[0])
print(raw_img.shape[1])

def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        i = int(i)
        hex_color += ("{:02x}".format(i))
    return hex_color

img = cv2.resize(raw_img, (900, 600), interpolation = cv2.INTER_AREA)
img = img.reshape(img.shape[0]*img.shape[1], 3)

clf = KMeans(n_clusters =5)
color_labels = clf.fit_predict(img)
center_colors = clf.cluster_centers_
counts = Counter(color_labels)
ordered_colors = [center_colors[i] for i in counts.keys()]
hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
plt.figure(figsize = (12, 8))
plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
plt.savefig(f"{img_name[:-4]}-analysis.png")
print(hex_colors)
