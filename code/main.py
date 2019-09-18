import skmeans
import matplotlib.pyplot as plt
import learn_utils
import numpy as np
import import_dataset as imp
import detection

#

#Import dataset
dataset_array = imp.read("data/nab/realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv", 4000 )
print(dataset_array)
#dataset_array = dataset_array
train_array = dataset_array[0:1500]

#Training: learn the normal behaviour from a dataset with no errors, using clustering to find its segment centroids.
segment_len = 32
slide_len = 16
km = skmeans.K_means(20)

segments = km.segmentation(train_array, segment_len, slide_len)

#Use my kmeans algorithm
centr = km.fit(segments)

#Print some of the clusters (shapes) obtained
learn_utils.plot_waves(centr, step=2)

#Reconstruction
#_____________________________________________________________

#Introducing an anomaly
#dataset_array[1210:1300] = 0
#plt.plot(dataset_array) 
#plt.show()

#Use clusters to recostruct the data modified with the anomaly
segments = km.segmentation(dataset_array, segment_len, slide_len)

lab = km.predict(centr, segments)
print(lab)

#Print a recostructed segment as sample
segment = segments[0]
nearest_centroid = centr[lab[0]]
print(segment)
print(nearest_centroid)
plt.figure()
plt.plot(segment, label="Windowed segment")
plt.plot(nearest_centroid, label="Nearest centroid")
plt.legend()
plt.show()

#Use recostruction algorithm to find the noise
noise, mean = detection.reconstruction(dataset_array, segments, lab, centr, slide_len, segment_len)
print("noise")
print(noise)

#Use a treshold detector to find out the anomlies
anomalies = detection.anomaly_detection(noise,4)














