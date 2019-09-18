from sklearn.metrics import pairwise_distances_argmin
import numpy as np

class K_means:

	def __init__(self, k = 3, tolerance = 0.0000000000000000001, max_iterations = 100):
		self.k = k
		self.tolerance = tolerance
		self.max_iterations = max_iterations
		self.rng = np.random.RandomState(2)

	def segmentation(self, dataset, segment_len, slide_len):
		segments = []
		for start_pos in range(0, len(dataset), slide_len):
		    end_pos = start_pos + segment_len
		    segment = np.copy(dataset[start_pos:end_pos])
		    if len(segment) != segment_len:
		        continue
		    segments.append(segment)

		print("Produced %d waveform segments" % len(segments))
		print(segments)
	
		return segments

	def farthest(self, segments, distances, labels):
	    	counts = np.bincount(labels)
	    	print("FARTHEST")
	    	print(counts)
	    	print(np.argmax(counts))
	    	print(distances[:, np.argmax(counts)])
	    	print(np.argmax(distances[:, np.argmax(counts)],axis =0))
	    	new_centroid_idx = np.argmax(distances[:, np.argmax(counts)],axis =0)
	    	new_centroid = segments[new_centroid_idx]
	    	print(new_centroid)
	
        		
	    	return new_centroid	

	def fit(self, X):
	    z = self.rng.permutation(len(X))[:self.k]
	    X = np.array(X, dtype= "float64")
	    centers = X[z]

	    distances = np.zeros((len(X),self.k))
	    labels = np.zeros(len(X))
	    it = 0

	    while True:
	        for i in range(self.k):
        		distances[:,i] = np.linalg.norm(X - centers[i], axis=1)	
        	labels= np.argmin(distances, axis = 1)
        	
        	print("Labels")
        	print(labels)
        	print("DISTANCES")
        	print(distances)

       		center_temp = []

       		for i in range(self.k):
       			if(i in labels):
	        		new_centers = X[labels == i].mean(0)
	        		center_temp.append(new_centers)
	        	else:
	        		center_temp.append(self.farthest(X, distances, labels))
       		
	        new_centers = np.array(center_temp, dtype= "float64")

	        it = it + 1
	        
	        if np.all(centers == new_centers) or it == self.max_iterations:
	            break

	        centers = new_centers
	    
	    return centers

	def predict(self, centers, windowed_segments):
		distances = np.zeros((len(windowed_segments), self.k))
		for i in range(self.k):
			distances[:,i] = np.linalg.norm(windowed_segments - centers[i], axis=1)

		labels= np.argmin(distances, axis = 1)
		print("Predicted Labels")
		print(labels)

		return labels



	    

	    	