import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats


def anomaly_detection(noise, coef):
	
	pos_treshold = np.mean(noise) + coef * np.std(noise)
	neg_treshold = np.mean(noise) - coef * np.std(noise)
	print(neg_treshold)
	print(pos_treshold)
	print(np.mean(noise))
	anomalies = []
	for i in range(0,len(noise)):
		if(noise[i] <= neg_treshold):
			anomalies.append((i,noise[i]))
		if(noise[i] > pos_treshold):
			anomalies.append((i, noise[i]))
	print(anomalies)
	print("ANOMLIESS")
	noise = np.array(anomalies, dtype= "float64")
	print(noise[:])
	plt.hist(noise[:])
	plt.show()
		
	return noise

def reconstruction(data, segments, label, centroid, slide_len, segment_len):
	n = 0
	reconstruction = np.zeros(len(data)) 
	win_seg = np.zeros(len(data)) 
	pos = 0
	for segment in segments:
		nearest_centroid = centroid[label[n]]
		win_seg[pos:pos+segment_len] += segment
		reconstruction[pos:pos+segment_len] += nearest_centroid
		print("RECONSTRUCTION")
		print(reconstruction)
		pos = n * slide_len
		n = n + 1

	
	noise = reconstruction[0:len(reconstruction)] - win_seg[0:len(win_seg)]
	print(np.mean(noise))
	print(np.median(noise))
	print("Len noiseee")
	print(len(noise))
	n_plot_samples = len(noise)

	plt.plot(win_seg[0:n_plot_samples], label="Original waveform")
	plt.plot(reconstruction[0:n_plot_samples], label="Reconstructed waveform")
	plt.plot(noise[0:n_plot_samples], label="Noise")
	plt.legend()
	plt.show()
	plt.hist(noise)
	plt.show()

	return noise, np.mean(noise)

	








	
