import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read(src, n):
	dataset = pd.read_csv(src)
	print(dataset)
	dataset = dataset.value
	print(dataset)
	print(dataset[0])
	dataset_array = np.zeros(len(dataset))
	for i in range(0,len(dataset)):
		dataset_array[i] = dataset[i]

	plt.title(src) 
	plt.plot(dataset_array[0:n]) 
	plt.show()	

	print(dataset_array[0:n])

	return dataset_array[0:n]