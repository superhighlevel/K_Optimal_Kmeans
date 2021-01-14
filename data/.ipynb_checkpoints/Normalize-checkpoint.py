import numpy as np 
import pandas as pd 
class XX:
	def normalize(X):
		n,d = X.shape
		temp = np.zeros(shape=(n,d))
		for i in range(d):
			mean = np.mean(X[:,i])
			std = np.std(X[:,i])
			temp[:,i] = (X[:,i]-mean)/std
		return temp