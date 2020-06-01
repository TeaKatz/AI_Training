import numpy as np


class EuclideanDistance:
	def __call__(self, samples, new_sample):
		"""
		Return Euclidean distance between new sample point and other sample points
		"""
		distance = np.sqrt(np.sum(np.square(samples - new_sample), axis=-1))
		return distance
