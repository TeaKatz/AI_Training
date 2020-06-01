import numpy as np


class ManhattanDistance:
	def __call__(self, samples, new_sample):
		""" 
		Return Manhanttan distance between new sample point and other sample points
		"""
		distance = np.sum(np.abs(samples - new_sample), axis=-1)
		return distance
