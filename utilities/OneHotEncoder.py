import numpy as np
import pandas as pd


class OneHotEncoder:
	def __init__(self, columns, inplace=True, drop=True):
		assert type(columns) == str or type(columns) == list, "columns must be string or list of string."
		
		self.columns = [columns] if type(columns) == str else columns
		self.inplace = inplace
		self.drop = drop
		self.min = []
		self.max = []
		self.class_nums = []
		
	def fit(self, data):
		assert type(data) == pd.DataFrame
		
		self.min = []
		self.max = []
		self.class_nums = []
		for column in self.columns:
			_min = min(data[column])
			_max = max(data[column])
			self.min.append(_min)
			self.max.append(_max)
			self.class_nums.append(_max - _min + 1)
		
	def transform(self, data):
		assert type(data) == pd.DataFrame
		
		if self.inplace:
			# Copy DataFrame
			data = data.copy()
		
		# Encode data
		for i, column in enumerate(self.columns):
			for j, value in enumerate(range(self.min[i], self.max[i] + 1)):
				encoded = np.equal(data[column], value).astype(np.int32)
				data["{}_{}".format(column, j)] = encoded
			if self.drop:
				# Drop encoded column
				data.drop(columns=column, inplace=True)
		return data
	
	def fit_transform(self, data):
		self.fit(data)
		return self.transform(data)