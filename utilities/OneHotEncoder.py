class OneHotEncoder:
	def __init__(self):
		self.classes = None
	
	def fit(self, x):
		"""
		x is Series of shape (batch_size, )
		"""
		self.classes = list(set(x))
	
	def transform(self, x):
		"""
		x is Series of shape (batch_size, )
		"""
		encoded = np.zeros([len(x), len(self.classes)])
		for i, _class in enumerate(self.classes):
			match_indices = np.where(x == _class)[0]
			encoded[match_indices, i] = 1
		return encoded
	
	def fit_transform(self, x):
		"""
		x is Series of shape (batch_size, )
		"""
		self.fit(x)
		return self.transform(x)