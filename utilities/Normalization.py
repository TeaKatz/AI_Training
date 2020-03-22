class Normalization:
	def __init__(self, epsilon=1e-9):
		"""
		epsilon is a constant value used to avoid division by zero
		"""
		self.epsilon = epsilon
		self.max = None
		self.min = None
	
	def fit(self, x):
		"""
		x is DataFrame or ndarray of shape (batch_size, feature_nums)
		"""
		self.max = x.max(axis=0)
		self.min = x.min(axis=0)
	
	def transform(self, x):
		"""
		x is DataFrame or ndarray of shape (batch_size, feature_nums)
		return ndarray
		"""
		if type(x) == pd.DataFrame:
			return ((x - self.min) / (self.max - self.min + self.epsilon)).to_numpy()
		else:
			return (x - self.min) / (self.max - self.min + self.epsilon)
	
	def fit_transform(self, x):
		"""
		x is DataFrame or ndarray of shape (batch_size, feature_nums)
		"""
		self.fit(x)
		return self.transform(x)