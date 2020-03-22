class Standardization:
	def __init__(self, epsilon=1e-9):
		"""
		epsilon is a constant value used to avoid division by zero
		"""
		self.epsilon = epsilon
		self.mean = None
		self.std = None

	def fit(self, x):
		"""
		epsilon is a constant value used to avoid division by zero
		"""
		N = x.shape[0]
		self.mean = np.sum(x, axis=0) / N
		self.std = np.sqrt(np.sum(np.square(x - self.mean), axis=0) / N)

	def transform(self, x):
		"""
		epsilon is a constant value used to avoid division by zero
		return ndarray
		"""
		if type(x) == pd.DataFrame:
			return ((x - self.mean) / (self.std + self.epsilon)).to_numpy()
		else:
			return (x - self.mean) / (self.std + self.epsilon)

	def fit_transform(self, x):
		"""
		epsilon is a constant value used to avoid division by zero
		"""
		self.fit(x)
		return self.transform(x)