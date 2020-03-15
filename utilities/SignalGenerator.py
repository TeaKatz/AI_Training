import numpy as np


class SignalGenerator:
	def __init__(self, period, signal_type="sine", amplitude=1, vertical_shift=0, phase_shift=0):
		assert signal_type.lower() in ["sine", "cos", "half-sine", "half-cos", "sawtooth"], "get unknown signal_type: '{}'".format(signal_type)
		
		self.period = period
		self.signal_type = signal_type.lower()
		self.amplitude = amplitude
		self.vertical_shift = vertical_shift
		self.phase_shift = phase_shift
		
	def _cal_x(self, t):
		"""
		Convert t into x which has range of [0, 1]
		"""
		x = np.mod(t + self.phase_shift, self.period)
		x = x / self.period
		return x
	
	def _cal_y(self, x):
		"""
		Calculate y from x according to signal type
		"""
		if self.signal_type == "sine":
			y = np.sin(2 * np.pi * x) * self.amplitude + self.vertical_shift
		elif self.signal_type == "cos":
			y = np.cos(2 * np.pi * x) * self.amplitude + self.vertical_shift
		elif self.signal_type == "half-sine":
			y = np.sin(2 * np.pi * (x - 0.5) / 2) * self.amplitude + self.vertical_shift
		elif self.signal_type == "half-cos":
			y = np.cos(2 * np.pi * x / 2) * self.amplitude + self.vertical_shift
		else:
			y = x * self.amplitude + self.vertical_shift
		return y
		
	def __call__(self, start_t, stop_t):
		t = np.arange(start_t, stop_t)
		x = self._cal_x(t)
		y = self._cal_y(x)
		return y
		