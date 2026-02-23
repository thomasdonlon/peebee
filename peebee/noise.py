"""
This submodule defines various noise models that can be used when optimizing acceleration models.
"""

import numpy as np
from .glob import kpcs2tommsyr

#TODO: Manually converting into mm/s/yr = km/s/Myr for numerical stability, but we should be using astropy units ideally

#-----------------------------------------------------------------
# Noise Model Base Class
#-----------------------------------------------------------------

class NoiseModel:
	"""Base class for noise models used in fitting."""
	
	def __init__(self, **kwargs):
		self.name = 'Uninitialized'
		self.param_names = []
		self.params = {}
		
		# Let subclasses handle parameter setting after they've set up their structure
		# Parameters will be set by subclass calling _finish_init if needed
	
	def _finish_init(self, **kwargs):
		"""Called by subclasses after they've set up param_names and default params."""
		if kwargs:
			self.set_params(kwargs)
	
	@property
	def nparams(self):
		"""Number of parameters in the noise model."""
		return len(self.param_names)
	
	def set_params(self, params):
		"""Set noise model parameters from dictionary or list."""
		if isinstance(params, dict):
			for param_name, value in params.items():
				if param_name in self.param_names:
					self.params[param_name] = value
				else:
					raise ValueError(f"Parameter '{param_name}' not found in {self.name} noise model. Available: {self.param_names}")
		elif isinstance(params, (list, tuple, np.ndarray)):
			if len(params) != self.nparams:
				raise ValueError(f"{self.name} noise model requires {self.nparams} parameters, got {len(params)}")
			for i, value in enumerate(params):
				self.params[self.param_names[i]] = value
		else:
			raise ValueError("Parameters must be provided as dict, list, tuple, or array")
	
	def get_params(self):
		"""Get current parameter values as dictionary."""
		return self.params.copy()
	
	def likelihood(self, residuals):
		"""Calculate likelihood contribution for given residuals. Must be implemented by subclasses."""
		raise NotImplementedError(f"Uninitialized noise model has no likelihood() method")

#-----------------------------------------------------------------
# Specific Noise Model Classes  
#-----------------------------------------------------------------

class GaussianNoise(NoiseModel):
	"""Gaussian noise model with parameter sigma."""
	
	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Gaussian'
		self.param_names = ['sigma']
		self.params = {'sigma': 1.0}  # Default value
		self._finish_init(**kwargs)
	
	def likelihood(self, residuals):
		"""Gaussian likelihood: -ln(L) = 0.5*ln(2*pi*sigma^2) + 0.5*residuals^2/sigma^2"""
		sigma = self.params['sigma']
		residuals *= kpcs2tommsyr
		return np.sum(0.5*np.log(2*np.pi*sigma**2) + 0.5*residuals**2/sigma**2)

class LorentzNoise(NoiseModel):
	"""Lorentz (Cauchy) noise model with parameter gamma."""
	
	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Lorentz'  
		self.param_names = ['gamma']
		self.params = {'gamma': 1.0}  # Default value
		self._finish_init(**kwargs)
	
	def likelihood(self, residuals):
		"""Lorentz likelihood: -ln(L) = ln(pi*gamma) + ln(1 + residuals^2/gamma^2)"""
		gamma = self.params['gamma']
		residuals *= kpcs2tommsyr
		return np.sum(np.log(np.pi*gamma) + np.log(1 + residuals**2/gamma**2))

class PowerLawNoise(NoiseModel):
	"""Power law noise model with parameter zeta (Moran et al. 2024)."""
	
	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'PowerLaw'
		self.param_names = ['zeta']
		self.params = {'zeta': 2.0}  # Default value
		self._finish_init(**kwargs)
	
	def likelihood(self, residuals):
		"""Power law likelihood: -ln(L) = zeta*ln(residuals) + ln(r_min^(1-zeta) - r_max^(1-zeta)) - ln(zeta-1)"""
		zeta = self.params['zeta']
		residuals *= kpcs2tommsyr
		return np.sum(zeta*np.log(residuals) + np.log(np.min(residuals)**(1 - zeta) - np.max(residuals)**(1-zeta)) - np.log(zeta - 1))