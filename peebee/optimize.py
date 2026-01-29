"""
This submodule provides an object-oriented fitter for fitting gravitational potential models 
to pulsar acceleration data. The Fitter class provides a clean interface and simplified parameter 
management using qualified parameter names.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import astropy.units as u 

from .transforms import convert_to_frame
from .models import Model, CompositeModel
from .glob import kpctocm, r_sun
from .noise import get_noise_model_likelihood

#-----------------------------------------------------------------
# RESULTS CLASS
#-----------------------------------------------------------------

class FitResults:
	"""Container for optimization results."""
	
	def __init__(self):
		self.success = False
		self.best_fit_params = None
		self.uncertainties = None
		self.reduced_chi2 = None
		self.aic = None
		self.scipy_result = None
		self.hessian = None
		self.hessian_inv = None
		self.message = ""

	def __repr__(self):
		if self.success:
			return f"FitResults(success=True, reduced_chi2={self.reduced_chi2:.3f}, aic={self.aic:.3f})"
		else:
			return f"FitResults(success=False, message='{self.message}')"

#-----------------------------------------------------------------
# FITTER CLASS
#-----------------------------------------------------------------

class Fitter:
	"""
	Object-oriented fitter for gravitational potential models.
	
	Usage:
		fitter = Fitter(model)
		fitter.set_data(l, b, d, alos, alos_err)
		fitter.configure_params({"NFW.m_vir": (1e10, 1e14), "NFW.r_s": (5, 50)})
		result = fitter.optimize(method='differential_evolution')
	"""
	
	def __init__(self):
		self.model = None
		self.data = None
		self.param_bounds = {}
		self.results = None
		self._best_fit_params = None
		
		# Configuration options
		self.sun_pos = (r_sun, 0., 0.)
		self.negative_mass = False #TODO: remove the negative mass stuff
		self.scale = 1.0 #TODO: remove the scale stuff, that's just for internal testing
	
	def set_model(self, model):
		"""Set the gravitational potential model to fit."""
		self.model = model
		# Clear any existing configuration that might be invalid
		self.param_bounds = {}
		self._best_fit_params = None
		self.results = None
	
	@convert_to_frame('gal')
	def set_data(self, l, b, d, alos, alos_err, frame='gal', sun_pos=None):
		"""
		Set the pulsar acceleration data.
		
		:l: Galactic longitude (deg) 
		:b: Galactic latitude (deg)
		:d: Heliocentric distance (kpc)
		:alos: Observed line-of-sight acceleration (cm/s^2)
		:alos_err: Uncertainty in line-of-sight acceleration (cm/s^2) 
		:frame: Coordinate frame ('gal', 'cart', 'icrs', 'ecl')
		:sun_pos: Solar position (kpc) [optional, uses default if not provided]
		"""
		if sun_pos is not None:
			self.sun_pos = sun_pos
			
		self.data = {
			'l': np.array(l),
			'b': np.array(b), 
			'd': np.array(d),
			'alos': np.array(alos),
			'alos_err': np.array(alos_err)
		}
	
	def configure_params(self, param_bounds_dict):
		"""
		Configure which parameters to optimize and their bounds.
		
		:param_bounds_dict: Dictionary of parameter names and bounds
			Example: {"NFW.m_vir": (1e10, 1e14), "NFW.r_s": (5, 50)}
		"""
		if self.model is None:
			raise ValueError("Must set model before configuring parameters")
			
		# Validate parameter names
		available_params = set()
		if isinstance(self.model, CompositeModel):
			available_params = set(self.model.param_names)
		else:
			available_params = set(self.model.param_names)
			
		for param_name in param_bounds_dict.keys():
			if param_name not in available_params:
				raise ValueError(f"Parameter '{param_name}' not found in model. Available: {sorted(available_params)}")
		
		self.param_bounds = param_bounds_dict.copy()
	
	def set_optimization_options(self, sun_pos=None, negative_mass=False, scale=1.0):
		"""Set additional optimization options."""
		if sun_pos is not None:
			self.sun_pos = sun_pos
		self.negative_mass = negative_mass
		self.scale = scale
	
	@property
	def best_fit_params(self): #XXX come back here
		"""Best-fit parameters from most recent optimization."""
		return self._best_fit_params
	
	def _extract_param_values(self, param_vector):
		"""Extract parameter values for optimization."""
		param_dict = {}
		param_names = list(self.param_bounds.keys())
		for i, param_name in enumerate(param_names):
			param_dict[param_name] = param_vector[i]
		return param_dict
	
	def _get_current_optimization_values(self):
		"""Get current parameter values in optimization space (log space for log params)."""
		if isinstance(self.model, CompositeModel):
			opt_params = self.model.get_optimization_params() #XXX check this
		else:
			opt_params = self.model.get_optimization_params()
		
		# Extract only the parameters we're optimizing
		current_values = []
		for param_name in self.param_bounds.keys():
			if param_name in opt_params:
				current_values.append(opt_params[param_name])
			else:
				# Handle case where qualified name might not match exactly
				if isinstance(self.model, CompositeModel):
					raise ValueError(f"Parameter '{param_name}' not found in model optimization parameters")
				else:
					# For single models, try without prefix
					if '.' in param_name:
						_, clean_name = param_name.split('.', 1)
						if clean_name in opt_params:
							current_values.append(opt_params[clean_name])
						else:
							raise ValueError(f"Parameter '{param_name}' not found in model")
		return np.array(current_values)
	
	def _objective_function(self, param_vector):
		"""Objective function for scipy optimizers."""
		# Update model parameters
		param_dict = self._extract_param_values(param_vector)
		self._update_model_params(param_dict)
		
		# Calculate model predictions
		model_alos = self.model.alos(self.data['l'], self.data['b'], self.data['d'], sun_pos=self.sun_pos)
		
		if self.negative_mass:
			model_alos *= -1
		
		# Calculate residual sum of squares
		rss = 0.5 * np.sum((model_alos * kpctocm - self.data['alos'])**2 / (self.data['alos_err']**2)) * self.scale
		
		# Add noise model contribution if specified
		if self.noise_model != 'none':
			resids = np.abs(model_alos * kpctocm - self.data['alos'])
			noise_like = get_noise_model_likelihood(self.noise_model, resids, param_vector)
			rss += noise_like
		
		return rss
	
	def _update_model_params(self, param_dict):
		"""Update model parameters with qualified names using log-aware methods."""
		if isinstance(self.model, CompositeModel):
			self.model.set_optimization_params(param_dict)
		else:
			# For single models, strip any prefix if present and create clean param dict
			clean_params = {}
			for param_name, value in param_dict.items():
				if '.' in param_name:
					_, clean_name = param_name.split('.', 1)
					clean_params[clean_name] = value
				else:
					clean_params[param_name] = value
			
			self.model.set_optimization_params(clean_params)
	
	def optimize(self, method='differential_evolution', **kwargs):
		"""
		Run optimization to fit model parameters.
		
		:method: Optimization algorithm ('differential_evolution' or 'gradient_descent')
		:kwargs: Additional arguments passed to scipy optimizer
		"""
		if self.model is None:
			raise ValueError("Must set model before optimizing")
		if self.data is None:
			raise ValueError("Must set data before optimizing")
		if not self.param_bounds:
			raise ValueError("Must configure parameters before optimizing")
		
		# Prepare bounds and initial values
		param_names = list(self.param_bounds.keys())
		bounds = [self.param_bounds[name] for name in param_names]
		
		# Get current parameter values in optimization space (log space for log params)
		current_opt_values = self._get_current_optimization_values()
		
		results = FitResults()
		
		try:
			if method == 'gradient_descent' or method == 'gd':
				# Gradient descent requires initial guess
				if 'x0' not in kwargs:
					# Use current parameter values in optimization space as initial guess
					kwargs['x0'] = current_opt_values
				
				scipy_result = minimize(self._objective_function, bounds=bounds, **kwargs)
				
			elif method == 'differential_evolution' or method == 'de':
				# Set reasonable population size if not provided
				if 'popsize' not in kwargs:
					kwargs['popsize'] = 10 * len(param_names)
				
				scipy_result = differential_evolution(self._objective_function, bounds, **kwargs)
				
			else:
				raise ValueError(f"Unknown optimization method: {method}")
			
			# Store results
			results.scipy_result = scipy_result
			results.success = scipy_result.success
			
			if results.success:
				# Extract best parameters
				best_param_dict = self._extract_param_values(scipy_result.x)
				results.best_fit_params = best_param_dict
				self._best_fit_params = best_param_dict.copy()
				
				# Update the model with best-fit parameters
				self._update_model_params(best_param_dict)
				
				# Calculate fit statistics
				n_data = len(self.data['alos'])
				n_params = len(param_names)
				
				if n_data > n_params:
					results.reduced_chi2 = scipy_result.fun / (n_data - n_params)
					results.aic = 2 * n_params + scipy_result.fun
					
					# Calculate parameter uncertainties
					try:
						results.uncertainties = self._calculate_uncertainties(scipy_result.x, param_names)
					except Exception as e:
						print(f"Warning: Could not calculate uncertainties: {e}")
						results.uncertainties = {name: np.nan for name in param_names}
				else:
					print("Warning: Not enough datapoints to calculate fit statistics!")
					results.reduced_chi2 = np.nan
					results.aic = np.nan
					results.uncertainties = {name: np.nan for name in param_names}
					
			else:
				results.message = f"Optimization failed: {scipy_result.message}"
				
		except Exception as e:
			results.success = False
			results.message = f"Optimization error: {str(e)}"
		
		self.results = results
		return results
	
	def _calculate_uncertainties(self, best_params, param_names, h=1e-4):
		"""Calculate parameter uncertainties using finite difference Hessian."""
		n_params = len(best_params)
		hess = np.zeros((n_params, n_params))
		
		# Calculate step sizes
		dparams = h * np.abs(np.array(best_params))
		dparams = np.where(dparams == 0, h, dparams)  # Avoid zero steps
		
		# Calculate Hessian using finite differences
		for i in range(n_params):
			for j in range(i, n_params):
				p = best_params.copy()
				
				# f(x+hi, y+hj)
				p[i] += dparams[i]
				p[j] += dparams[j]
				f_pp = self._objective_function(p)
				
				# f(x+hi, y-hj)  
				p = best_params.copy()
				p[i] += dparams[i]
				p[j] -= dparams[j]
				f_pm = self._objective_function(p)
				
				# f(x-hi, y+hj)
				p = best_params.copy()
				p[i] -= dparams[i] 
				p[j] += dparams[j]
				f_mp = self._objective_function(p)
				
				# f(x-hi, y-hj)
				p = best_params.copy()
				p[i] -= dparams[i]
				p[j] -= dparams[j]
				f_mm = self._objective_function(p)
				
				# Second derivative
				hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * dparams[i] * dparams[j])
				hess[j, i] = hess[i, j]  # Symmetric
		
		# Calculate uncertainties
		try:
			hess_inv = np.linalg.inv(hess)
			errors = np.sqrt(np.diag(hess_inv) / len(self.data['alos']))
			uncertainty_dict = {name: err for name, err in zip(param_names, errors)}
			return uncertainty_dict
		except np.linalg.LinAlgError:
			raise ValueError("Hessian matrix is singular - cannot calculate uncertainties")
	
	def evaluate_model(self, print_out=True):
		"""Evaluate current model fit without optimization."""
		if self.model is None or self.data is None:
			raise ValueError("Must set both model and data before evaluation")
		
		model_alos = self.model.alos(self.data['l'], self.data['b'], self.data['d'], sun_pos=self.sun_pos)
		rss_val = 0.5 * np.sum((model_alos * kpctocm - self.data['alos'])**2 / (self.data['alos_err']**2))
		
		n_data = len(self.data['alos']) 
		n_params = self.model.nparams
		
		if n_data > n_params:
			chi2 = rss_val / (n_data - n_params)
			aic = 2 * n_params + rss_val
		else:
			chi2 = np.nan
			aic = np.nan
		
		if print_out:
			print(f'RSS: {rss_val}')
			print(f'Reduced chi^2: {chi2}')
			print(f'AIC: {aic}')
			print(f'Model predictions: {model_alos * kpctocm}')
			print(f'Observations: {self.data["alos"]}')
			print(f'Residuals: {model_alos * kpctocm - self.data["alos"]}')
		
		return chi2, aic
