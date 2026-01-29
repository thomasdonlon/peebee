"""
This submodule contains a few sampling routines for generating accelerations from populations of objects, which can be used for random bootstrapping or unit tests.
"""

import numpy as np
from .glob import r_sun, fix_arrays

#===============================================================================
# Functions
#===============================================================================

#generate a sample of n alos sources from a given model according to a spherically symmetric power law centered on COM and with index gamma
#the relative likelihood of finding a source at distance r from the COM is then $/rho /propto (r/r_0)^\gamma$. 
def sample_alos_sources_RPL(model, n, com, rscale, gamma, low_lim=0.01, up_lim=5, sun_pos=(r_sun, 0., 0.)):
	#low_lim: sets the lower bound of the distribution at low_lim*rscale
	#up_lim: sets the upper bound of the distribution at up_lim*rscale

	#generate uniform random variables and convert them to a power law distribution
	urv_uni = np.random.random(size=n)
	urv_pow = ((up_lim**(gamma + 1) - low_lim**(gamma + 1))*urv_uni + low_lim**(gamma + 1))**(1/(gamma+1))

	#generate cartesian coordinates
	phi = np.random.random(size=n)*2*np.pi
	theta = (np.random.random(size=n)-0.5)*np.pi
	r = urv_pow*rscale

	x = r*np.cos(phi)*np.cos(theta) + com[0]
	y = r*np.sin(phi)*np.cos(theta) + com[1]
	z = r*np.sin(theta) + com[2]

	alos = model.alos(x, y, z, frame='cart', sun_pos=sun_pos)

	return x, y, z, alos

#generate a sample of n alos sources from a given model over a uniform volume 
#bounds = ((x_min, x_max), (y_min, y_max), (z_min, z_max))
def sample_alos_sources_uniform(model, n, bounds, sun_pos=(r_sun, 0., 0.)):

	#generate cartesian coordinates from uniform random variables
	x = np.random.random(size=n)*(bounds[0][1] - bounds[0][0]) + bounds[0][0]
	y = np.random.random(size=n)*(bounds[1][1] - bounds[1][0]) + bounds[1][0]
	z = np.random.random(size=n)*(bounds[2][1] - bounds[2][0]) + bounds[2][0]

	alos = model.alos(x, y, z, frame='cart', sun_pos=sun_pos)

	return x, y, z, alos

#perturb a set of values with a few different noise models
@fix_arrays
def perturb_value(value, value_unc, noise_model='gaussian', relative_err=False):
	#value: array of values to be perturbed
	#value_unc: either a single uncertainty value (float) or an array of uncertainties the same length as value
	#noise_model: string specifying the noise model to use. 
	#relative_err: if True, value_unc is treated as a fractional uncertainty rather than an absolute one

	if relative_err:
		noise_level = value_unc * np.abs(value)
	else:
		noise_level = value_unc

	if noise_model == 'gaussian':
		value_perturbed = value + noise_level*np.random.normal(loc=0., scale=1., size=len(value))

	elif noise_model == 'lorentzian':
		value_perturbed = value + noise_level*np.random.standard_cauchy(size=len(value))

	elif noise_model == 'uniform':
		value_perturbed = value + noise_level*np.random.uniform(low=-1, high=1, size=len(value))

	else:
		raise ValueError(f'Noise model {noise_model} not recognized. Available models are "gaussian", "lorentzian", and "uniform".')

	return value_perturbed

#generate a sample of n alos sources randomly distributed on the sky within max_distance from Sun
def sample_sky_uniform(model, n, max_distance=3.0, sun_pos=(r_sun, 0., 0.)):
	"""
	Generate uniform random sample of line-of-sight accelerations from sky positions.
	
	:model: Gravitational model to compute accelerations from
	:n: Number of sample points to generate
	:max_distance: Maximum heliocentric distance in kpc 
	:sun_pos: Solar position in Galactocentric coordinates (kpc)
	:return: l (deg), b (deg), d (kpc), alos (kpc/s^2)
	"""
	# Generate uniform random positions on unit sphere
	# Use method from Muller et al. for uniform distribution
	u1 = np.random.random(n)
	u2 = np.random.random(n)
	
	# Convert to spherical coordinates
	l = u1 * 360.0  # Galactic longitude [0, 360) degrees
	b = np.arcsin(2*u2 - 1) * 180.0 / np.pi  # Galactic latitude [-90, 90] degrees
	
	# Generate uniform random distances
	d = np.random.random(n)**(1.0/3.0) * max_distance  # Uniform in volume
	
	# Compute line-of-sight accelerations
	alos = model.alos(l, b, d, frame='gal', sun_pos=sun_pos)
	
	return l, b, d, alos