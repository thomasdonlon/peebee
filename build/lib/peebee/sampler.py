"""
This submodule contains a few sampling routines for generating accelerations from populations of objects, which can be used for random bootstrapping or unit tests.
"""

import numpy as np
from .glob import r_sun

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

