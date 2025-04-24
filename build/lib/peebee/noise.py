"""
This submodule defines various noise models that can be used when optimizing acceleration models.
"""

import numpy as np

#-----------------------------------------------------------------
# Helper Functions
#-----------------------------------------------------------------

#takes in a string indicating the noise model to use, plus the residuals and params. 
#the last N items in params (where N is the number of parameters required by the noise model) are the noise parameters
def get_noise_model_likelihood(noise_model, resids, params):
	like = 0

	#could do some fancy lookup and lambda thing here but I'm not a nerd
	if noise_model == 'gaussian':
		like = gaussian_noise(resids, params[-1])
	elif noise_model == 'lorentz':
		like = lorentz_noise(resids, params[-1])
	elif noise_model == 'power_law':
		like = power_law_noise(resids, params[-1])
	else: #passed in a noise model that is not supported
		raise Exception(f'{noise_model} is not a supported noise model.')

	return like

#-----------------------------------------------------------------
# Noise models
#-----------------------------------------------------------------

def gaussian_noise(resids, sigma):
	return np.sum(0.5*np.log(2*np.pi*sigma**2) + 0.5*resids**2/sigma**2)

def lorentz_noise(resids, gamma): #cauchy-lorentz distribution
	return np.sum(np.log(np.pi*gamma) + np.log(1 + resids**2/gamma**2))

def power_law_noise(resids, zeta): #i.e. Moran et al. (2024)
	#the bounds of the power law normalization are just taken to be the smallest and largest residuals
	return np.sum(zeta*np.log(resids) + np.log(np.min(resids)**(1 - zeta) - np.max(resids)**(1-zeta)) - np.log(zeta - 1))