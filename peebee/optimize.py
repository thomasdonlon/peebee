"""
This submodule is built around an optimizer that takes in accelerations and their 3-dimensional positions, and can optimize a variety of potential models to that information. 
It can also be used to evaluate how good of a fit a given potential is when applied to a set of data. 
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from uncertainties import ufloat
import astropy.units as u 

from .transforms import convert_to_frame
from .models import Model, CompositeModel
from .glob import kpctocm, r_sun
from .noise import get_noise_model_likelihood

#-----------------------------------------------------------------
# CLASSES
#-----------------------------------------------------------------

#helper class to interface between scipy optimizers and peebee models
#specifically, this allows us to pass only the free parameters to the optimizer
# while still being able to update the full model with fixed parameters as needed
class Fitter:
	def __init__(self, model=None, noise_model='none'):
		self.params = {}
		self.set_model(model)
		self.noise_model = noise_model

		#need to keep track of which parameters are being optimized
		self.param_names_to_optimize = []
		
	def set_model(self, model):
		self.model = model
		for n in model.param_names:
			self.params[n] = model.params[n]

	#may change how noise models work later as a class, but for now this is fine
	def set_noise_model(self, noise_model):
		self.noise_model = noise_model

	def set_params_to_optimize(self, param_names):
		self.param_names_to_optimize = param_names

	def update_params(self, params):
		#have to get crafty about how we update the parameters, remembering that None params are disabled
		#and that composite models and models have different setups for their params

		if isinstance(self.model, Model): #single model version 

			new_params = dict()
			i = 0
			for n in self.model.param_names:
				if n in self.model._disabled_param_names:
					new_params[n] = self.model.get_param_default(n)
				else:
					new_params[n] = params[i]
					i += 1

		elif isinstance(self.model, CompositeModel):

			new_params = []
			i = 0
			for m in self.model.models:
				single_model_params = dict()
				for n in m.param_names:
					if n in m._disabled_param_names:
						single_model_params[n] = m.get_param_default(n)
					else:
						single_model_params[n] = params[i]
						i += 1
				new_params.append(single_model_params)

		self.model.set_params(new_params)



#-----------------------------------------------------------------
# Likelihood and Uncertainty Functions
#-----------------------------------------------------------------

def rss(params, model, data, scale, sun_pos, negative_mass, noise_model, print_out=False): #has to take sun_pos, etc as arg (not kwarg) for scipy func's

	#scale is only a positional argument for the sake of how scipy passes things through optimization functions
	#it is always set to 1. unless the user specifies a different value

	#honestly, scale should probably be removed because it isn't really something that should be played around with

	#have to get crafty about how we update the parameters, remembering that None params are disabled
	#and that composite models and models have different setups for their params

	if isinstance(model, Model): #single model version 

		new_params = dict()
		i = 0
		for n in model.param_names:
			if n in model._disabled_param_names:
				new_params[n] = model.get_param_default(n)
			else:
				new_params[n] = params[i]
				i += 1

	elif isinstance(model, CompositeModel):

		new_params = []
		i = 0
		for m in model.models:
			single_model_params = dict()
			for n in m.param_names:
				if n in m._disabled_param_names:
					single_model_params[n] = m.get_param_default(n)
				else:
					single_model_params[n] = params[i]
					i += 1
			new_params.append(single_model_params)

	model.set_params(new_params)
	model_alos = model.alos(data['l'], data['b'], data['d'], sun_pos=sun_pos)

	if negative_mass: #this is to allow for negative density in terms of fitting perturbations to existing acceleration profiles 
		model_alos *= -1

	rss = 0.5*np.sum((model_alos*kpctocm - data['alos'])**2/(data['alos_err']**2))*scale

	#add in a noise model, if one was specified
	#keep in mind that the noise parameters must be the last N parameters of the params list
	#(this is flimsy and should be changed in a future update, but probably requires substantial rework of peebee)
	if noise_model != 'none':
		resids = np.abs(model_alos*kpctocm - data['alos'])
		noise_like = get_noise_model_likelihood(noise_model, resids, params)
		rss += noise_like

	if print_out:
		print('model:', model_alos*kpctocm)
		print('observed:', data['alos'])
		print('residual:', (model_alos*kpctocm - data['alos']))
		print('chi2:', (model_alos*kpctocm - data['alos'])**2/(data['alos_err']**2))
		if scale != 1.:
			print('scaled chi2:', (model_alos*kpctocm - data['alos'])**2/(data['alos_err']**2)*scale)
		print()
	return rss

def rss_no_update(l, b, d, alos, alos_err, model, scale=1., noise_model='none', noise_model_params=[], print_out=True, sun_pos=(r_sun, 0., 0.)): #for evaluating existing models
	model_alos = model.alos(l, b, d, sun_pos=sun_pos)

	if print_out:
		print('model:', model_alos*kpctocm)
		print('observed:', alos)
		print('residual:', (model_alos*kpctocm - alos))
		print()

	rss = 0.5*np.sum((model_alos*kpctocm - alos)**2/(alos_err**2))*scale

	if noise_model != 'none':
		resids = np.abs(model_alos*kpctocm - alos)
		noise_like = get_noise_model_likelihood(noise_model, resids, noise_model_params)
		rss += noise_like

	return rss

def approx_hess_inv(params, model, data, dparams, noise_model='none', sun_pos=(r_sun, 0., 0.), negative_mass=False):
	n_params = len(params)
	hess = np.zeros((n_params, n_params))

	for j in range(n_params):
		for k in range(n_params):
			if k >= j: # top half of the matrix

				#this feels like a silly way to write all this out but it works
				p = params.copy()
				p[j] += dparams[j]
				p[k] += dparams[k]
				llpp = rss(p, model, data, 1., sun_pos, negative_mass, noise_model) #-lgL = chi^2/2

				p = params.copy()
				p[j] += dparams[j]
				p[k] -= dparams[k]
				llpm = rss(p, model, data, 1., sun_pos, negative_mass, noise_model)

				p = params.copy()
				p[j] -= dparams[j]
				p[k] += dparams[k]
				llmp = rss(p, model, data, 1., sun_pos, negative_mass, noise_model)

				p = params.copy()
				p[j] -= dparams[j]
				p[k] -= dparams[k]
				llmm = rss(p, model, data, 1., sun_pos, negative_mass, noise_model)

				hess[j, k] = (llpp - llpm - llmp + llmm)/(4*dparams[j]*dparams[k])
				hess[k,j] = hess[j,k] #make it symmetric

	try:
		hess_inv = np.linalg.inv(hess)
	except np.linalg.LinAlgError as err:
		print('NumPy threw an error, usually this is because the hessian was singular.\nProviding params and hessian:')
		print(f'params: {params}')
		print(f'hess: {hess}')
		raise err

	return hess, hess_inv

#-----------------------------------------------------------------
# Functions
#-----------------------------------------------------------------

#don't optimize anything, just spit out values, chi^2, AIC
@convert_to_frame('gal')
def evaluate_model(l, b, d, alos, alos_err, model, frame='gal', scale=1., noise_model='none', noise_model_params=[], print_out=True, sun_pos=(r_sun, 0, 0)):
	rss_eval = rss_no_update(l, b, d, alos, alos_err, model, scale=scale, noise_model=noise_model, noise_model_params=noise_model_params, print_out=print_out, sun_pos=sun_pos)
	chi2 = rss_eval/(len(alos) - model.nparams)
	aic = 2*len(alos) + rss_eval

	if print_out:
		print(f'RSS: {rss_eval}')
		print(f'Reduced chi^2: {chi2}')
		print(f'AIC: {aic}')

	return chi2, aic

#TODO: implement shgo? other algos? 
#TODO: implement MCMC version?
@convert_to_frame('gal')
def fit_model(l, b, d, alos, alos_err, model, bounds, frame='gal', mode='gd', scale=1., h=0.0001, sun_pos=(r_sun, 0, 0), negative_mass=False, noise_model='none', **kwargs):
	#mode needs to be either gd for gradient descent or de for differential evolution

	#this just simplifies passing things through the different functions (without making the end user do it themselves)
	data = {'l':l,
	        'b':b,
	        'd':d,
	        'alos':alos,
	        'alos_err':alos_err}

	#collect print_out kwarg, set False as default
	print_out = False
	if 'print_out' in kwargs.keys():
		print_out = kwargs['print_out']

		#and scrape it from the kwargs dict
		kwargs.pop('print_out')

	#do the optimization
	if mode == 'gd': #use gradient descent optimizer (fast but can get stuck in local minima if you don't know what the likelihood surface looks like)
		if not ('x0' in kwargs.keys()):
			raise Exception("x0 must be specified for fit_model option 'gd'")
		result = minimize(rss, bounds=bounds, args=(model, data, scale, sun_pos, negative_mass, noise_model), **kwargs)

	elif mode == 'de': #use differential evolution as optimizer (slower but less likely to get stuck in local minima)

		#gather relevant keywords for which we want to change the default values
		if isinstance(model, Model):
			popsize = 20*model.nparams
		elif isinstance(model, CompositeModel):
			popsize = 20*np.sum(model.nparams)

		if 'popsize' in kwargs.keys():
			popsize = kwargs['popsize']

			#and scrape it from the kwargs dict
			kwargs.pop('popsize')

		result = differential_evolution(rss, bounds, args=(model, data, scale, sun_pos, negative_mass, noise_model), popsize=popsize, **kwargs)

	else:
		raise Exception("Unsupported option for fit_model kwarg 'mode': currently supported values are ['gd', 'de']")

	#extra calcs and printing stuff afterwards

	#also have to collect scale here to pass to the rss function below
	scale = 1.
	if 'scale' in kwargs.keys():
		scale = kwargs['scale']

	#check that there are enough datapoints to calculate a reduced chi^2
	params = result.x
	if (len(alos) - len(params) - 1) <= 0:
		print("Warning: Not enough datapoints to constrain result!")
		red_chi2 = -999.
		errors = np.zeros(len(params)) - 999.
		aic = -999.
	else:
		red_chi2 = result.fun/(len(alos) - len(params) - 1)
		aic = 2*len(params) + result.fun
		if print_out:
			rss(params, model, data, scale, sun_pos, negative_mass, noise_model, print_out=print_out)
			print(f'Final reduced chi^2: {red_chi2}')
			print(f'Final AIC: {aic}')
			print(f'Params: {model.params}')

		#calculate errors
		try:
			hess, hess_inv = approx_hess_inv(params, model, data, h*np.abs(np.array(params)), noise_model=noise_model, sun_pos=sun_pos)
			if print_out:
				print(f'Hessian: {hess}')
				print(f"Std. Errors: {np.diag(hess_inv/len(alos))**0.5}")

			errors = np.diag(hess_inv/len(alos))**0.5
		except np.linalg.LinAlgError:
			print('The Hessian matrix was (probably) singular. This can happen if the optimizer failed or if you have a small number of datapoints.')
			errors = np.zeros(len(params)) - 999.

	return params, errors, red_chi2, aic, result
