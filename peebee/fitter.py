"""
This submodule is built around an optimizer that takes in accelerations and their 3-dimensional positions, and can optimize a variety of potential models to that information. 
IT can also be used to evaluate how good of a fit a given potential is when applied to a set of data. 
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from uncertainties import ufloat
import astropy.units as u 

from .transforms import convert_to_frame
from .models import Model, CompositeModel
from .glob import kpctocm, r_sun

#-----------------------------------------------------------------
# Helper Functions
#-----------------------------------------------------------------

def rss(params, model, data, scale, sun_pos, print_out=False): #has to take sun_pos as arg (not kwarg) for scipy func's

	#scale is only a positional argument for the sake of how scipy passes things through optimization functions
	#it is always set to 1. unless the user specifies a different value

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
	rss = np.sum((model_alos*kpctocm - data['alos'])**2/(data['alos_err']**2))*scale
	if print_out:
		print('model:', model_alos*kpctocm)
		print('observed:', data['alos'])
		print('residual:', (model_alos*kpctocm - data['alos']))
		print('chi2:', (model_alos*kpctocm - data['alos'])**2/(data['alos_err']**2))
		if scale != 1.:
			print('scaled chi2:', (model_alos*kpctocm - data['alos'])**2/(data['alos_err']**2)*scale)
		print()
	return rss

def rss_no_update(l, b, d, alos, alos_err, model, scale=1., print_out=True, sun_pos=(r_sun, 0., 0.)): #for evaluating existing models
	model_alos = model.alos(l, b, d, sun_pos=sun_pos)

	if print_out:
		print('model:', model_alos*kpctocm)
		print('observed:', alos)
		print('residual:', (model_alos*kpctocm - alos))
		print()

	rss = np.sum((model_alos*kpctocm - alos)**2/(alos_err**2))*scale
	return rss

def approx_hess_inv(params, model, data, dparams, sun_pos=(r_sun, 0., 0.)):
	n_params = len(params)
	hess = np.zeros((n_params, n_params))

	for j in range(n_params):
		for k in range(n_params):
			if k >= j: # top half of the matrix

				#this feels like a silly way to write all this out but it works
				p = params.copy()
				p[j] += dparams[j]
				p[k] += dparams[k]
				llpp = 0.5*rss(p, model, data, 1., sun_pos) #-lgL = chi^2/2

				p = params.copy()
				p[j] += dparams[j]
				p[k] -= dparams[k]
				llpm = 0.5*rss(p, model, data, 1., sun_pos)

				p = params.copy()
				p[j] -= dparams[j]
				p[k] += dparams[k]
				llmp = 0.5*rss(p, model, data, 1., sun_pos)

				p = params.copy()
				p[j] -= dparams[j]
				p[k] -= dparams[k]
				llmm = 0.5*rss(p, model, data, 1., sun_pos)

				hess[j, k] = (llpp - llpm - llmp + llmm)/(4*dparams[j]*dparams[k])
				hess[k,j] = hess[j,k] #make it symmetric

	try:
		hess_inv = np.linalg.inv(hess)
	except np.linalg.LinAlgError as err:
		print('(probably) a singular matrix error: providing params and hess:')
		print(f'params: {params}')
		print(f'hess: {hess}')
		raise err

	return hess, hess_inv

#-----------------------------------------------------------------
# Functions
#-----------------------------------------------------------------

#don't optimize anything, just spit out values, chi^2, AIC
@convert_to_frame('gal')
def evaluate_model(l, b, d, alos, alos_err, model, frame='gal', scale=1., print_out=True, sun_pos=(r_sun, 0, 0)):
	rss_eval = rss_no_update(l, b, d, alos, alos_err, model, scale=scale, print_out=print_out, sun_pos=sun_pos)
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
def fit_model(l, b, d, alos, alos_err, model, bounds, frame='gal', mode='gd', scale=1., h=0.0001, sun_pos=(r_sun, 0, 0), **kwargs):
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
		result = minimize(rss, bounds=bounds, args=(model, data, scale, sun_pos,), **kwargs)

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

		result = differential_evolution(rss, bounds, args=(model, data, scale, sun_pos,), popsize=popsize, **kwargs)

	else:
		raise Exception("Unsupported option for fit_model kwarg 'mode': currently supported values are ['gd', 'de']")

	#extra calcs and printing stuff afterwards

	#also have to collect scale here to pass to the rss function below
	scale = 1.
	if 'scale' in kwargs.keys():
		scale = kwargs['scale']

	params = result.x
	red_chi2 = result.fun/(len(alos) - len(params) - 1)
	aic = 2*len(params) + result.fun
	if print_out:
		rss(params, model, data, scale, print_out=print_out)
		print(f'Final reduced chi^2: {red_chi2}')
		print(f'Final AIC: {aic}')
		print(f'Params: {model.params}')

	#calculate errors
	hess, hess_inv = approx_hess_inv(params, model, data, [h]*len(params), sun_pos=sun_pos)
	if print_out:
		print(f'Hessian: {hess}')
		print(f"Std. Errors: {np.diag(hess_inv/len(alos))**0.5}")

	errors = np.diag(hess_inv/len(alos))**0.5

	return params, errors, red_chi2, aic, result
