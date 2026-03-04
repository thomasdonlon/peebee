"""
This submodule contains various functions and routines that either make things easier elsewhere in the peebee code,
or are helpful functions to avoid doing simple tasks over and over. 
This includes acceleration calculations.
"""

import numpy as np
import functools
import matplotlib.pyplot as plt

#TODO: Use astropy units to make conversions happen automatically

#===============================================================================
# DECORATORS
#===============================================================================

#this generically allows functions to take in either arrays or single values
# and turns everything into (numpy) arrays behind the scenes
def fix_arrays(func):
    """:meta private:"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        #self.__doc__ = args[0].__doc__ #I removed this because it was breaking things, hopefully @wraps fixes the documentation?

        use_array = True

        #this *should* work but python3 is difficult with how it exactly passes (bound) methods
        #check if func is method and correct for that (can't turn __self__ into an array)
        # check_arg = 0
        # if inspect.ismethod(func): #first arg is self
        #     check_arg = 1

        #so let's do it the "naive" way
        check_arg = 0
        try:
            test = np.array(args[0], dtype=float)
        except TypeError: #args[0] is __self__ (probably)
            check_arg = 1

        if not(isinstance(args[check_arg], np.ndarray)): #check whether the first input is an array (assume all inputs are symmetric)
            use_array = False
            args = tuple([np.array([args[i+check_arg]], dtype=float) for i in range(len(args)-check_arg)]) #if they aren't, convert the args to arrays

        ret = func(*args, **kwargs) #catches the output from the function

        if not use_array: #convert them back if necessary
            if not(isinstance(ret, tuple)): #check whether we can actually iterate on the returned values (i.e. whether the func returns a single value or multiple)
                ret = ret[0]
            else:
                ret = tuple([ret[i][0] for i in range(len(ret))])

        return ret
    return wrapper


#===============================================================================
#FUNCTIONS
#===============================================================================

#--------------------------------------------------------------------------------
# General Utility Functions
#--------------------------------------------------------------------------------

#I can never be bothered to write out a pandas df or whatever so I wrote this instead
def write_to_csv(path, *args, titles=None):
	"""
	Write arrays to CSV file with optional column headers.
	
	:path (str): Output file path
	:*args (array_like): Variable number of arrays to write as columns
	:titles (list, optional): Column header names. Default is None.
	
	:returns: None
	"""

	if len(args) == 0: 
		raise Exception("Have to provide at least one array-like in addition to the file path")

	#start piping to file
	with open(path, 'w') as f: #automatically closes file when out of scope

		#write the header if titles were given
		if not (titles is None):
			title_str = ''
			for title in titles:
				title_str += title + ','
			title_str.rstrip(',') #adds 1 too many commas
			title_str += '\n'
			f.write(title_str)

		#write data
		for i in range(len(args[0])):
			out_str = ''
			for j in range(len(args)):
				out_str += str(args[j][i]) + ','
			out_str.rstrip(',') #adds 1 too many commas
			out_str += '\n'
			f.write(out_str)

#2 functions below added at the request of Lorenzo Addy, probably identical to his code

#helper function for getting array-wise magnitudes rather than writing things out
def mags(arr):
	"""
	Compute magnitudes of vectors in an array.
	
	:arr (array_like): Input array of vectors (..., N) where last axis contains vector components
	
	:returns: magnitudes (array_like) - Vector magnitudes
	"""
	return np.sum(arr**2, axis=-1)**0.5

#helper function for getting array-wise dot products
def dot(arr1, arr2):  #vecsi are Nx3 arrays
	"""
	Compute element-wise dot products between arrays of vectors.
	
	:arr1 (array_like): First array of vectors (..., N)
	:arr2 (array_like): Second array of vectors (..., N)
	
	:returns: dot_products (array_like) - Element-wise dot products
	"""
	return np.sum(arr1*arr2, axis=-1)

#--------------------------------------------------------------------------------
# Pulsar Related Functions
#--------------------------------------------------------------------------------

def _normalize_density(arr):
	#Helper function for normalizing a density array (e.g. a probability distribution) so that it has peak amplitude of 1. This is just for plotting/numerical purposes.
    arr -= np.min(arr)
    arr /= np.max(arr)
    return arr

def _lk_bias_lorimer_density_model(R, z):
	#Helper function for computing the Lorimer et al. (2006) model density at a given R and z, used for LK bias correction.

	#set up Lorimer et al. (2006) model #TODO: more realistic model?
	A = 41 #kpc^-2
	B = 1.9 
	C = 5
	E = 0.5 #kpc #NOTE: This is only the case for MSPs, for canonical pulsars this is more like 300 pc, but we don't really use peebee for CPs
	R_sun = 8.5 #kpc
	
	return A*(R/R_sun)**B*np.exp(-C*(R-R_sun)/R_sun)*np.exp(-1*np.abs(z)/E)

def remove_lk_bias(psr_l, psr_b, flux, px, px_err, n_bins=500, plot=False):
	"""
	Compute the Lutz-Kelker bias correction for a pulsar, according to the method from Verbiest, Lorimer, & McLaughlin (2010).

		:psr_l (float): Galactic longitude (degrees)
		:psr_b (float): Galactic latitude (degrees)
		:flux (float): Integrated flux density (mJy, typically at 1.4 GHz)
		:px (float): Measured parallax (mas)
		:px_err (float): Measured parallax error (mas)
		:n_bins (int, optional): Number of bins in the parallax grid. Default is 500. Higher values give more precise results but take longer to compute.
		:plot (bool, optional): If True, generates plots of the priors and parallax distributions for visualization. Default is False.

	:returns: lk_correction (tuple) - Corrected parallax (mas), lower error (mas), upper error (mas)
	"""
      
	if px <= 0 or px_err <= 0:
		raise ValueError("Parallax and parallax error must be positive values.")
      
	dpx = 2*px / n_bins #resolution in parallax for grid spacing
	px_arr = np.arange(dpx, 2*px, dpx) #grid to evaluate over
      
	#pulsar measurement
	prob_measured = 1/(2*np.pi*px_err**2)**0.5 * np.exp(-0.5*(px-px_arr)**2/px_err**2)
	prob_measured = _normalize_density(prob_measured)
      
	#pulsar volumetric prior	
	dist_arr= 1/px_arr

	R_sun = 8.5 #kpc - Solar position -- TODO: should grab this from the global constants, but would have to also adjust in the Lorimer 2006 model
	x = R_sun - dist_arr*np.cos(np.pi/180*psr_l)*np.cos(np.pi/180*psr_b)
	y = dist_arr*np.sin(np.pi/180*psr_l)*np.cos(np.pi/180*psr_b)
	z = dist_arr*np.sin(np.pi/180*psr_b)
	R = (x**2 + y**2)**0.5
	vol_prior = _lk_bias_lorimer_density_model(R, z)*dist_arr**3
	vol_prior /= px_arr**2 #|dD/dpx|
	vol_prior = _normalize_density(vol_prior)

	#pulsar luminosity prior
	mean_log_int_lum = np.log10(0.07)
	log_scale = 0.9
	lum_arr = flux*px_arr**-2 # S px^-2 = L
	lum_prior = 1/((2*np.pi)**0.5*log_scale*lum_arr) * np.exp(-0.5*(np.log10(lum_arr)-mean_log_int_lum)**2/log_scale**2) #log-normal distribution
	lum_prior /= px_arr**3 #|dL/dpx|
	lum_prior = _normalize_density(lum_prior)

	#multiply priors together and normalize
	total_prior = vol_prior * lum_prior
	total_prior = _normalize_density(total_prior)

	#compute quantiles of corrected distribution
	corrected_px_distribution = _normalize_density(prob_measured*total_prior)
	cdf = np.cumsum(corrected_px_distribution)
	cdf /= np.max(cdf)

	med = px_arr[np.argmin(np.abs(cdf - 0.5))]
	low = px_arr[np.argmin(np.abs(cdf - (0.5 - 0.341)))]
	high = px_arr[np.argmin(np.abs(cdf - (0.5 + 0.341)))]

	if plot:
		#plot 1
		plt.figure(figsize=(8,6))
		plt.plot(px_arr, vol_prior, label='Volumetric Prior')
		plt.plot(px_arr, lum_prior, label='Luminosity Prior')
		plt.plot(px_arr, total_prior, label='Total Prior')

		plt.title('Priors for LK bias correction')
		plt.legend()
		plt.xlim(0, 2*px)
		plt.ylim(0, 1.05)
		plt.xlabel('Parallax (mas)')
		plt.ylabel('Normalized Density')
		plt.show()

		#plot 2
		plt.figure(figsize=(8,6))
		plt.plot(px_arr, prob_measured, label='Original Measurement')
		plt.plot(px_arr, total_prior, label='Total Prior')
		plt.plot(px_arr, corrected_px_distribution, label='Corrected PX Distribution')
		plt.plot([med, med], [0, corrected_px_distribution[np.argmin(np.abs(px_arr - med))]], 'k--')
		plt.plot([low, low], [0, corrected_px_distribution[np.argmin(np.abs(px_arr - low))]], 'k--')
		plt.plot([high, high], [0, corrected_px_distribution[np.argmin(np.abs(px_arr - high))]], 'k--')

		plt.xlabel('Parallax (mas)')
		plt.ylabel('Normalized Density')
		plt.title('LK bias correction')
		plt.legend()
		plt.xlim(0, 2*px)
		plt.ylim(0, 1.05)
		plt.show()

	return round(med, 8), round(med-low, 8), round(high-med, 8) #rounding eliminates some numerical noise in bin edges