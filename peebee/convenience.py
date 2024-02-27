"""
Text here for Sphinx (I think)
"""

import numpy as np
from .glob import fix_arrays, r_sun
from .transforms import convert_to_frame

#TODO: Use astropy units to make conversions happen automatically

#===============================================================================
# DECORATORS
#===============================================================================

#this generically allows functions to take in either arrays or single values
# and turns everything into (numpy) arrays behind the scenes
def fix_arrays(func):
    def wrapper(*args, **kwargs):

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

@fix_arrays
def pbdot_gr(pb, mp, mc, e):
	"""
	Compute $\\dot{P}^\\mathrm{GR}_b$, the change in orbital period due to emission of gravitational waves.
	Adapted from Weisberg & Huang (2016). 

	:pb: Orbital period (s)
	:mp: Mass of the pulsar (M$_\\odot$)
	:mc: Mass of the companion (M$_\\odot$)
	:e: Orbital eccentricity
	"""

	mc *= 1.989e30 #to kg
	mp *= 1.989e30

	#TODO: c should be grabbed from somewhere rather than being 3e8
	pbdot_gr = -192*np.pi*(6.67e-11)**(5/3)/(5*(3e8)**5) * (pb/(2*np.pi))**(-5/3) * (1-e**2)**(-7/2) * (1 + (73/24)*e**2 + (37/96)*e**4) * mp*mc/((mp + mc)**(1/3))

	return pbdot_gr

@fix_arrays
def pdot_shk(p, mu, d):
	"""
	Compute $\\dot{P}^\\mathrm{Shk}$, the change in observed period due to the Shklovskii Effect (See Shklovskii, 1970).
	

	:p: (Orbital or spin) period (s)
	:mu: Total proper motion (mas/yr)
	:d: Distance from Sun (kpc)
	"""

	mu *= 1.537e-16 #to rad/s
	d = 3.086e19 #to m

	#TODO: c should be grabbed from somewhere rather than being 3e8
	pdot_shk = p*mu**2*d/3e8

	return pdot_shk

@fix_arrays
@convert_to_frame('gal')
def dm_over_bary_alos(l, b, d, model_bary, model_dm, frame='gal'):
	"""
	Compute $|a_\\mathrm{DM}|/|a_\\mathrm{bary}|$, the ratio of the (magnitudes of the) relative contributions of the dark matter and baryonic components
	of the Galaxy at a given point. Useful for estimating ability to constrain dark matter information from measurements.

	:coord1-3: Galactocentric Cartesian coordinates (kpc) or Galactic longitude, latitude (deg) and heliocentric distance (kpc). 
	Toggle between these options with the 'frame' flag.
	:model_bary: peebee.model for the baryonic component of the Galactic potential (i.e. disk + bulge)
	:model_dm: peebee.model for the dark component of the Galactic potential (i.e. halo)
	:frame: [default value = 'gal'] Toggle the input frame. Options are 'cart' for Galactocentric Cartesian (X,Y,Z), 'gal' for heliocentric Galactic coordinates (l,b,d),
	'icrs' for equatorial coordinates (ra, dec, d), and 'ecl' for ecliptic coordinates (lam, bet, d) 
	"""

	alos_bary = model_bary.alos(l, b, d)
	alos_dm = model_dm.alos(l, b, d)

	return np.abs(alos_dm)/np.abs(alos_bary)
