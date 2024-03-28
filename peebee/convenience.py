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

	#TODO: c should be grabbed from somewhere rather than being 3e8 (and G)
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
def alos_obs(*args, frame='gal'):
	"""
	Compute $a_\\mathrm{los}$, the line-of-sight acceleration of a pulsar given its observed properties. 
	Automatically determines whether to use GR if the number of inputs is 6 (no GR) or 9 (with GR). 
	$a_\\mathrm{los}$ is computed as $$\\frac{a_\\mathrm{los} P_b}{c} = \\dot{P_b}^\\mathrm{Obs} - \\dot{P_b}^\\mathrm{Shk} - \\dot{P_b}^\\mathrm{GR} $$
	where $\\dot{P_b}^\\mathrm{GR} = 0$ if mp, mc, and e are not provided. 

	:coord1-3: Galactocentric Cartesian coordinates (kpc) or Galactic longitude, latitude (deg) and heliocentric distance (kpc). 
	Toggle between these options with the 'frame' flag.
	:pb: binary orbital period of the pulsar (s)
	:pbdot_obs: the observed time derivative of the binary orbital period (s/s)
	:mu: the observed proper motion (mas/yr)
	:mp: (optional) the mass of the pulsar (M$_\\odot$)
	:mc: (optional) the mass of the companion (M$_\\odot$)
	:e: (optional) orbital eccentricity of the binary
	:frame: [default value = 'gal'] Toggle the input frame. Options are 'cart' for Galactocentric Cartesian (X,Y,Z), 'gal' for heliocentric Galactic coordinates (l,b,d),
	'icrs' for equatorial coordinates (ra, dec, d), and 'ecl' for ecliptic coordinates (lam, bet, d) 
	"""

	if len(args) == 6:
		mode = 'non_gr'
	elif len(args) == 9:
		mode = 'gr'
	else:
		raise Exception('alos_obs() only works with 6 arguments (non-GR mode) or 9 arguments (GR mode).')

	l = args[0]
	b = args[1]
	d = args[2]

	pb = args[3]
	pbdot_obs = args[4]
	mu = args[5]

	pbdot_shk = pdot_shk(pb, mu, d)

	pbdot_gr = 0 #if non_gr mode, just ignore gr term
	if mode == 'gr':
		mp = args[6]
		mc = args[7]
		e = args[8]
		pbdot_gr = pbdot_gr(pb, mp, mc, e)

	pbdot_act = pbdot_obs - pbdot_shk - pbdot_gr

	#should get c from somewhere, also unitful rather than hard-coded conversion here
	return pbdot_act*3e8/pb*3.154e10 #to mm/s/yr

@fix_arrays
@convert_to_frame('gal')
def pbdot_intr(l, b, d, pb, pbdot_obs, mu, frame='gal'):
	"""
	Compute $\\dot{P}_b^\\mathrm{Intr}$, the binary orbital period derivative of the pulsar not due to the Shklovskii Effect.
	This can be interpreted as the observed decay of the binary orbital period due to emission of gravitational waves. 

	:coord1-3: Galactocentric Cartesian coordinates (kpc) or Galactic longitude, latitude (deg) and heliocentric distance (kpc). 
	Toggle between these options with the 'frame' flag.
	:pb: binary orbital period of the pulsar (s)
	:pbdot_obs: the observed time derivative of the binary orbital period (s/s)
	:mu: the observed proper motion (mas/yr)
	:frame: [default value = 'gal'] Toggle the input frame. Options are 'cart' for Galactocentric Cartesian (X,Y,Z), 'gal' for heliocentric Galactic coordinates (l,b,d),
	'icrs' for equatorial coordinates (ra, dec, d), and 'ecl' for ecliptic coordinates (lam, bet, d) 
	"""

	pbdot_shk = pdot_shk(pb, mu, d)

	return pbdot_obs - pbdot_shk

@fix_arrays
@convert_to_frame('gal')
def intr_over_gr(l, b, d, pb, pbdot_obs, mu, mp, mc, e, frame='gal'):
	"""
	Compute $\\dot{P}_b^\\mathrm{Intr}/\\dot{P}_b^\\mathrm{GR}$, the ratio of the observed orbital decay of the binary to the theoretical orbtial decay due to
	the emission of gravitational waves. If the observation is consistent with general relativity, this should be equal to 1. 

	:coord1-3: Galactocentric Cartesian coordinates (kpc) or Galactic longitude, latitude (deg) and heliocentric distance (kpc). 
	Toggle between these options with the 'frame' flag.
	:pb: binary orbital period of the pulsar (s)
	:pbdot_obs: the observed time derivative of the binary orbital period (s/s)
	:mu: the observed proper motion (mas/yr)
	:mp: (optional) the mass of the pulsar (M$_\\odot$)
	:mc: (optional) the mass of the companion (M$_\\odot$)
	:e: (optional) orbital eccentricity of the binary
	:frame: [default value = 'gal'] Toggle the input frame. Options are 'cart' for Galactocentric Cartesian (X,Y,Z), 'gal' for heliocentric Galactic coordinates (l,b,d),
	'icrs' for equatorial coordinates (ra, dec, d), and 'ecl' for ecliptic coordinates (lam, bet, d) 
	"""

	pbdot_shk = pdot_shk(pb, mu, d)
	pbdot_gr = pbdot_gr(pb, mp, mc, e)
	pbdot_intr = pbdot_intr(l, b, d, pb, pbdot_obs, mu, frame='gal')

	return pbdot_intr/pbdot_gr

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

#I can never be bothered to write out a pandas df or whatever so I wrote this instead
def write_to_csv(path, *args, titles=None):

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

