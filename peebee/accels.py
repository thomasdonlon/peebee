"""
This submodule contains acceleration calculations for both orbital and spin pulsar data.
"""

import numpy as np
from .glob import fix_arrays
from .transforms import convert_to_frame

#TODO: Use astropy units to make conversions happen automatically

#===============================================================================
# Functions
#===============================================================================

@fix_arrays
def pbdot_gr(pb, mp, mc, e):
	"""
	Compute $\\dot{P}^\\mathrm{GR}_b$, the change in orbital period due to emission of gravitational waves.
	Adapted from Weisberg & Huang (2016). 

	:pb: Orbital period (days)
	:mp: Mass of the pulsar (M$_\\odot$)
	:mc: Mass of the companion (M$_\\odot$)
	:e: Orbital eccentricity
	
	Returns: pbdot_gr (s/s)
	"""

	mc *= 1.989e30 #to kg
	mp *= 1.989e30

	pb *= (24*3600) #to s

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
	
	Returns: pdot_shk (s/s)
	"""

	mu *= 1.537e-16 #to rad/s
	d *= 3.086e19 #to m

	#TODO: c should be grabbed from somewhere rather than being 3e8
	pdot_shk = p*mu**2*d/3e8

	return pdot_shk

@fix_arrays
def psdot_b(ps, psdot_obs, mu, d):
	"""
	Compute $\\dot{P}^\\mathrm{B}_s$, the change in spin period due to intrinsic magnetic spindown of the pulsar.
	We use the empirically calibrated estimate from Donlon et al. (2025). 

	:ps: Spin period (s)
	:psdot_obs: the observed time derivative of the spin period (s/s)
	:mu: Total proper motion (mas/yr)
	:d: Distance from Sun (kpc)
	
	Returns: psdot_b (s/s)
	"""

	psdot_shk = pdot_shk(ps, mu, d)
	psdot_obs_i = psdot_obs - psdot_shk
	
    #TODO: if we're adding uncertainties at some point, we'll have to add in the intrinsic bsurf scatter term
	Bsurf_i = 3.2e19 * np.sqrt(ps*psdot_obs_i) #to Gauss, using the standard magnetic dipole formula for surface magnetic field strength
	psdot_b = 5.9e-29 * Bsurf_i - 2.8e-21 #Donlon et al. (2025) eq. 12

	return psdot_b

@fix_arrays
def alos_orb_gal(*args):
	"""
	Compute $a_\\mathrm{los}$, the line-of-sight acceleration of a pulsar given its observed properties. 
	Automatically determines whether to use GR if the number of inputs is 6 (no GR) or 9 (with GR). 
	$a_\\mathrm{los}$ is computed as 
	$$ 
	\\frac{a _\\mathrm{los} P_b}{c} = \\dot{P_b}^\\mathrm{Obs} - \\dot{P_b}^\\mathrm{Shk} - \\dot{P_b}^\\mathrm{GR} 
	$$
	where $\\dot{P_b}^\\mathrm{GR} = 0$ is assumed if mp, mc, and e are not provided. 

	:d: Heliocentric distance (kpc).
	:pb: binary orbital period of the pulsar (days)
	:pbdot_obs: the observed time derivative of the binary orbital period (s/s)
	:mu: the observed proper motion (mas/yr)
	:mp: (optional) the mass of the pulsar (M$_\\odot$)
	:mc: (optional) the mass of the companion (M$_\\odot$)
	:e: (optional) orbital eccentricity of the binary

	Returns: alos (mm/s/yr)
	"""

	if len(args) == 4:
		mode = 'non_gr'
	elif len(args) == 7:
		mode = 'gr'
	else:
		raise Exception('alos_obs() only works with 4 arguments (non-GR mode) or 7 arguments (GR mode).')

	d = args[0]

	pb = args[1] * (24*3600) #to s
	pbdot_obs = args[2]
	mu = args[3]

	pbdot_shk_i = pdot_shk(pb, mu, d)

	pbdot_gr_i = 0 #if non_gr mode, just ignore gr term
	if mode == 'gr':
		mp = args[4]
		mc = args[5]
		e = args[6]
		pbdot_gr_i = pbdot_gr(pb, mp, mc, e)

	pbdot_act = pbdot_obs - pbdot_shk_i - pbdot_gr_i

	#TODO: should get c from somewhere, also unitful rather than hard-coded conversion here
	return pbdot_act*3e8/pb*3.154e10 #to mm/s/yr

@fix_arrays
def alos_spin_gal(d, ps, psdot_obs, mu):
	"""
	Compute $a_\\mathrm{los}$, the line-of-sight acceleration of a pulsar given its observed properties. 
	$a_\\mathrm{los}$ is computed as 
	$$ 
	\\frac{a _\\mathrm{los} P_s}{c} = \\dot{P_s}^\\mathrm{Obs} - \\dot{P_s}^\\mathrm{Shk} - \\dot{P_s}^\\mathrm{B} 
	$$
	where $\\dot{P_s}^\\mathrm{B}$ is the intrinsic spin-down rate of the pulsar (see `psdot_b` function, Donlon et al. 2025). 

	:d: Heliocentric distance (kpc).
	:ps: spin period of the pulsar (s)
	:psdot_obs: the observed time derivative of the spin period (s/s)
	:mu: the observed proper motion (mas/yr)
	
	Returns: alos (mm/s/yr)
	"""

	psdot_shk = pdot_shk(ps, mu, d)
	psdot_b = psdot_b(ps, psdot_obs, mu, d)

	psdot_act = psdot_obs - psdot_shk - psdot_b

	#TODO: should get c from somewhere, also unitful rather than hard-coded conversion here
	return psdot_act*3e8/ps*3.154e10 #to mm/s/yr

@fix_arrays
@convert_to_frame('gal')
def dm_over_bary_alos(l, b, d, model_bary, model_dm, frame='gal'):
	"""
	Compute ${ | a _\\mathrm{DM} | / | a _\\mathrm{bary} | }$, the ratio of the (magnitudes of the) relative contributions of the dark matter and baryonic components
	of the Galaxy to the acceleration at a given point. Useful for estimating the ability to constrain dark matter information from measurements.

	:coord1-3: Galactocentric Cartesian coordinates (kpc) or Galactic longitude, latitude (deg) and heliocentric distance (kpc). Toggle between these options with the 'frame' flag.
	:model_bary: peebee.model for the baryonic component of the Galactic potential (i.e. disk + bulge)
	:model_dm: peebee.model for the dark component of the Galactic potential (i.e. halo)
	:frame: [default value = 'gal'] Toggle the input frame. Options are 'cart' for Galactocentric Cartesian (X,Y,Z), 'gal' for heliocentric Galactic coordinates (l,b,d), 'icrs' for equatorial coordinates (ra, dec, d), and 'ecl' for ecliptic coordinates (lam, bet, d) 
	
	Returns: dm_over_bary_alos (dimensionless)
	"""

	alos_bary = model_bary.alos(l, b, d, frame=frame)
	alos_dm = model_dm.alos(l, b, d, frame=frame)

	return np.abs(alos_dm)/np.abs(alos_bary)