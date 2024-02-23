"""
Text here for Sphinx (I think)
"""

import numpy as np
import transforms
from .decorators import fix_arrays, convert_to_frame

r_sun = 8.0 #TODO: This should be set in a single settings/global file and imported, should be changeable

#TODO: Use astropy units to make conversions happen automatically

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
	pbdot_gr = -192*np.pi*(6.67e-11)**(5/3)/(5*(3e8)**5) * (p/(2*np.pi))**(-5/3) * (1-e**2)**(-7/2) * (1 + (73/24)*e**2 + (37/96)*e**4) * mp*mc/((mp + mc)**(1/3))

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
