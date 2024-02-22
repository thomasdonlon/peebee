"""
Text here for Sphinx (I think)
"""

import numpy as np

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
