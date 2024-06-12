"""
Peebee contains many different potential models, which can be used to either generate acceleration data, or can be fit to existing acceleration data. 
A variety of models are incorporated here, and are added as they become relevant. If you do not see a specific potential you are interested in, you can 
reach out and ask that it be added to a future version of peebee.
"""

#TODO: dict names can collide when adding two models if param names are identical
#TODO: implement astropy units in a way that ensures proper units are always output
#TODO: Allow single inputs as well as arrays (currently only arrays supported)
#TODO: all accel functions should have the fix_arrays and convert_to_frame() decorators

import numpy as np
import astropy.units as u
from galpy.potential import HernquistPotential, evaluatezforces, evaluateRforces
from gala.potential.potential import MiyamotoNagaiPotential
from gala.units import UnitSystem

from .convenience import mags
from .transforms import convert_to_frame
from .glob import fix_arrays, r_sun

#G = 4.301e-6 #kpc/Msun (km/s)^2
G = 4.516e-39 #kpc^3/Msun/s^2  (all accels will be in kpc/s^2) #kpc/s^2 * kpc^2/Msun
# Rsun = 8.0 #kpc
# vlsr = 220 #km/s
Rsun = 8.178 #kpc
vlsr = 232.8 #km/s
kmtokpc = 3.241e-17
rho_crit = 125.6 #critical density assuming LCDM, msun/kpc^3
c = 9.716e-12 #kpc/s

# #===================================================================
# # HELPER FUNCTIONS
# #===================================================================

# def norm(arr): #arr should have shape Nx3
# 	for i in range(len(arr)):
# 		arr[i] = arr[i]/np.sum(arr[i]**2)**0.5
# 	return arr

#===================================================================
# MODEL DEFINITIONS
#
# each model needs the following methods: 
#   __init(self, <params>) (see examples)
#   acc(self, x, y, z) 
#===================================================================

class Model:

	def __init__(self):
		self.name = 'Uninitialized'
		self.params = dict()
		self.param_names = []

		#track whether parameters are stored as regular or log10(param)
		self._logparams = []

		#track which parameters are disabled, i.e. they become the default
		self._disabled_params = [] #list of ints, 0 if the ith par. is enabled, 1 if disabled
		self._disabled_param_names = [] #list of strings, for convenience

	@property #shows up as self.nparams
	def nparams(self):
		return len(self.param_names)

	@property #shows up as self.n_opt_params
	def n_opt_params(self): #count the number of non-None objects in self.param_defaults dict
		return np.sum(np.array([not v is None for v in self.param_defaults])) #yeah it's gross but one liners are sick

	@property #shows up as self.n_req_params
	def n_req_params(self):
		return self.nparams - self.n_opt_params

	def param_names_to_str(self):
		out = ''
		for pname, i in self.param_names:
			if not self.param_defaults[i] is None:
				out += pname + ', '
			else:
				out += pname + '(opt.), ' 
		return out.rstrip(', ')

	#to be called when making a new model, after setting name, the param names, etc.
	def _finish_init_model(self, **kwargs):
		if len(kwargs) == 0: #handle the Model() init case, set everything = 1.
			params = dict()
			for j in self.param_names:
				params[j] = 1.
		else:
			params = kwargs

		self._disabled_params = np.zeros(len(self.param_names))

		#auto-fill any missing values (if they have defaults)
		while len(params) < len(self.param_names):
			for i, pname in enumerate(self.param_names):
				if not pname in params:
					params[pname] = self.param_defaults[i]
					self._disabled_params[i] = 1
					self._disabled_param_names.append(pname)

		self.set_params(params, ignore_name_check=False)
		self._logparams = [0]*self.nparams

	def set_params(self, params, ignore_name_check=False):

		#set params from args and param names (as well as the self._param grabbables)

		#assert that parameters have the correct length and that the names match
		#the name matching thing might be overkill but it prevents some potential headaches -- if it ends up being too slow it could be supressed
		assert self.nparams == len(params), f'{self.name} Model requires 0 or {self.n_req_params}-{self.nparams} arguments ({self.param_names_to_str()})'
		if not ignore_name_check:
			if not set(self.param_names) == set(params.keys()):
				raise Exception(f'Name Mismatch(es) in params: {self.name} Model has parameters {set(self.param_names)} but {set(params.keys())} were provided.')
		
		for i in range(len(self.param_names)):
			self.params[self.param_names[i]] = params[self.param_names[i]]

	def get_param_names(self):
		return param_names

	def accel(self, x, y, z, **kwargs): #should catch everything?
		raise NotImplementedError('Uninitialized model has no acc() method. Try initializing an existing model or defining your own.')

	@fix_arrays
	@convert_to_frame('gal')
	def alos(self, l, b, d, frame='gal', d_err=None, sun_pos=(r_sun, 0., 0.)): #includes solar accel!
		"""
		Compute the line-of-sight component of the acceleration. This is acceleration relative to the Sun. 

		:coord1-3: Galactocentric Cartesian coordinates (kpc) or Galactic longitude, latitude (deg) and heliocentric distance (kpc). Toggle between these options with the 'frame' flag.
		:frame: [default value = 'gal'] Toggle the input frame. Options are 'cart' for Galactocentric Cartesian (X,Y,Z), 'gal' for heliocentric Galactic coordinates (l,b,d), 'icrs' for equatorial coordinates (ra, dec, d), and 'ecl' for ecliptic coordinates (lam, bet, d) 
		:sun_pos: [optional, default value = (8.0, 0.0, 0.0) kpc] The position of the Sun in Galactocentric Cartesian coordinates (X,Y,Z).

		"""

		#heliocentric, can't use frame='cart' because that's Galactocentric
		x = -d*np.cos(l*np.pi/180)*np.cos(b*np.pi/180)
		y = d*np.sin(l*np.pi/180)*np.cos(b*np.pi/180)
		z = d*np.sin(b*np.pi/180)

		asun = np.array(self.accel(sun_pos[0], sun_pos[1], sun_pos[2])).T
		accels = np.array(self.accel(sun_pos[0] + x, sun_pos[1] + y, sun_pos[2] + z)).T - asun  #subtract off solar accel

		los_vecs = (np.array([x, y, z]/d).T)
		los_accels = np.sum(accels*los_vecs, axis=-1) #works for arrays and floats

		if d_err is not None:

			alos_plus_derr = self.alos(l, b, d+d_err, sun_pos=sun_pos)
			alos_minus_derr = self.alos(l, b, d-d_err, sun_pos=sun_pos)

			return los_accels, np.abs(alos_plus_derr - alos_minus_derr)/2

		else:
			return los_accels

	@fix_arrays
	@convert_to_frame('gal')
	def atan(self, l, b, d, frame='gal', sun_pos=(r_sun, 0., 0.), angular=True): #includes solar accel! Adapted from code by Lorenzo Addy
		"""
		Compute the magnitude of the tangential component of the acceleration, i.e. the "proper" acceleration. This is acceleration relative to the Sun, perpendicular to our line of sight. 

		:coord1-3: Galactocentric Cartesian coordinates (kpc) or Galactic longitude, latitude (deg) and heliocentric distance (kpc). Toggle between these options with the 'frame' flag.
		:frame: [default value = 'gal'] Toggle the input frame. Options are 'cart' for Galactocentric Cartesian (X,Y,Z), 'gal' for heliocentric Galactic coordinates (l,b,d), 'icrs' for equatorial coordinates (ra, dec, d), and 'ecl' for ecliptic coordinates (lam, bet, d) 
		:sun_pos: [optional, default value = (8.0, 0.0, 0.0) kpc] The position of the Sun in Galactocentric Cartesian coordinates (X,Y,Z).
		:angular: [optional, default value = True] Output is an angular acceleration if True, or a linear acceleration if False.

		"""

		#heliocentric, can't use frame='cart' because that's Galactocentric
		x = -d*np.cos(l*np.pi/180)*np.cos(b*np.pi/180)
		y = d*np.sin(l*np.pi/180)*np.cos(b*np.pi/180)
		z = d*np.sin(b*np.pi/180)

		asun = np.array(self.accel(sun_pos[0], sun_pos[1], sun_pos[2])).T
		accels = np.array(self.accel(sun_pos[0] + x, sun_pos[1] + y, sun_pos[2] + z)).T - asun  #subtract off solar accel

		los_vecs = (np.array([x, y, z]/d).T)
		los_accels = np.sum(accels*los_vecs, axis=-1) #works for arrays and floats

		tan_accels = np.sum((accels - los_accels*los_vecs)**2, axis=1)**0.5 #magnitude of tangential accels

		if angular: #kpc/s^2 -> radians/s^2
			tan_accels /= d #TODO: not sure this is correct because it doesn't include cos(b) for one component

		return tan_accels

	@fix_arrays
	@convert_to_frame('gal')
	def a_gal_sph(self, l, b, d, frame='gal', sun_pos=(r_sun, 0., 0.), angular=True): #includes solar accel! Adapted from code by Lorenzo Addy
		"""
		Compute the 3-dimensional heliocentric acceleration. This is acceleration relative to the Sun.

		:coord1-3: Galactocentric Cartesian coordinates (kpc) or Galactic longitude, latitude (deg) and heliocentric distance (kpc). Toggle between these options with the 'frame' flag.
		:frame: [default value = 'gal'] Toggle the input frame. Options are 'cart' for Galactocentric Cartesian (X,Y,Z), 'gal' for heliocentric Galactic coordinates (l,b,d), 'icrs' for equatorial coordinates (ra, dec, d), and 'ecl' for ecliptic coordinates (lam, bet, d) 
		:sun_pos: [optional, default value = (8.0, 0.0, 0.0) kpc] The position of the Sun in Galactocentric Cartesian coordinates (X,Y,Z).
		:return: $a _ \\mathrm{los}$ (acceleration along our line of sight); $a _ \\mathrm{l}$ (acceleration in the Galactic longitude direction); $a _ \\mathrm{b}$ (acceleration in the Galactic latitude direction), (kpc/s$^2$)
		:rtype: array-like (float,); array-like (float,); array-like (float,)
		"""

		l *= np.pi/180
		b *= np.pi/180

		#heliocentric, can't use frame='cart' because that's Galactocentric
		x = -d*np.cos(l)*np.cos(b)
		y = d*np.sin(l)*np.cos(b)
		z = d*np.sin(b)

		asun = np.array(self.accel(sun_pos[0], sun_pos[1], sun_pos[2])).T
		accels = np.array(self.accel(sun_pos[0] + x, sun_pos[1] + y, sun_pos[2] + z)).T - asun  #subtract off solar accel

		rhats = (np.array([x, y, z]/d).T)
		rhats = np.nan_to_num(rhats, nan=0.0, posinf=0.0, neginf=0.0) #fixes any divide by 0
		alos = np.dot(accels*rhats)

		atan_vec = accels - alos*rhats

		bhats = np.array([np.cos(l)*np.sin(b), -np.sin(l)*np.sin(b), np.cos(b)])
		lhats = np.array([np.sin(l), np.cos(l), 0])

		atan_b = np.dot(atan_vec, bhats)
		atan_l = np.dot(atan_vec, lhats)

		#fix polar axis
		polar_indx = (b == 90.) | (b == -90.)
		np.place(atan_l, polar_indx, 0.)
		np.place(atan_b, (b == 90.), -mags(atan_vec[(b == 90.)]))
		np.place(atan_b, (b == -90.), mags(atan_vec[(b == -90.)]))

		if angular: #kpc/s^2 -> radians/s^2
			atan_b /= d 
			atan_l = atan_l / d * np.cos(b)

		return alos, atan_l, atan_b

	#model1 + model2 returns a new CompositeModel
	def __add__(self, model2):
		if isinstance(model2, CompositeModel): #if a CompositeModel is involved, use the CompositeModel __add__ method instead
			return model2 + self
		else: #neither Model is CompositeModel
			out = CompositeModel()
			out.name = f'{self.name}+{model2.name}'
			out.models = [self] + [model2]

		return out

	def toggle_log_params(self, l, adjust_vals=True):
		#l: list of ints to toggle (i.e. [0,2] would switch the toggle for params 0 and 2)
		if type(l) is int:
			self._logparams[l] = not(self._logparams[l])
			if adjust_vals:
				self.params[self.param_names[l]] = np.log10(self.params[self.param_names[l]])
		else:
			for i in l:
				self._logparams[i] = not(self._logparams[i])
				if adjust_vals:
					self.params[self.param_names[i]] = np.log10(self.params[self.param_names[i]])

	def log_corr_params(self):
		out = list(self.params.values())
		for i in range(len(out)):
			if self._logparams[i]:
				out[i] = 10**out[i]
		return out 

	#helper function to make it easier to get/assign default params
	def get_param_default(self, param_name):
		i = self.param_names.index(param_name)
		return self.param_defaults[i]



class CompositeModel:

	def __init__(self):
		self.name = 'Uninitialized'
		self.models = [] #tracks models that build up this Potential, this is the main way to access them

	@property #shows up as self.params, -> a list of dicts
	def params(self):
		out = []
		for m in self.models:
			out.append(m.params)
		return out
	
	@property #shows up as self.param_names, -> a list of lists of strings
	def param_names(self):
		out = []
		for m in self.models:
			out.append(m.param_names)
		return out

	@property #shows up as self.nparams, -> a list of ints
	def nparams_list(self):
		out = []
		for m in self.models:
			out.append(m.nparams)
		return out

	@property #shows up as self.nparams, -> int (sum of all component model nparams)
	def nparams(self):
		out = 0
		for m in self.models:
			out += m.nparams
		return out

	def set_params(self, params):
		#set params for each of the models
		#changing only one model's parameters must be done manually via self.models[i].set_params()
		for i, m in enumerate(self.models):
			m.set_params(params[i])

	@fix_arrays
	@convert_to_frame('cart')
	def accel(self, x, y, z, frame='cart', **kwargs): #should catch everything? Don't think we currently pass any kwargs to accel though
		if len(self.models) == 0:
			raise NotImplementedError('Uninitialized CompositeModel has no Models.')
		else:
			#print()
			try: #this try-except checks whether x, y, z are ints or array-like
				out = np.zeros((3, len(x)))
			except TypeError:
				out = np.zeros(3) 
			for m in self.models:
				#print(m.name)
				#print(np.array(m.accel(x, y, z, **kwargs))*3.086e21)
				out += m.accel(x, y, z, **kwargs)
			return out[0], out[1], out[2]

	@fix_arrays
	@convert_to_frame('gal')
	def alos(self, l, b, d, frame='gal', sun_pos=(r_sun, 0., 0.)): #includes solar accel!

		#heliocentric, can't use frame='cart' because that's Galactocentric
		x = -d*np.cos(l*np.pi/180)*np.cos(b*np.pi/180)
		y = d*np.sin(l*np.pi/180)*np.cos(b*np.pi/180)
		z = d*np.sin(b*np.pi/180)

		alossun = np.array(self.accel(sun_pos[0], sun_pos[1], sun_pos[2])).T
		accels = np.array(self.accel(sun_pos[0] + x, sun_pos[1] + y, sun_pos[2] + z)).T - alossun  #subtract off solar accel

		los_vecs = (np.array([x, y, z]/d).T)
		if len(np.shape(los_vecs)) > 1: #TODO: make this less clunky (requires allowing for array or non-array input)
			los_accels = np.sum(accels*los_vecs, axis=1)
		else:
			los_accels = np.sum(accels*los_vecs)

		return los_accels

	#model1 + model2 returns a new composite potential model
	#can be used with either Model or CompositeModel
	def __add__(self, model2):
		out = CompositeModel()
		out.name = f'{self.name}+{model2.name}'

		if isinstance(model2, Model):
			out.models = self.models + [model2]
		elif isinstance(model2, CompositeModel):
			out.models = self.models + model2.models

		return out

	def toggle_log_params(self, aloi): #aloi is a list of (opt. lists of) ints with the same size as self.models
		#(i.e. aloi = [0,[0,2]] would switch the toggle for param 0 in model 0, and params 0 & 2 in model 1)
		# aloi = [0,[0,2]] and [[0],[0,2]] are equivalent, [0,[0,2]] and [0,0,2] are *not*
		for l, m in zip(aloi, self.models):
			m.toggle_log_params(l)

#--------------------------
# NFW
#--------------------------
class NFW(Model):

	#rho0 = Msun/kpc^3, rs = kpc
	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'NFW'
		self.param_names = ['m_vir', 'r_s', 'q']
		self.param_defaults = [None, None, 1.] #None if required param
		self._finish_init_model(**kwargs)

	def accel(self, x, y, z):
		mvir, rs, q = self.log_corr_params()

		#TODO: remove this later if it isn't a problem
		# if q is None:
		# 	r = (x**2 + y**2 + z**2)**0.5
		# else:
		r = (x**2 + y**2 + (z/q)**2)**0.5
		R = (x**2 + y**2)**0.5

		rvir = (3/(4*np.pi) * mvir / (200*rho_crit))**(1/3) 
		c = rvir / rs
		# rho0 = mvir / (4*np.pi * rs**3 * (np.log(1 + c) - c/(1+c)) )

		# ar = -4*np.pi*G*rho0*rs**2*R*(rs * np.log(1 + r/rs) / r**3 - 1 / (r**2 * (1 + r/rs)))
		# az = -4*np.pi*G*rho0*rs**2*z*(rs * np.log(1 + r/rs) / r**3 - 1 / (r**2 * (1 + r/rs)))

		# ax = ar*x/R
		# ay = ar*y/R

		ar_r = (G*mvir)/(np.log(1+c) - c/(1+c)) * (r/(r+rs) - np.log(1 + r/rs))/r**3 #ar/r
		ax = ar_r*x
		ay = ar_r*y
		az = ar_r*z #also z/q?

		return ax, ay, az

#--------------------------
# Hernquist
#--------------------------
class Hernquist(Model):

	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Hernquist'
		self.param_names = ['m_tot', 'r_s']
		self.param_defaults = [None, None] #None if required param
		self._finish_init_model(**kwargs)

	def accel(self, x, y, z):
		mtot, rs = self.log_corr_params()

		R = (x**2 + y**2)**0.5

		pot = HernquistPotential(2*mtot*u.M_sun, rs*u.kpc)
		az = evaluatezforces(pot, R*u.kpc, z*u.kpc, ro=Rsun*u.kpc, vo=vlsr*u.km/u.s)*1.028e-30 #km/s/Myr to kpc/s^2
		ar = evaluateRforces(pot, R*u.kpc, z*u.kpc, ro=Rsun*u.kpc, vo=vlsr*u.km/u.s)*1.028e-30 #km/s/Myr to kpc/s^2

		ax = ar*x/R
		ay = ar*y/R

		return ax, ay, az

#--------------------------
# Plummer
#--------------------------
class Plummer(Model):

	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Plummer'
		self.param_names = ['m_tot', 'r_s', 'x', 'y', 'z'] 
		self.param_defaults = [None, None, 0., 0., 0.] #None if required param
		self._finish_init_model(**kwargs)

	def accel(self, x, y, z):

		mtot, rs, xc, yc, zc = self.log_corr_params()

		if (xc is None) or (yc is None) or (zc is None):
			xc, yc, zc = 0., 0., 0.

		xi, yi, zi = x-xc, y-yc, z-zc

		ri = np.sqrt(xi**2 + yi**2 + zi**2)
		frac = -G*mtot/((ri**2 + rs**2)**(3/2))
		ax = frac*xi
		ay = frac*yi
		az = frac*zi
	
		return ax, ay, az

#--------------------------
# Miyamoto-Nagai Disk
#--------------------------
class MiyamotoNagaiDisk(Model):

	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Miyamoto-Nagai Disk'
		self.param_names = ['m_tot', 'a', 'b']
		self.param_defaults = [None, None, None] #None if required param
		self._finish_init_model(**kwargs)

	def accel(self,x,y,z):
		mtot, a, b = self.log_corr_params()

		R = (x**2 + y**2)**0.5

		abz = (a + (z**2 + b**2)**0.5)
		ar = -G*mtot*R/(R**2 + abz**2)**(3/2)
		az = -G*mtot*z*abz/((b**2 + z**2)**0.5*(R**2 + abz**2)**(3/2))

		ax = ar*x/R
		ay = ar*y/R

		return ax, ay, az

	# #gala version (very slow!)
	# def accel(self,x,y,z):
	# 	mtot, a, b = self.log_corr_params()

	# 	pot = MiyamotoNagaiPotential(mtot*u.solMass, a*u.kpc, b*u.kpc, units=UnitSystem(u.kpc, u.s, u.solMass, u.radian))
	# 	a = pot.acceleration(np.array([x,y,z])*u.kpc).value
		
	# 	if isinstance(x, float):
	# 		return a[0][0], a[1][0], a[2][0]
	# 	else:
	# 		return a[0], a[1], a[2]

#--------------------------
# Point Mass
#--------------------------
class PointMass(Model):

	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Point Mass'
		self.param_names = ['m', 'x', 'y', 'z']
		self.param_defaults = [None, 0., 0., 0.] #None if required param
		self._finish_init_model(**kwargs)

	def accel(self,x,y,z):
		m, x0, y0, z0 = self.log_corr_params()

		xi = x0 - x
		yi = y0 - y
		zi = z0 - z

		ai = G*m/(xi**2 + yi**2 + zi**2)**(3/2)
		ax = ai*xi
		ay = ai*yi
		az = ai*zi

		return ax, ay, az

#--------------------------
# Alpha Beta (flexible implementation)
#--------------------------
class OortExpansion(Model):

	#vert_only flg allows you to just use the vertical potential
	def __init__(self, vert_only=False, **kwargs):
		super().__init__()
		self.name = 'Quillen Flexible'
		self.param_names = ['alpha1', 'alpha2', 'beta', 'vcirc']
		self.param_defaults = [None, 0., 0., vlsr] #None if required param
		self._vert_only = vert_only
		self._finish_init_model(**kwargs)

	def set_vert_only(self, b):
		self._vert_only = b

	def accel(self,x,y,z):
		alpha1, alpha2, beta, vcirc = self.log_corr_params()

		#TODO: remove this later if there aren't problems
		# if alpha2 is None:
		# 	alpha2 = 0.

		R = (x**2 + y**2)**0.5

		az = -alpha1*z - alpha2*z**2

		if self._vert_only:
			ar = 0.
		else:
			if beta == 0.:
				ar = vcirc**2/R*kmtokpc**2
			else:
				ar = (vcirc**2*kmtokpc**2)*((1./Rsun)**(2.*beta))*(R**((2.*beta)-1.))

		ax = -ar*x/R
		ay = -ar*y/R

		return ax, ay, az

#--------------------------
# Cross
#--------------------------
class Cross(Model):

	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Cross'
		self.param_names = ['alpha', 'gamma']
		self.param_defaults = [None, None] #None if required param
		self._finish_init_model(**kwargs)

	def accel(self,x,y,z):
		alpha, gamma = self.log_corr_params()

		R = np.sqrt(x*x + y*y)

		#gamma *= -1

		#ar = (vlsr**2/R)*(1.+gamma*(z**2))*kmtokpc**2
		ar = (vlsr**2/R)*kmtokpc**2 - gamma*z**2/R
		#az = -alpha*z -(vlsr**2*np.log(R/Rsun)*2.*gamma*z)*kmtokpc**2
		az = -( alpha*z - 2*gamma*z*np.log(R/Rsun) )

		ax = -ar*x/R
		ay = -ar*y/R

		return ax, ay, az

#--------------------------
# Anharmonic Disk
#--------------------------
class AnharmonicDisk(Model): #TODO: can in theory take as many terms of the power expansion as you want

	#alpha1 = 1/s^2
	#alpha2 = 1/s^2/kpc
	#alpha3 = 1/s^2/kpc^2
	def __init__(self, neg_alpha2=False, **kwargs):
		super().__init__()
		self.name = 'Anharmonic Disk'
		self.neg_alpha2 = neg_alpha2
		self.param_names = ['alpha1', 'alpha2', 'alpha3']
		self.param_defaults = [None, 0., 0.] #None if required param
		self._finish_init_model(**kwargs)

	def accel(self, x, y, z):
		alpha1, alpha2, alpha3 = self.log_corr_params()

		#TODO: remove this later if no problems
		# #allow alpha2 and 3 to be None; note large overlap with qbeta, might want to combine these two?
		# if alpha2 is None:
		# 	alpha2 = 0.
		# if alpha3 is None:
		# 	alpha3 = 0.

		if self.neg_alpha2:
			alpha2 *= -1

		R = (x**2 + y**2)**0.5

		ar = -vlsr**2/R*kmtokpc**2
		az = -alpha1*z - alpha2*z**2 - alpha3*z**3

		ax = ar*x/R
		ay = ar*y/R

		return ax, ay, az

#--------------------------
# Sinusoidal Disk
#--------------------------
class SinusoidalDisk(Model):

	#alpha = 1/s^2
	#amp = kpc
	#lambda = kpc
	#phi = radians
	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Anharmonic Disk'
		self.param_names = ['alpha', 'amp', 'lam', 'phi']
		self.param_defaults = [None, None, None, None] #None if required param
		self._finish_init_model(**kwargs)

	def accel(self, x, y, z):
		alpha, amp, lam, phi = self.log_corr_params()

		k = 2*np.pi/lam

		R = (x**2 + y**2)**0.5

		ar = -vlsr**2/R*kmtokpc**2 - alpha*amp*k*np.cos(k*R + phi)*(amp*np.sin(k*R+phi) + z)
		az = -alpha*(amp*np.sin(k*R+phi)+z)

		ax = ar*x/R
		ay = ar*y/R

		return ax, ay, az

#--------------------------
# Isothermal Disk
#--------------------------
class IsothermalDisk(Model):

	#sigma = km/s (velocity dispersion)
	#z0 = kpc, local grav. midplane
	#b = kpc, scale height
	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Isothermal Disk'
		self.param_names = ['sigma', 'z0', 'b']
		self.param_defaults = [None, None, None] #None if required param
		self._finish_init_model(**kwargs)

	def accel(self, x, y, z):
		sigma, z0, b = self.log_corr_params()

		R = (x**2 + y**2)**0.5

		ar = -vlsr**2/R*kmtokpc**2
		az = -(sigma*kmtokpc)**2/b * np.tanh((z-z0)/b)

		ax = ar*x/R
		ay = ar*y/R

		return ax, ay, az

#--------------------------
# Local Expansion
#--------------------------
class LocalExpansion(Model):

	#dadr = 1/s^2
	#dadphi = 1/s^2
	#dadz = 1/s^2
	def __init__(self, neg_dadr=False, neg_dadphi=False, **kwargs):
		super().__init__()
		self.name = 'Local Expansion'
		self.param_names = ['dadr', 'dadphi', 'dadz']
		self.param_defaults = [None, None, None] #None if required param
		self._finish_init_model(**kwargs)
		self.neg_dadr = neg_dadr
		self.neg_dadphi = neg_dadphi

	def accel(self, x, y, z):
		dadr, dadphi, dadz = self.log_corr_params()

		#allows for negative values when using log 
		if self.neg_dadr:
			dadr *= -1
		if self.neg_dadphi:
			dadphi *= -1

		R = (x**2 + y**2)**0.5
		phi = np.arctan2(y,x)

		arsun = vlsr**2/Rsun*kmtokpc**2
		ar = -(arsun+dadr*(R-Rsun))
		aphi = -dadphi*phi*R
		az = -dadz*z

		ax = ar*x/R + aphi*y/R
		ay = ar*y/R + aphi*x/R

		return ax, ay, az

#---------------------------------------------------------------------------
# (Outdated) PTA Expansion
# see Damour & Taylor (1991), Nice & Taylor (1995), Holmberg & Flynn (2004), Lazaridis et al. (2009)
#---------------------------------------------------------------------------
class DamourTaylorPotential(Model):

	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Damour-Taylor Potential'
		self.param_names = []
		self.param_defaults = []
		self._finish_init_model(**kwargs)

	def accel(self, x, y, z):

		R = (x**2 + y**2)**0.5 #galactocentric

		ar = -(vlsr*kmtokpc)**2/R
		az = -np.sign(z)*(2.27*np.abs(z) + 3.68*(1 - np.exp(-4.31*np.abs(z))))*3.241e-31 #to kpc/s^2 in the weird expansion units, by default this gives 1e-9 cm/s^2

		ax = ar*x/R
		ay = ar*y/R

		return ax, ay, az

#---------------------------------------------------------------------------
# Spherical 
# see Phinney (1993)
#---------------------------------------------------------------------------
class SphericalFlatRC(Model):

	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Spherical Potential with Flat Rotation Curve'
		self.param_names = ['vcirc']
		self.param_defaults = [0.] #None if required param
		self._finish_init_model(**kwargs)

	def accel(self, x, y, z):

		if vcirc == 0.:
			vcirc = vlsr

		r = (x**2 + y**2 + z**2)**0.5 #galactocentric
		l = np.arctan2(y, -x)
		b = np.arcsin(z/r)

		ar = -(vcirc*kmtokpc)**2/r

		ax = -ar*np.cos(l)*np.cos(b)
		ay = ar*np.sin(l)*np.cos(b)
		az = ar*np.sin(b)

		return ax, ay, az

#---------------------------------------------------------------------------
# Uniform Line-of-Sight Acceleration 
# useful for things like globular clusters where the MW acceleration field is ~ constant
#---------------------------------------------------------------------------
class UniformAlos(Model):

	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Uniform Line-of-Sight Acceleration'
		self.param_names = ['alos']
		self.param_defaults = [0.] #None if required param
		self._finish_init_model(**kwargs)

	def accel(self, x, y, z, **kwargs): #should catch everything?
		raise NotImplementedError('UniformAlos model has no acc() method, it is meant to be used with alos().')

	def alos(self, l, b, d, **kwargs): #should catch everything?
		return np.zeros(len(l)) + self.alos

#---------------------------------------------------------------------------
# Uniform 3D Acceleration 
# useful for things like globular clusters where the MW acceleration field is ~ constant
#---------------------------------------------------------------------------
class Uniform3DAccel(Model):

	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Uniform Acceleration'
		self.param_names = ['ax', 'ay', 'az']
		self.param_defaults = [0., 0., 0.] #None if required param
		self._finish_init_model(**kwargs)

	def accel(self, x, y, z, **kwargs): #should catch everything?
		ax, ay, az = self.log_corr_params()
		return np.zeros(len(x)) + ax, np.zeros(len(x)) + ay, np.zeros(len(x)) + az

#-----------------------------------------
# Cox & Gomez (2002) Spiral Arm Potential
# Rather than do out all the derivatives I just numerically evaluate the acceleration at each point
#    this is slower than if I had evaluated things out, but much better for my sanity
#    might eventually add in functionality for linear combinations eventually, but for now we're just going to use 1 term in the sum (n=1)
#    if speed is enough of a problem that the linear combination inside the model becomes important we're also better off analytically computing the derivative
#-----------------------------------------
class CoxGomezSpiralArm(Model):

	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Cox-Gomez Spiral Arm'
		self.param_names = ['N', 'alpha', 'rs', 'rh', 'phi0', 'rho0']
		#N: number of spiral arms
		#alpha: the pitch angle (in degrees)
		#rs: scale length, kpc (i.e. double-exp disk)
		#h: scale height, kpc (i.e. double-exp disk)
		#r0: fiducial radius, kpc
		#phi0: fiducial rotation phase (rad)
		#rho_0: fiducial density at r0 and phi0, Msun/kpc^3

		self.param_defaults = [2., 15., 7., 0.18, 0., 3.123e7] #None if required param; these are from Cox & Gomez (2002) as their fiducial parameters
		self._finish_init_model(**kwargs)

	def set_number_arms(self, N):
		#this is a really dumb helper function that helps with optimization. However, it is really symptom of a greater problem that you can't currently lock parameters when fitting.
		self.params['N'] = N

	#helper function to evlauate the potential, since this is done multiple times below
	def pot(self, x, y, z):
		N, alpha, rs, rh, phi0, rho0 = self.log_corr_params()

		r = (x**2 + y**2 + z**2)**0.5
		phi = np.arctan2(y,-x) #left-handed, in radians

		Kn = N/(r*np.sin(alpha*np.pi/180))
		Bn = Kn*rh*(1 + 0.4*Kn*rh)
		Dn = (1 + Kn*rh + 0.3*(Kn*rh)**2)/(1 + 0.3*Kn*rh)
		gamma = N*(phi - phi0 - np.log(r/r_sun)/np.tan(alpha*np.pi/180)) #something like radians

		out = -4*np.pi*G*rh*rho0 * np.exp(-(r - r_sun)/rs) * (1 / (Kn * Dn)) * np.cos(gamma) * (np.cosh(Kn*z/Bn))**(-Bn)

		return out 

	def accel(self, x, y, z, sun_pos=(r_sun, 0., 0.)): #this is probably pretty slow, fyi

		#compute accelerations of each object
		dr = 0.001 #kpc

		pot_base    = self.pot(x, y, z)
		pot_plus_dx = self.pot(x+dr, y,    z   )
		pot_plus_dy = self.pot(x,    y+dr, z   )
		pot_plus_dz = self.pot(x,    y,    z+dr)
		
		dpot_dx = -(pot_plus_dx - pot_base)/dr
		dpot_dy = -(pot_plus_dy - pot_base)/dr
		dpot_dz = -(pot_plus_dz - pot_base)/dr

		#compute Solar acceleration and subtract off
		pot_base_sun    = self.pot(sun_pos[0],    sun_pos[1],    sun_pos[2]   )
		pot_plus_dx_sun = self.pot(sun_pos[0]+dr, sun_pos[1],    sun_pos[2]   )
		pot_plus_dy_sun = self.pot(sun_pos[0],    sun_pos[1]+dr, sun_pos[2]   )
		pot_plus_dz_sun = self.pot(sun_pos[0],    sun_pos[1],    sun_pos[2]+dr)
		
		dpot_dx_sun = -(pot_plus_dx_sun - pot_base_sun)/dr
		dpot_dy_sun = -(pot_plus_dy_sun - pot_base_sun)/dr
		dpot_dz_sun = -(pot_plus_dz_sun - pot_base_sun)/dr

		ax = dpot_dx - dpot_dx_sun
		ay = dpot_dy - dpot_dy_sun
		az = dpot_dz - dpot_dz_sun

		return ax, ay, az

#-------------------------------------------------
# Generic Gala Potential Wrapper
# allows us to easily alos for any Gala potential
# TODO: cannot update params right now
#-------------------------------------------------
class GalaPotential(Model):

	#pot: a (instantiated) gala potential object
	def __init__(self, pot, **kwargs):
		super().__init__()
		self.name = 'Gala Potential Instance'
		self.param_names = []
		self.param_defaults = []
		self._finish_init_model(**kwargs)
		self.pot = pot

	def accel(self, x, y, z): 

		a = self.pot.acceleration(np.array([x,y,z])*u.kpc).to(u.kpc/u.s**2).value
		
		#have to do this to homogenize output to the correct type/shape
		if isinstance(x, float): #fails if x is an int (but it shouldn't ever be...? I think?)
			return a[0][0], a[1][0], a[2][0]
		else:
			return a[0], a[1], a[2]

		return ax, ay, az

#--------------------------

#-------------------------------------------------
# Generic Galpy Potential Wrapper
# allows us to easily alos for any Galpy potential
# TODO: cannot update params right now
#-------------------------------------------------
class GalpyPotential(Model):

	#pot: a (instantiated) gala potential object
	def __init__(self, pot, **kwargs):
		super().__init__()
		self.name = 'Galpy Potential Instance'
		self.param_names = []
		self.param_defaults = []
		self._finish_init_model(**kwargs)
		self.pot = pot

	def accel(self, x, y, z): 

		R = (x**2 + y**2)**0.5

		az = (evaluatezforces(self.pot, R*u.kpc, z*u.kpc, ro=Rsun*u.kpc, vo=vlsr*u.km/u.s)*(u.km/u.s/u.Myr)).to(u.kpc/u.s**2).value #convert from galpy coords
		ar = (evaluateRforces(self.pot, R*u.kpc, z*u.kpc, ro=Rsun*u.kpc, vo=vlsr*u.km/u.s)*(u.km/u.s/u.Myr)).to(u.kpc/u.s**2).value

		ax = ar*x/R
		ay = ar*y/R

		return ax, ay, az

#--------------------------