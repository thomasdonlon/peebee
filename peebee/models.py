#helper file for model_diff.py, stores all the info about models 
#Written by Tom Donlon, 2023, UAH

#TODO: dict names can collide when adding two models if param names are identical
#TODO: implement astropy units in a way that ensures proper units are always output
#TODO: Allow single inputs as well as arrays (currently only arrays supported)

import numpy as np
import astropy.units as u
from galpy.potential import HernquistPotential, evaluatezforces, evaluateRforces
from gala.potential.potential import MiyamotoNagaiPotential
from gala.units import UnitSystem

from .transforms import convert_to_frame
from .glob import fix_arrays

#G = 4.301e-6 #kpc/Msun (km/s)^2
G = 4.516e-39 #kpc^3/Msun/s^2  (all accels will be in kpc/s^2) #kpc/s^2 * kpc^2/Msun
# Rsun = 8.0 #kpc
# vlsr = 220 #km/s
Rsun = 8.178 #kpc
vlsr = 232.8 #km/s
kmtokpc = 3.241e-17
rho_crit = 125.6 #msun/kpc^3
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

	@property #shows up as self.nparams
	def nparams(self):
		return len(self.param_names)

	#to be called when making a new model, after setting name, the param names, etc.
	def _finish_init_model(self, **kwargs):
		if len(kwargs) == 0: #handle the Model() init case, set everything = 1.
			params = dict()
			for j in self.param_names:
				params[j] = 1.
		else:
			params = kwargs
		self.set_params(params, ignore_name_check=True)
		self._logparams = [0]*self.nparams

	def set_params(self, params, ignore_name_check=False):

		#set params from args and param names (as well as the self._param grabbables)

		#assert that parameters have the correct length and that the names match
		#the name matching thing might be overkill but it prevents some potential headaches -- if it ends up being too slow it could be supressed
		assert self.nparams == len(params), f'{self.name} Model requires 0 or {self.nparams} arguments ({self.param_names})'
		if not ignore_name_check:
			assert set(self.params.keys()) == set(params.keys()), f'Name Mismatch(es) in params: {self.name} Model has parameters {set(self.param_names)} but {set(params.keys())} were provided.'
		
		for i in range(len(self.param_names)):
			self.params[self.param_names[i]] = params[self.param_names[i]]

	def get_param_names(self):
		return param_names

	def accel(self, x, y, z, **kwargs): #should catch everything?
		raise NotImplementedError('Uninitialized model has no acc() method. Try initializing an existing model or defining your own.')

	@fix_arrays
	@convert_to_frame('gal')
	def alos(self, l, b, d, frame='gal', d_err=None): #includes solar accel!

		#heliocentric
		x = -d*np.cos(l*np.pi/180)*np.cos(b*np.pi/180)
		y = d*np.sin(l*np.pi/180)*np.cos(b*np.pi/180)
		z = d*np.sin(b*np.pi/180)

		alossun = np.array(self.accel(Rsun,0.,0.)).T
		accels = np.array(self.accel(Rsun + x, y, z)).T - alossun  #subtract off solar accel

		los_vecs = (np.array([x, y, z]/d).T)
		if len(np.shape(los_vecs)) > 1: #TODO: make this less clunky (requires allowing for array or non-array input)
			los_accels = np.sum(accels*los_vecs, axis=1)
		else:
			los_accels = np.sum(accels*los_vecs)

		if d_err is not None:

			alos_plus_derr = self.alos(l, b, d+d_err)
			alos_minus_derr = self.alos(l, b, d-d_err)

			return los_accels, np.abs(alos_plus_derr - alos_minus_derr)/2

		else:
			return los_accels

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
	def nparams(self):
		out = []
		for m in self.models:
			out.append(m.nparams)
		return out

	def set_params(self, params):
		#set params for each of the models
		#changing only one model's parameters must be done manually via self.models[i].set_params()
		for i, m in enumerate(self.models):
			m.set_params(params[i])

	def accel(self, x, y, z, **kwargs): #should catch everything? Don't think we currently pass any kwargs to accel though
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

	def alos(self, l, b, d): #includes solar accel!
		#heliocentric
		x = -d*np.cos(l*np.pi/180)*np.cos(b*np.pi/180)
		y = d*np.sin(l*np.pi/180)*np.cos(b*np.pi/180)
		z = d*np.sin(b*np.pi/180)

		alossun = np.array(self.accel(Rsun,0.,0.)).T
		#accels = np.array([a - alossun for a in np.array(self.accel(Rsun + x, y, z)).T])  #subtract off solar accel
		accels = np.array(self.accel(Rsun + x, y, z)).T - alossun  #subtract off solar accel
		# print('alossun', alossun)
		# print('accels', np.array(self.accel(Rsun + x, y, z)).T)
		# print('Relative accels', accels)

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
		self._finish_init_model(**kwargs)

	def accel(self, x, y, z):
		mvir, rs, q = self.log_corr_params()

		if q is None:
			r = (x**2 + y**2 + z**2)**0.5
		else:
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

		return ax,ay,az

#--------------------------
# Hernquist
#--------------------------
class Hernquist(Model):

	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Hernquist'
		self.param_names = ['m_tot', 'r_s']
		self._finish_init_model(**kwargs)

	def accel(self, x, y, z):
		mtot, rs = self.log_corr_params()

		R = (x**2 + y**2)**0.5

		pot = HernquistPotential(2*mtot*u.M_sun, rs*u.kpc)
		az = evaluatezforces(pot, R*u.kpc, z*u.kpc, ro=Rsun*u.kpc, vo=vlsr*u.km/u.s)*1.028e-30 #km/s/Myr to kpc/s^2
		ar = evaluateRforces(pot, R*u.kpc, z*u.kpc, ro=Rsun*u.kpc, vo=vlsr*u.km/u.s)*1.028e-30 #km/s/Myr to kpc/s^2

		ax = ar*x/R
		ay = ar*y/R

		return ax,ay,az

#--------------------------
# Miyamoto-Nagai Disk
#--------------------------
class MiyamotoNagaiDisk(Model):

	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Miyamoto-Nagai Disk'
		self.param_names = ['m_tot', 'a', 'b']
		self._finish_init_model(**kwargs)

	def accel(self,x,y,z):
		mtot, a, b = self.log_corr_params()

		R = (x**2 + y**2)**0.5

		abz = (a + (z**2 + b**2)**0.5)
		ar = -G*mtot*R/(R**2 + abz**2)**(3/2)
		az = -G*mtot*z*abz/((b**2 + z**2)**0.5*(R**2 + abz**2)**(3/2))

		ax = ar*x/R
		ay = ar*y/R

		return ax,ay,az

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

		return ax,ay,az

#--------------------------
# Quillen Beta
#--------------------------
class QuillenFlexible(Model): #TODO: a more elegant updating scheme that takes into account whether alpha2 and beta are set

	#vert_only flg allows you to just use the vertical potential
	def __init__(self, vert_only=False, **kwargs):
		super().__init__()
		self.name = 'Quillen Flexible'
		self.param_names = ['alpha1', 'alpha2', 'beta']
		self._vert_only = vert_only
		self._finish_init_model(**kwargs)

	def set_vert_only(self, b):
		self._vert_only = b

	def accel(self,x,y,z):
		alpha1, alpha2, beta = self.log_corr_params()

		if alpha2 is None:
			alpha2 = 0.

		R = (x**2 + y**2)**0.5

		#az = -alpha1*z - alpha2*(z*z)*np.sign(z)
		az = -alpha1*z - alpha2*z**2
		if self._vert_only:
			ar = 0.
		else:
			if beta is None:
				ar = vlsr**2/R*kmtokpc**2
			else:
				ar = (vlsr**2*kmtokpc**2)*((1./Rsun)**(2.*beta))*(R**((2.*beta)-1.))

		ax = -ar*x/R
		ay = -ar*y/R

		return ax,ay,az

#--------------------------
# Quillen Beta
#--------------------------
class QuillenVariableVcirc(Model): #TODO: a more elegant updating scheme that takes into account whether alpha2 and beta are set

	#vert_only flg allows you to just use the vertical potential
	def __init__(self, vert_only=False, **kwargs):
		super().__init__()
		self.name = 'Quillen Flexible'
		self.param_names = ['alpha1', 'alpha2', 'beta', 'vcirc']
		self._vert_only = vert_only
		self._finish_init_model(**kwargs)

	def set_vert_only(self, b):
		self._vert_only = b

	def accel(self,x,y,z):
		alpha1, alpha2, beta, vcirc = self.log_corr_params()

		if alpha2 is None:
			alpha2 = 0.

		R = (x**2 + y**2)**0.5

		#az = -alpha1*z - alpha2*(z*z)*np.sign(z)
		az = -alpha1*z - alpha2*z**2
		if self._vert_only:
			ar = 0.
		else:
			if beta is None:
				ar = vcirc**2/R*kmtokpc**2
			else:
				ar = (vcirc**2*kmtokpc**2)*((1./Rsun)**(2.*beta))*(R**((2.*beta)-1.))

		ax = -ar*x/R
		ay = -ar*y/R

		return ax,ay,az

#--------------------------
# Cross
#--------------------------
class Cross(Model):

	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Cross'
		self.param_names = ['alpha', 'gamma']
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

		return ax,ay,az

#--------------------------
# Anharmonic Disk
#--------------------------
class AnharmonicDisk(Model):

	#alpha1 = 1/s^2
	#alpha2 = 1/s^2/kpc
	#alpha3 = 1/s^2/kpc^2
	def __init__(self, neg_alpha2=False, **kwargs):
		super().__init__()
		self.name = 'Anharmonic Disk'
		self.neg_alpha2 = neg_alpha2
		self.param_names = ['alpha1', 'alpha2', 'alpha3']
		self._finish_init_model(**kwargs)

	def accel(self, x, y, z):
		alpha1, alpha2, alpha3 = self.log_corr_params()

		#allow alpha2 and 3 to be None; note large overlap with qbeta, might want to combine these two?
		if alpha2 is None:
			alpha2 = 0.
		if alpha3 is None:
			alpha3 = 0.

		if self.neg_alpha2:
			alpha2 *= -1

		R = (x**2 + y**2)**0.5

		ar = -vlsr**2/R*kmtokpc**2
		#print(-alpha1*z, - alpha2*z**2, - alpha3*z**3)
		az = -alpha1*z - alpha2*z**2 - alpha3*z**3

		ax = ar*x/R
		ay = ar*y/R

		return ax,ay,az

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
		self.param_names = ['alpha', 'amp', 'lambda', 'phi']
		self._finish_init_model(**kwargs)

	def accel(self, x, y, z):
		alpha, amp, lam, phi = self.log_corr_params()

		k = 2*np.pi/lam

		R = (x**2 + y**2)**0.5

		ar = -vlsr**2/R*kmtokpc**2 - alpha*amp*k*np.cos(k*R + phi)*(amp*np.sin(k*R+phi) + z)
		az = -alpha*(amp*np.sin(k*R+phi)+z)

		ax = ar*x/R
		ay = ar*y/R

		return ax,ay,az

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
		self._finish_init_model(**kwargs)

	def accel(self, x, y, z):
		sigma, z0, b = self.log_corr_params()

		R = (x**2 + y**2)**0.5

		ar = -vlsr**2/R*kmtokpc**2
		az = -(sigma*kmtokpc)**2/b * np.tanh((z-z0)/b)

		ax = ar*x/R
		ay = ar*y/R

		return ax,ay,az

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

		return ax,ay,az

#---------------------------------------------------------------------------
# (Outdated) PTA Expansion
# see Damour & Taylor (1991), Nice & Taylor (1995), Holmberg & Flynn (2004), Lazaridis et al. (2009)
#---------------------------------------------------------------------------
class DamourTaylorPotential(Model):

	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Damour-Taylor Potential'
		self.param_names = []
		self._finish_init_model(**kwargs)

	def accel(self, x, y, z):

		R = (x**2 + y**2)**0.5 #galactocentric

		ar = -(vlsr*kmtokpc)**2/R
		az = -np.sign(z)*(2.27*np.abs(z) + 3.68*(1 - np.exp(-4.31*np.abs(z))))*3.241e-31 #to kpc/s^2 in the weird expansion units, by default this gives 1e-9 cm/s^2

		ax = ar*x/R
		ay = ar*y/R

		return ax,ay,az

#---------------------------------------------------------------------------
# Spherical 
# see Phinney (1993)
#---------------------------------------------------------------------------
class SphericalFlatRC(Model):

	def __init__(self, **kwargs):
		super().__init__()
		self.name = 'Spherical Potential with Flat Rotation Curve'
		self.param_names = []
		self._finish_init_model(**kwargs)

	def accel(self, x, y, z):

		r = (x**2 + y**2 + z**2)**0.5 #galactocentric
		l = np.arctan2(y, -x)
		b = np.arcsin(z/r)

		ar = -(vlsr*kmtokpc)**2/r

		ax = -ar*np.cos(l)*np.cos(b)
		ay = ar*np.sin(l)*np.cos(b)
		az = ar*np.sin(b)

		return ax,ay,az

#-------------------------------------------------
# Generic Gala Potential Wrapper
# allows us to easily alos for any Gala potential
# TODO: make this more homogenous
#       i.e. act the same as the other models
# CANNOT UPDATE PARAMS RIGHT NOW
#-------------------------------------------------
class GalaPotential(Model):

	#pot: a (instantiated) gala potential object
	def __init__(self, pot, **kwargs):
		super().__init__()
		self.name = 'Gala Potential Instance'
		self.param_names = []
		self._finish_init_model(**kwargs)
		self.pot = pot

	def accel(self, x, y, z): 

		a = self.pot.acceleration(np.array([x,y,z])*u.kpc).to(u.kpc/u.s**2).value
		
		#have to do this to homogenize output to the correct type/shape
		if isinstance(x, float): #fails if x is an int (but it shouldn't ever be...? I think?)
			return a[0][0], a[1][0], a[2][0]
		else:
			return a[0], a[1], a[2]

		return ax,ay,az

#--------------------------

#-------------------------------------------------
# Generic Galpy Potential Wrapper
# allows us to easily alos for any Galpy potential
# TODO: CANNOT UPDATE PARAMS RIGHT NOW
#-------------------------------------------------
class GalpyPotential(Model):

	#pot: a (instantiated) gala potential object
	def __init__(self, pot, **kwargs):
		super().__init__()
		self.name = 'Galpy Potential Instance'
		self.param_names = []
		self._finish_init_model(**kwargs)
		self.pot = pot

	def accel(self, x, y, z): 

		R = (x**2 + y**2)**0.5

		az = (evaluatezforces(self.pot, R*u.kpc, z*u.kpc, ro=Rsun*u.kpc, vo=vlsr*u.km/u.s)*(u.km/u.s/u.Myr)).to(u.kpc/u.s**2).value #convert from galpy coords
		ar = (evaluateRforces(self.pot, R*u.kpc, z*u.kpc, ro=Rsun*u.kpc, vo=vlsr*u.km/u.s)*(u.km/u.s/u.Myr)).to(u.kpc/u.s**2).value

		ax = ar*x/R
		ay = ar*y/R

		return ax,ay,az

#--------------------------