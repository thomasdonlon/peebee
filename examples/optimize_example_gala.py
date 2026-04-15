# Example file that demonstrates how to use the optimizer module to optimize a simple potential model to mock data.
# This file shows that GalaPotential models are also optimizable
# Note that we generate the mock data from a known potential model (with some noise), so we know that the optimizer should be able to find the correct parameters fairly accurately.
# In a real use case, you would replace the mock data with your own data, and you may need to adjust the potential model and optimization settings accordingly.

import numpy as np
from peebee import optimize, sampling, models
from peebee.noise import GaussianNoise
from gala.potential import MilkyWayPotential2022

#------------------------------
# Generate mock data
#------------------------------

print('Generating mock data...')

# Define a simple Galactic potential model (disk + halo)
model = models.GalaPotential(MilkyWayPotential2022())

# turn on log scaling for the masses to avoid numerical issues during optimization
# (if you don't do this, the optimizer will have a hard time finding the right parameters because the mass can vary over many orders of magnitude)
model.toggle_log_params(['halo.m', 'disk.m'])
original_params = model.params

#fix the names to match the names used in the optimization below
correct_params = {}
for name in original_params.keys():
    if name == 'halo.m':
        correct_params['NFWPotential.m'] = original_params[name]
    elif name == 'halo.r_s':
        correct_params['NFWPotential.r_s'] = original_params[name]
    elif name == 'disk.m':
        correct_params['MN3ExponentialDiskPotential.m'] = original_params[name]
    elif name == 'disk.h_R':
        correct_params['MN3ExponentialDiskPotential.h_R'] = original_params[name]
    elif name == 'disk.h_z':
        correct_params['MN3ExponentialDiskPotential.h_z'] = original_params[name]

# Generate mock data: positions and velocities of 30 pulsars in the potential, with some noise
# distributed uniformly within a box with 5 kpc sides centered on the Sun
num_pulsars = 50
bounds = np.array([[5.5, 9.5], [-2.5, 2.5], [-2.5, 2.5]])  # bounds in kpc
x_mock, y_mock, z_mock, alos_mock = sampling.sample_alos_sources_uniform(model, num_pulsars, bounds) #alos is in kpc/s^2 here

# Add some noise to the alos values (10% relative uncertainty)
alos_mock_noise = 0.05 * np.abs(alos_mock)  # 10% relative uncertainty
alos_mock_noisy = sampling.perturb_value(alos_mock, value_unc=0.05, noise_model='gaussian', relative_err=True)

print('Done generating mock data.\n')

#------------------------------
# Test 1
#------------------------------

print('\nOptimizing all parameters...')

#only going to fit the disk and halo components of the potential to simplify the optimization, but you could also add the other components if you wanted to
model = models.GalaPotential(MilkyWayPotential2022()['halo']) + models.GalaPotential(MilkyWayPotential2022()['disk'])

#vary the parameters a little bit so that we actually have something to optimize
model.set_params({'NFWPotential.m': 4e11, 'NFWPotential.r_s': 17., 'MN3ExponentialDiskPotential.m': 3e10, 'MN3ExponentialDiskPotential.h_R': 3.0, 'MN3ExponentialDiskPotential.h_z': 0.4})
model.toggle_log_params(['NFWPotential.m', 'MN3ExponentialDiskPotential.m'])

# Optimize all parameters simultaneously (disk + halo)
fitter_all = optimize.Fitter()
fitter_all.set_model(model)
fitter_all.set_data(x_mock, y_mock, z_mock, alos_mock_noisy, alos_mock_noise, frame='cart')

#fit a noise model
gaussian_noise = GaussianNoise(sigma=1.0)  # Initial value
fitter_all.set_noise_model(gaussian_noise)

# Define all parameters to be optimized
param_names_all = ['NFWPotential.m', 'NFWPotential.r_s', 'MN3ExponentialDiskPotential.m', 'MN3ExponentialDiskPotential.h_R', 'MN3ExponentialDiskPotential.h_z', 'noise.sigma']
param_bounds_all = [(9., 12.), (1., 100.), (8., 12.), (0.01, 10.), (0.01, 10.), (0.01, 10.)]  # bounds in kpc and log10 solar masses

# Build the parameter dictionary for all parameters
param_dict_all = {}
for name, bounds in zip(param_names_all, param_bounds_all):
    param_dict_all[name] = bounds

fitter_all.configure_params(param_dict_all)
fitter_all.optimize(method='gradient_descent')

# Print the optimized parameters for all
results = fitter_all.results
print(results)
print("Optimized parameters (all):")
for name, value in results.best_fit_params.items():
    if name.split('.')[0] != 'noise':
        print(f"{name}: {value:.3f}±{results.uncertainties[name]:.3f}, correct value: {correct_params[name]:.3f}")

print('Done optimizing all parameters.\n')
