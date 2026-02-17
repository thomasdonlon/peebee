# Example file that demonstrates how to use the optimizer module to optimize a simple potential model to mock data.
# Note that we generate the mock data from a known potential model (with some noise), so we know that the optimizer should be able to find the correct parameters fairly accurately.
# In a real use case, you would replace the mock data with your own data, and you may need to adjust the potential model and optimization settings accordingly.

import numpy as np
import matplotlib.pyplot as plt
from peebee import optimize, models, sampling

#------------------------------
# Generate mock data
#------------------------------

print('Generating mock data...')

# Define a simple Galactic potential model (disk + halo)
model = models.NFW(m_vir=1e12, r_s=30.0) + models.MiyamotoNagaiDisk(m_tot=5e10, a=3.0, b=0.3)

# turn on log scaling for the masses to avoid numerical issues during optimization
# (if you don't do this, the optimizer will have a hard time finding the right parameters because the mass can vary over many orders of magnitude)
model.toggle_log_params(['NFW.m_vir', 'Miyamoto-Nagai Disk.m_tot'])

# Generate mock data: positions and velocities of 30 pulsars in the potential, with some noise
# distributed uniformly within a box with 5 kpc sides centered on the Sun
num_pulsars = 30
bounds = np.array([[5.5, 9.5], [-2.5, 2.5], [-2.5, 2.5]])  # bounds in kpc
x_mock, y_mock, z_mock, alos_mock = sampling.sample_alos_sources_uniform(model, num_pulsars, bounds) #alos is in kpc/s^2 here

# Add some noise to the alos values (10% relative uncertainty)
alos_mock_noise = 0.1 * np.abs(alos_mock)  # 10% relative uncertainty
alos_mock_noisy = sampling.perturb_value(alos_mock, value_unc=0.1, noise_model='gaussian', relative_err=True)

print('Done generating mock data.\n')

#------------------------------
# Test 1
#------------------------------
# Now we have our mock data (x_mock, y_mock, z_mock, alos_mock_noisy) that we want to fit with an optimized potential model.
# We will try to recover the parameters of the disk (scale_length, height, mass) while keeping the halo parameters fixed.

print('Optimizing disk parameters...')
correct_disk_params = [np.log10(5e10), 3.0, 0.3]  # log10(m_tot), a, b

# Construct the model we'll be fitting to the data (slightly off from the true model so that we have something to optimize)
# this is our guess for the underlying potential model for the gradient descent method
model = models.NFW(m_vir=1e12, r_s=30.0) + models.MiyamotoNagaiDisk(m_tot=4e10, a=2.7, b=0.2)
model.toggle_log_params(['NFW.m_vir', 'Miyamoto-Nagai Disk.m_tot'])

# Define the parameters to be optimized (the disk parameters)
param_names = ['m_tot', 'a', 'b']
param_bounds = [(9., 12.), (0.1, 10.0), (0.01, 1.0)]  # bounds in kpc and log10 solar masses

# Create an optimizer object and set the relevant information
fitter = optimize.Fitter()
fitter.set_model(model)
fitter.set_data(x_mock, y_mock, z_mock, alos_mock_noisy, alos_mock_noise, frame='cart')
fitter.set_noise_model('gaussian')

#build the parameter dictionary that we will pass to the optimizer (this is just a mapping from the parameter names to the actual parameters in the model)
param_dict = {}
for name, bounds in zip(param_names, param_bounds):
    param_dict[f"Miyamoto-Nagai Disk.{name}"] = bounds

fitter.configure_params(param_dict)

# Optimize the disk parameters
fitter.optimize(method='gradient_descent')

# Print the optimized parameters
results = fitter.results
print(results)
print("Optimized parameters (disk):")
for name, value in results.best_fit_params.items():
    print(f"{name}: {value:.3f}, correct value: {correct_disk_params[param_names.index(name.split('.')[1])]:.3f}")

print('Done optimizing disk parameters.\n')

#------------------------------
# Test 2
#------------------------------

print('\nOptimizing all parameters...')
correct_params = [np.log10(5e10), 3.0, 0.3, 12.0, 30.]  # log10(m_tot), a, b, log10(m_vir), r_s

# Construct the model we'll be fitting to the data (slightly off from the true model so that we have something to optimize)
# this is our guess for the underlying potential model for the gradient descent method
model = models.NFW(m_vir=9e11, r_s=27.0) + models.MiyamotoNagaiDisk(m_tot=4e10, a=2.7, b=0.2)
model.toggle_log_params(['NFW.m_vir', 'Miyamoto-Nagai Disk.m_tot'])

# Optimize all parameters simultaneously (disk + halo)
fitter_all = optimize.Fitter()
fitter_all.set_model(model)
fitter_all.set_data(x_mock, y_mock, z_mock, alos_mock_noisy, alos_mock_noise, frame='cart')
fitter_all.set_noise_model('gaussian')

# Define all parameters to be optimized
param_names_all = ['m_tot', 'a', 'b', 'm_vir', 'r_s']
param_bounds_all = [(9., 12.), (0.1, 10.0), (0.01, 1.0), (10., 14.), (5., 50.)]

# Build the parameter dictionary for all parameters
param_dict_all = {}
for name, bounds in zip(param_names_all, param_bounds_all):
    if name in ['m_tot', 'a', 'b']:
        param_dict_all[f"Miyamoto-Nagai Disk.{name}"] = bounds
    elif name in ['m_vir', 'r_s']:
        param_dict_all[f"NFW.{name}"] = bounds

fitter_all.configure_params(param_dict_all)
fitter_all.optimize(method='gradient_descent')

# Print the optimized parameters for all
results = fitter_all.results
print(results)
print("Optimized parameters (all):")
for name, value in results.best_fit_params.items():
    print(f"{name}: {value:.3f}, correct value: {correct_params[param_names_all.index(name.split('.')[1])]:.3f}")

print('Done optimizing all parameters.\n')

#------------------------------
# Test 3
#------------------------------

#Note that in Test 2, the optimization really struggles to fit the halo parameters. 
# This has to do with the fact that basically the pulsar data near the Sun only constrains 
# the acceleration at the Solar position (effectively the enclosed mass within the Solar radius),
# since the variation of the halo acceleration is small over the volume spanned by the data.
# This gives 1 effective constraint, but we are fitting two parameters. 
# If you fit the halo mass but keep the scale radius fixed, you can get a much better fit to the halo parameters,
# even with a somewhat inaccurate guess of the scale radius (here we are purposely off by 10%).

print('\nOptimizing parameters, keeping halo scale radius fixed...')
correct_params = [np.log10(5e10), 3.0, 0.3, 12.0]  # log10(m_tot), a, b, log10(m_vir)

model = models.NFW(m_vir=9e11, r_s=27.0) + models.MiyamotoNagaiDisk(m_tot=4e10, a=2.7, b=0.2)
model.toggle_log_params(['NFW.m_vir', 'Miyamoto-Nagai Disk.m_tot'])

# Optimize all parameters simultaneously (disk + halo)
fitter_all = optimize.Fitter()
fitter_all.set_model(model)
fitter_all.set_data(x_mock, y_mock, z_mock, alos_mock_noisy, alos_mock_noise, frame='cart')
fitter_all.set_noise_model('gaussian')

# Define all parameters to be optimized
param_names_all = ['m_tot', 'a', 'b', 'm_vir']
param_bounds_all = [(9., 12.), (0.1, 10.0), (0.01, 1.0), (10., 14.)]

# Build the parameter dictionary for all parameters
param_dict_all = {}
for name, bounds in zip(param_names_all, param_bounds_all):
    if name in ['m_tot', 'a', 'b']:
        param_dict_all[f"Miyamoto-Nagai Disk.{name}"] = bounds
    elif name in ['m_vir']:
        param_dict_all[f"NFW.{name}"] = bounds

fitter_all.configure_params(param_dict_all)
init_guess_all = (10.5, 2.0, 0.5, 12.0)  # initial guess for all parameters (log10(m_tot), a, b, log10(m_vir), r_s)
fitter_all.optimize(method='gradient_descent')  # initial guess for all parameters

# Print the optimized parameters for all
results = fitter_all.results
print(results)
print("Optimized parameters (all):")
for name, value in results.best_fit_params.items():
    print(f"{name}: {value:.3f}, correct value: {correct_params[param_names_all.index(name.split('.')[1])]:.3f}")

print('Done optimizing parameters.')