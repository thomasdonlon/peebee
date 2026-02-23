#!/usr/bin/env python3
"""
Example demonstrating the new NoiseModel system with the Fitter class.

This shows how to:
1. Create noise model instances
2. Set them in the Fitter
3. Configure noise parameters alongside potential model parameters
4. Run optimization with noise models
"""

import numpy as np
import sys
import os

# Add peebee to path (for example purposes)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from peebee.models import NFW
from peebee.noise import GaussianNoise, LorentzNoise, PowerLawNoise
from peebee.optimize import Fitter
from peebee.sampling import perturb_value

# Generate some synthetic data
np.random.seed(42)
n_pulsars = 10
l = np.random.uniform(0, 360, n_pulsars)  # Galactic longitude (deg)
b = np.random.uniform(-90, 90, n_pulsars)  # Galactic latitude (deg)
d = np.random.uniform(0.5, 3.0, n_pulsars)  # Distance (kpc)

# Create true model
true_model = NFW(m_vir=1e12, r_s=20.0)
true_alos = true_model.alos(l, b, d)

# Add noise and measurement uncertainties
alos_err = np.full_like(true_alos, 1 * np.mean(np.abs(true_alos)))  # 10% mean relative uncertainty
alos_obs = true_alos + np.random.normal(0, alos_err)  # Noisy observations

print("=== New NoiseModel System Demo ===")
print(f"Generated {n_pulsars} synthetic pulsars")
print(f"True parameters: m_vir=1e12, r_s=20.0")

# Example 1: No noise model (traditional chi-squared fitting)
print("\n--- Example 1: No noise model ---")
fitter = Fitter()
fit_model = NFW(m_vir=5e11, r_s=25.0) #initialize the model
fit_model.toggle_log_params('m_vir')
fitter.set_model(fit_model) 

fitter.set_data(l, b, d, alos_obs, alos_err)
fitter.configure_params({
    "m_vir": (10, 14),
    "r_s": (5, 50)
})

result = fitter.optimize(method='gd')
print(f"Result: {result}")
if result.success:
    print(f"Best fit: m_vir={result.best_fit_params['m_vir']:.2e}, r_s={result.best_fit_params['r_s']:.1f}")

# Example 2: Gaussian noise model
print("\n--- Example 2: Gaussian noise model ---")
fitter = Fitter()
fit_model = NFW(m_vir=5e11, r_s=25.0) #initialize the model
fit_model.toggle_log_params('m_vir')
fitter.set_model(fit_model) 

fitter.set_data(l, b, d, alos_obs, alos_err) 

# Create and set noise model
gaussian_noise = GaussianNoise(sigma=1.0)  # Initial value
fitter.set_noise_model(gaussian_noise)

# Configure parameters including noise parameter
fitter.configure_params({
    "m_vir": (10, 14),
    "r_s": (5, 50),
    "noise.sigma": (0.1, 10.0)  # Noise parameter with qualified name
})

result = fitter.optimize(method='gd')
print(f"Result: {result}")
if result.success:
    print(f"Best fit: m_vir={result.best_fit_params['m_vir']:.2e}, r_s={result.best_fit_params['r_s']:.1f}")
    print(f"Noise parameter: sigma={result.best_fit_params['noise.sigma']:.3f}")

# Example 3: Lorentz noise model
print("\n--- Example 3: Lorentz noise model ---")
fitter = Fitter()
fit_model = NFW(m_vir=5e11, r_s=25.0) #initialize the model
fit_model.toggle_log_params('m_vir')
fitter.set_model(fit_model) 

fitter.set_data(l, b, d, alos_obs, alos_err)

# Create and set different noise model
lorentz_noise = LorentzNoise(gamma=1.0)
fitter.set_noise_model(lorentz_noise)

fitter.configure_params({
    "m_vir": (10, 14),
    "r_s": (5, 50),
    "noise.gamma": (0.1, 5.0)  # Different noise parameter name
})

result = fitter.optimize(method='gd')
print(f"Result: {result}")
if result.success:
    print(f"Best fit: m_vir={result.best_fit_params['m_vir']:.2e}, r_s={result.best_fit_params['r_s']:.1f}")
    print(f"Noise parameter: gamma={result.best_fit_params['noise.gamma']:.3f}")

# Example 4: Demonstrate noise model interface directly
print("\n--- Example 4: Direct noise model usage ---")
residuals = np.abs(alos_obs - true_alos)

# Test different noise models on the same residuals
models_to_test = [
    GaussianNoise(sigma=0.5),
    LorentzNoise(gamma=0.5),
    PowerLawNoise(zeta=2.5)
]

for noise_model in models_to_test:
    likelihood_val = noise_model.likelihood(residuals)
    print(f"{noise_model.name:10s} likelihood: {likelihood_val:.3f}")
    print(f"           parameters: {noise_model.get_params()}")

print("\n=== Demo complete ===")