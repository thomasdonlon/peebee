#!/usr/bin/env python3
"""
Comprehensive optimizer validation test for peebee.

Tests the optimizer's ability to recover known NFW + Miyamoto-Nagai disk parameters
from mock line-of-sight acceleration data.

Requirements:
- Parameter recovery within 1% tolerance
- Test uses gradient descent optimization
- Both masses optimized in log space, scale lengths in linear space
- Mock data: 100 points within 3 kpc, NFW (1e12 Msun, 30 kpc) + MND (5e10 Msun, 3.5 kpc, 0.3 kpc)
"""

import numpy as np
import sys
import os

# Add parent directory to path to import peebee
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from peebee.models import NFW, MiyamotoNagaiDisk, CompositeModel
from peebee.optimize import Fitter
from peebee.sampling import sample_sky_uniform
from peebee.glob import kpctocm


def create_true_model():
    """Create the true model with known parameters for data generation."""
    # True parameters
    nfw = NFW(m_vir=1e12, r_s=30.0, q=1.0)  # Msun, kpc, dimensionless
    mnd = MiyamotoNagaiDisk(m_tot=5e10, a=3.5, b=0.3)  # Msun, kpc, kpc
    
    # Create composite model
    true_model = CompositeModel()
    true_model.add_model(nfw, 'NFW')
    true_model.add_model(mnd, 'MND')
    
    return true_model


def create_guess_model():
    """Create model with starting guess parameters (20% off from true values)."""
    # Starting guess parameters (20% off from true values)
    nfw_guess = NFW(m_vir=1.2e12, r_s=24.0, q=1.0)  # 20% higher mass, 20% lower r_s
    mnd_guess = MiyamotoNagaiDisk(m_tot=4e10, a=4.2, b=0.24)  # 20% lower mass, 20% higher a, 20% lower b
    
    # Create composite model
    guess_model = CompositeModel()
    guess_model.add_model(nfw_guess, 'NFW')
    guess_model.add_model(mnd_guess, 'MND')
    
    # Configure log parameters for masses
    guess_model.toggle_log_params(['NFW.m_vir', 'MND.m_tot'])
    
    return guess_model


def generate_mock_data(true_model, n_points=100, max_distance=3.0, noise_level=1e-10):
    """Generate mock line-of-sight acceleration data."""
    print(f"Generating {n_points} mock data points within {max_distance} kpc...")
    
    # Sample random sky positions and distances
    l, b, d, alos_true = sample_sky_uniform(true_model, n_points, max_distance)
    
    # Convert to cm/s^2 for observations (peebee standard)
    alos_obs = alos_true * kpctocm
    alos_err = np.ones_like(alos_obs) * noise_level  # Very small errors for clean test
    
    return l, b, d, alos_obs, alos_err


def setup_optimization(model, l, b, d, alos_obs, alos_err):
    """Set up the Fitter for optimization."""
    print("Setting up optimization...")
    
    fitter = Fitter()
    fitter.set_model(model)
    fitter.set_data(l, b, d, alos_obs, alos_err)
    
    # Define parameter bounds
    param_bounds = {
        'NFW.m_vir': (10.0, 14.0),      # log10(mass) bounds
        'NFW.r_s': (5.0, 100.0),        # linear scale length bounds
        'MND.m_tot': (9.0, 12.0),       # log10(mass) bounds  
        'MND.a': (1.0, 10.0),           # linear scale length bounds
        'MND.b': (0.1, 1.0)             # linear scale length bounds
    }
    
    fitter.configure_params(param_bounds)
    
    return fitter


def run_optimization(fitter):
    """Run the gradient descent optimization."""
    print("Running gradient descent optimization...")
    
    # Run optimization with gradient descent  
    result = fitter.optimize(method='gradient_descent')
    
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    
    print(f"Optimization converged successfully!")
    print(f"Reduced chi-squared: {result.reduced_chi2:.6f}")
    
    return result


def validate_results(result, tolerance=0.01):
    """Validate that recovered parameters match true values within tolerance."""
    print("Validating parameter recovery...")
    
    # True parameters (in optimization space - log for masses, linear for lengths)
    true_params = {
        'NFW.m_vir': np.log10(1e12),    # log10(Msun)
        'NFW.r_s': 30.0,                # kpc
        'MND.m_tot': np.log10(5e10),    # log10(Msun)
        'MND.a': 3.5,                   # kpc
        'MND.b': 0.3                    # kpc
    }
    
    print("\nParameter Recovery Assessment:")
    print("="*50)
    print(f"{'Parameter':<12} {'True':<12} {'Recovered':<12} {'Error %':<10} {'Status'}")
    print("-"*60)
    
    all_passed = True
    max_error = 0.0
    
    for param_name, true_value in true_params.items():
        recovered_value = result.best_fit_params[param_name]
        
        # Calculate relative error
        rel_error = abs(recovered_value - true_value) / abs(true_value)
        error_percent = rel_error * 100
        max_error = max(max_error, rel_error)
        
        # Check if within tolerance
        status = "PASS" if rel_error < tolerance else "FAIL"
        if rel_error >= tolerance:
            all_passed = False
            
        print(f"{param_name:<12} {true_value:<12.6f} {recovered_value:<12.6f} {error_percent:<10.3f} {status}")
    
    print("-"*60)
    print(f"Maximum relative error: {max_error*100:.3f}%")
    print(f"Required tolerance: {tolerance*100:.1f}%")
    
    if all_passed:
        print("\n✅ SUCCESS: All parameters recovered within tolerance!")
        return True
    else:
        print(f"\n❌ FAILURE: Some parameters exceed {tolerance*100:.1f}% tolerance!")
        return False


def display_physical_parameters(result):
    """Display the recovered parameters in physical units."""
    print("\nRecovered Physical Parameters:")
    print("="*40)
    
    # Convert log parameters back to linear
    nfw_mass = 10**result.best_fit_params['NFW.m_vir']
    nfw_rs = result.best_fit_params['NFW.r_s']
    mnd_mass = 10**result.best_fit_params['MND.m_tot']
    mnd_a = result.best_fit_params['MND.a']
    mnd_b = result.best_fit_params['MND.b']
    
    print(f"NFW Halo:")
    print(f"  Mass: {nfw_mass:.2e} M☉")
    print(f"  Scale radius: {nfw_rs:.2f} kpc")
    print(f"")
    print(f"Miyamoto-Nagai Disk:")
    print(f"  Mass: {mnd_mass:.2e} M☉")  
    print(f"  Scale length a: {mnd_a:.2f} kpc")
    print(f"  Scale height b: {mnd_b:.3f} kpc")


def test_optimizer_physics():
    """Main test function for optimizer physics validation."""
    print("Peebee Optimizer Physics Validation Test")
    print("="*45)
    print("Testing NFW + Miyamoto-Nagai parameter recovery")
    print("Requirements: 1% tolerance, gradient descent optimization")
    print()
    
    try:
        # 1. Create true model for data generation
        true_model = create_true_model()
        
        # 2. Generate mock data
        l, b, d, alos_obs, alos_err = generate_mock_data(true_model)
        
        # 3. Create guess model with wrong starting parameters
        guess_model = create_guess_model()
        
        # 4. Set up optimization
        fitter = setup_optimization(guess_model, l, b, d, alos_obs, alos_err)
        
        # 5. Run optimization  
        result = run_optimization(fitter)
        
        # 6. Display recovered parameters
        display_physical_parameters(result)
        
        # 7. Validate results
        success = validate_results(result, tolerance=0.01)
        
        if success:
            print(f"\n🎉 TEST PASSED: Optimizer successfully recovered all parameters!")
            return 0
        else:
            print(f"\n💥 TEST FAILED: Parameter recovery exceeded tolerance limits!")
            return 1
            
    except Exception as e:
        print(f"\n💥 TEST ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = test_optimizer_physics()
    sys.exit(exit_code)