Constructing Noise Models
==========================

This tutorial covers peebee's noise model framework for handling realistic observational uncertainties and outliers in pulsar acceleration data. We'll explore different noise distributions and show how to configure robust fitting procedures.

Overview
--------

Real astronomical data rarely follows perfect Gaussian statistics. Systematic errors, calibration issues, and unexpected astrophysical effects can create outliers and non-Gaussian noise distributions. Peebee's noise models provide robust alternatives to standard χ² fitting.

Available noise models:

- **GaussianNoise**: Standard Gaussian likelihood (traditional χ² fitting)
- **LorentzNoise**: Cauchy/Lorentzian distribution (heavy-tailed, robust to outliers)  
- **PowerLawNoise**: Power-law likelihood for extreme outlier scenarios

Understanding Noise Models
--------------------------

Each noise model implements a different likelihood function for the residuals.

Gaussian Noise Model
~~~~~~~~~~~~~~~~~~~~

The standard approach assumes Gaussian-distributed measurement errors:

.. code-block:: python

    import numpy as np
    from peebee.noise import GaussianNoise
    from peebee import optimize, models
    from peebee.models import NFW, MiyamotoNagaiDisk
    
    # Create synthetic data with Gaussian noise
    np.random.seed(42)
    
    # True model
    true_model = NFW(m_vir=1e12, r_s=20.0) + MiyamotoNagaiDisk(m_tot=5e10, a=3.0, b=0.3)
    
    # Generate clean accelerations
    l_test = np.random.uniform(0, 360, 30)
    b_test = np.random.uniform(-30, 30, 30)
    d_test = np.random.uniform(0.5, 2.5, 30)
    alos_clean = true_model.alos(l_test, b_test, d_test)
    
    # Add Gaussian noise
    uncertainties = 0.15 * np.ones_like(alos_clean)  # 0.15 mm/s/yr uncertainty
    alos_noisy = alos_clean + np.random.normal(0, uncertainties)
    
    print("Gaussian noise example:")
    print(f"Data points: {len(alos_noisy)}")
    print(f"RMS residual: {np.std(alos_noisy - alos_clean):.3f} mm/s/yr")
    print(f"Expected RMS: {np.mean(uncertainties):.3f} mm/s/yr")

Create and configure a Gaussian noise model:

.. code-block:: python

    # Initialize Gaussian noise model
    gaussian_noise = GaussianNoise(sigma=1.0)
    
    print(f"Gaussian noise model parameters: {gaussian_noise.params}")
    print(f"Parameter names: {gaussian_noise.param_names}")
    
    # Test likelihood calculation
    test_residuals = np.array([0.0, 1.0, -1.0, 2.0])
    likelihood = gaussian_noise.likelihood(test_residuals)
    print(f"Negative log-likelihood: {likelihood:.3f}")

Lorentzian Noise Model
~~~~~~~~~~~~~~~~~~~~~~

The Lorentzian (Cauchy) distribution has heavy tails, making it robust to outliers:

.. code-block:: python

    from peebee.noise import LorentzNoise
    
    # Create data with outliers
    alos_outliers = alos_clean + np.random.normal(0, uncertainties)
    
    # Add some extreme outliers (5% of data)
    n_outliers = max(1, len(alos_outliers) // 20)
    outlier_indices = np.random.choice(len(alos_outliers), n_outliers, replace=False)
    alos_outliers[outlier_indices] += np.random.normal(0, 10*uncertainties[outlier_indices])
    
    print(f"\nAdded {n_outliers} extreme outliers")
    print(f"Outlier amplitudes: {alos_outliers[outlier_indices] - alos_clean[outlier_indices]}")
    print(f"RMS with outliers: {np.std(alos_outliers - alos_clean):.3f} mm/s/yr")
    
    # Initialize Lorentzian noise model
    lorentz_noise = LorentzNoise(gamma=1.0)
    
    print(f"Lorentzian noise model parameters: {lorentz_noise.params}")

Power-Law Noise Model  
~~~~~~~~~~~~~~~~~~~~~

For extreme cases with very heavy tails:

.. code-block:: python

    from peebee.noise import PowerLawNoise
    
    # Initialize power-law noise model
    power_law_noise = PowerLawNoise(zeta=1.5)
    
    print(f"Power-law noise parameter ζ = {power_law_noise.params['zeta']}")
    
    # Compare likelihood behavior for different noise models
    test_residual = 5.0  # Large outlier
    
    gauss_ll = gaussian_noise.likelihood(np.array([test_residual]))
    lorentz_ll = lorentz_noise.likelihood(np.array([test_residual]))
    power_ll = power_law_noise.likelihood(np.array([test_residual]))
    
    print(f"\nLikelihood for residual = {test_residual}:")
    print(f"  Gaussian:   {gauss_ll:.3f}")
    print(f"  Lorentzian: {lorentz_ll:.3f}")
    print(f"  Power-law:  {power_ll:.3f}")
    print("  (Lower = more likely; Lorentzian/Power-law are more tolerant)")

Configuring Noise Models in Fitting
-----------------------------------

Let's compare how different noise models handle the same dataset.

Setting Up the Test
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create a model to fit
    fit_model = NFW(m_vir=8e11, r_s=15.0) + MiyamotoNagaiDisk(m_tot=4e10, a=2.5, b=0.25)
    
    # Common fitter configuration
    def setup_fitter(noise_model, data_alos):
        fitter = optimize.Fitter()
        fitter.set_model(fit_model)
        fitter.set_data(l_test, b_test, d_test, data_alos, uncertainties)
        fitter.set_noise_model(noise_model)
        
        # Configure parameters
        fitter.configure_params({
            'NFW.m_vir': (5e11, 2e12),
            'NFW.r_s': (5.0, 40.0),
            'Miyamoto-Nagai Disk.m_tot': (1e10, 1e11),
            'Miyamoto-Nagai Disk.a': (1.0, 5.0),
            'Miyamoto-Nagai Disk.b': (0.1, 1.0)
        })
        
        return fitter
    
    print("Setting up noise model comparison...")

Gaussian Fitting
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Fit with Gaussian noise model
    gaussian_noise = GaussianNoise(sigma=1.0)
    gaussian_fitter = setup_fitter(gaussian_noise, alos_outliers)
    
    # Add noise parameter to optimization
    gaussian_fitter.configure_params({'noise.sigma': (0.1, 10.0)})
    
    print("Fitting with Gaussian noise model...")
    gaussian_results = gaussian_fitter.optimize(method='gradient_descent')
    
    print(f"Gaussian fit results:")
    print(f"  Reduced χ²: {gaussian_results.reduced_chi2:.3f}")
    print(f"  AIC: {gaussian_results.aic:.1f}")
    print(f"  Noise σ: {gaussian_results.best_fit_params['noise.sigma']:.3f}")

Lorentzian Fitting  
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Fit with Lorentzian noise model
    lorentz_noise = LorentzNoise(gamma=1.0)
    lorentz_fitter = setup_fitter(lorentz_noise, alos_outliers)
    
    # Add noise parameter to optimization
    lorentz_fitter.configure_params({'noise.gamma': (0.1, 10.0)})
    
    print("Fitting with Lorentzian noise model...")
    lorentz_results = lorentz_fitter.optimize(method='gradient_descent')
    
    print(f"Lorentzian fit results:")
    print(f"  Reduced χ²: {lorentz_results.reduced_chi2:.3f}")
    print(f"  AIC: {lorentz_results.aic:.1f}")
    print(f"  Noise γ: {lorentz_results.best_fit_params['noise.gamma']:.3f}")

Parameter Recovery Analysis
---------------------------

Compare how each noise model handles outliers:

.. code-block:: python

    # Compare parameter recovery
    true_params = true_model.params
    
    results_list = [
        ("Gaussian", gaussian_results),
        ("Lorentzian", lorentz_results)
    ]
    
    print("\nParameter recovery comparison:")
    print("Parameter                   True Value    Gaussian    Lorentzian  Recovery Diff")
    print("-" * 80)
    
    for param in ['NFW.m_vir', 'NFW.r_s', 'Miyamoto-Nagai Disk.m_tot']:
        true_val = true_params[param]
        
        gauss_val = gaussian_results.best_fit_params[param]
        lorentz_val = lorentz_results.best_fit_params[param]
        
        gauss_recovery = (gauss_val - true_val) / true_val * 100
        lorentz_recovery = (lorentz_val - true_val) / true_val * 100
        
        print(f"{param:25s} {true_val:10.2e} {gauss_val:10.2e} {lorentz_val:11.2e} "
              f"{lorentz_recovery - gauss_recovery:+9.1f}%")
    
    # Residual analysis
    fit_model.set_params(gaussian_results.best_fit_params)
    residuals_gaussian = alos_outliers - fit_model.alos(l_test, b_test, d_test)
    
    fit_model.set_params(lorentz_results.best_fit_params) 
    residuals_lorentz = alos_outliers - fit_model.alos(l_test, b_test, d_test)
    
    print(f"\nResidual analysis:")
    print(f"  Gaussian RMS:   {np.std(residuals_gaussian):.3f} mm/s/yr")
    print(f"  Lorentzian RMS: {np.std(residuals_lorentz):.3f} mm/s/yr")

Advanced Noise Model Configuration
----------------------------------

Working with Fixed Noise Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes you want to fix noise parameters based on external knowledge:

.. code-block:: python

    # Fix noise parameter based on instrumental knowledge
    fixed_noise = LorentzNoise(gamma=0.2)  # Known instrumental characteristics
    
    fixed_fitter = optimize.Fitter()
    fixed_fitter.set_model(fit_model)
    fixed_fitter.set_data(l_test, b_test, d_test, alos_outliers, uncertainties)
    fixed_fitter.set_noise_model(fixed_noise)
    
    # Don't include noise parameter in optimization
    fixed_fitter.configure_params({
        'NFW.m_vir': (5e11, 2e12),
        'NFW.r_s': (5.0, 40.0),
        'Miyamoto-Nagai Disk.m_tot': (1e10, 1e11)
    })
    
    print("Fitting with fixed noise parameter...")
    fixed_results = fixed_fitter.optimize(method='gradient_descent')
    
    print(f"Fixed noise results:")
    print(f"  Reduced χ²: {fixed_results.reduced_chi2:.3f}")
    print(f"  Fixed γ: {fixed_noise.params['gamma']:.3f}")

Noise Model Parameter Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Update noise model parameters after initialization
    dynamic_noise = GaussianNoise(sigma=1.0)
    print(f"Initial σ: {dynamic_noise.params['sigma']}")
    
    # Update parameter
    dynamic_noise.set_params({'sigma': 2.5})
    print(f"Updated σ: {dynamic_noise.params['sigma']}")
    
    # Get parameter names and current values
    print(f"All parameters: {dynamic_noise.get_params()}")

Model Selection with Information Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Compare models using AIC
    print("\nModel selection:")
    print("Noise Model     AIC    ΔAIC   Best?")
    print("-" * 35)
    
    aics = [gaussian_results.aic, lorentz_results.aic]
    best_aic = min(aics)
    
    for (name, results) in results_list:
        delta_aic = results.aic - best_aic
        is_best = "✓" if results.aic == best_aic else ""
        print(f"{name:12s} {results.aic:7.1f} {delta_aic:6.1f} {is_best:>6s}")
    
    print("\nInterpretation:")
    print("  ΔAIC < 2:  Models are essentially equivalent") 
    print("  ΔAIC 2-7:  Some support for best model")
    print("  ΔAIC > 10: Strong support for best model")

Noise Model Diagnostics
-----------------------

Understanding noise model behavior:

Direct Likelihood Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Calculate likelihoods for the same set of residuals
    test_residuals = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])  # Example residuals
    
    # Set up noise models with similar scales
    gauss = GaussianNoise(sigma=1.0)
    lorentz = LorentzNoise(gamma=1.0)
    
    print("Likelihood comparison for different residuals:")
    print("Residual  Gaussian   Lorentzian   Ratio")
    print("-" * 40)
    
    for res in test_residuals:
        # Calculate negative log-likelihoods
        ll_gauss = gauss.likelihood(np.array([res]))
        ll_lorentz = lorentz.likelihood(np.array([res]))
        ratio = ll_lorentz / ll_gauss if ll_gauss != 0 else np.inf
        
        print(f"{res:8.1f} {ll_gauss:9.3f} {ll_lorentz:11.3f} {ratio:8.3f}")
    
    print("\nInterpretation:")
    print("  Ratio < 1: Lorentzian more tolerant of residual")
    print("  Ratio > 1: Gaussian more tolerant of residual")

Working with Real Data
----------------------

Practical guidelines for real pulsar datasets:

Data Assessment
~~~~~~~~~~~~~~~

.. code-block:: python

    # Example: Assess real data for appropriate noise model choice
    def assess_data_for_noise_model(alos_obs, alos_err):
        """Assess data characteristics to guide noise model selection"""
        
        # Calculate normalized residuals (need a rough model first)
        median_acc = np.median(alos_obs)
        residuals_from_median = alos_obs - median_acc
        normalized = residuals_from_median / alos_err
        
        # Statistical tests
        n_outliers = np.sum(np.abs(normalized) > 3)
        outlier_fraction = n_outliers / len(normalized)
        
        rms = np.std(normalized)
        skewness = np.mean(normalized**3)
        kurtosis = np.mean(normalized**4) - 3  # Excess kurtosis
        
        print(f"Data assessment:")
        print(f"  Sample size: {len(alos_obs)}")
        print(f"  Outliers (>3σ): {n_outliers} ({outlier_fraction*100:.1f}%)")
        print(f"  RMS of normalized residuals: {rms:.2f}")
        print(f"  Skewness: {skewness:+.2f}")
        print(f"  Excess kurtosis: {kurtosis:+.2f}")
        
        # Recommendations
        print(f"\nRecommendations:")
        if outlier_fraction < 0.05 and abs(kurtosis) < 1:
            print("  → Use GaussianNoise (data appears well-behaved)")
        elif outlier_fraction < 0.15:
            print("  → Consider LorentzNoise (some outliers present)")
        else:
            print("  → Use LorentzNoise or PowerLawNoise (many outliers)")
        
        if abs(skewness) > 1:
            print("  → Data may have systematic bias - investigate")
    
    # Test with our outlier-contaminated data
    assess_data_for_noise_model(alos_outliers, uncertainties)

Iterative Outlier Identification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Use robust fitting to identify problematic data points
    def identify_outliers_iterative(fitter, threshold=3.0, max_iterations=3):
        """Iteratively identify outliers using robust fitting"""
        
        original_data = fitter.alos_obs.copy()
        original_err = fitter.alos_err.copy()
        
        for iteration in range(max_iterations):
            # Fit with current data
            results = fitter.optimize(method='gradient_descent')
            
            # Calculate residuals
            fitter.model.set_params(results.best_fit_params)
            predicted = fitter.model.alos(fitter.l, fitter.b, fitter.d)
            residuals = fitter.alos_obs - predicted
            normalized_residuals = residuals / fitter.alos_err
            
            # Identify outliers
            outlier_mask = np.abs(normalized_residuals) > threshold
            n_outliers = np.sum(outlier_mask)
            
            print(f"Iteration {iteration+1}: {n_outliers} outliers found")
            
            if n_outliers == 0:
                print("  No more outliers - converged")
                break
            
            # Remove outliers for next iteration
            keep_mask = ~outlier_mask
            fitter.set_data(
                fitter.l[keep_mask], fitter.b[keep_mask], fitter.d[keep_mask],
                fitter.alos_obs[keep_mask], fitter.alos_err[keep_mask]
            )
            
            print(f"  Removed {n_outliers} points, {len(fitter.alos_obs)} remaining")
        
        return results, outlier_mask
    
    # Example usage (would need a properly set up fitter)
    print("\nIterative outlier removal example:")
    print("This approach removes outliers completely rather than down-weighting them")

Best Practices
--------------

Guidelines for choosing and using noise models:

**Model Selection Strategy:**

1. **Start with data assessment**: Use statistical tests to understand your data
2. **Begin with Gaussian**: For well-understood, high-quality data
3. **Use Lorentzian for robustness**: When outliers are suspected but should be retained
4. **Compare models objectively**: Use AIC/BIC for model selection
5. **Validate with synthetic data**: Test your approach on simulated datasets

**Parameter Configuration:**

1. **Use physically motivated bounds**: Noise parameters should make sense
2. **Consider fixed parameters**: If instrumental characteristics are known
3. **Monitor convergence**: Some noise models can be harder to optimize
4. **Cross-validate results**: Check stability across different datasets

**Interpretation and Reporting:**

1. **Document your choice**: Report which noise model was used and why
2. **Show model comparison**: Include AIC/BIC comparisons
3. **Analyze residuals**: Validate your noise model choice post-fit
4. **Consider systematic effects**: Large noise parameters may indicate missing physics

Common Pitfalls
~~~~~~~~~~~~~~~

**Over-robust fitting**:
  - Very flexible noise models can mask real astrophysical signals
  - Balance robustness against statistical power

**Inappropriate parameter bounds**:
  - Noise parameters should be constrained by physical expectations
  - Unconstrained noise can compensate for model inadequacies

**Model comparison without validation**:
  - AIC/BIC prefer simpler models - validate on independent data
  - Consider cross-validation for model selection

Next Steps
----------

With robust noise modeling:

- Apply to real pulsar timing datasets with known systematics
- Explore hierarchical noise models for multi-survey data
- Combine with advanced optimization techniques
- Use noise diagnostics to identify instrumental or astrophysical systematics

Understanding noise models enables reliable parameter inference from real astronomical data with realistic uncertainties and outliers.