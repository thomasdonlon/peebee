# Peebee Test Suite

This directory contains tests for validating the peebee optimizer and physics calculations.

## Test Files

### `test_optimizer_physics.py`
Comprehensive physics validation test for the optimizer.

**Purpose:** Validates that the optimizer can accurately recover known parameters from mock gravitational acceleration data.

**Test Setup:**
- True model: NFW halo (1e12 M☉, 30 kpc) + Miyamoto-Nagai disk (5e10 M☉, 3.5 kpc, 0.3 kpc)
- Mock data: 100 line-of-sight acceleration measurements within 3 kpc of Sun
- Starting guess: 20% off from true parameters
- Optimization: Gradient descent with log-space mass parameters, linear-space scale lengths
- Success criteria: All parameters recovered within 1% tolerance

**Usage:**
```bash
# Run the test
python tests/test_optimizer_physics.py

# Expected output: All parameters recovered within tolerance
```

**What the test validates:**
- NFW and Miyamoto-Nagai model implementations are correct
- Parameter optimization in mixed log/linear space works properly  
- Gradient descent optimization converges to global minimum
- Composite model parameter handling functions correctly
- Line-of-sight acceleration calculations are accurate

**Maintenance:**
- Run this test after any changes to optimizer, models, or sampling code
- Test should pass with >99% parameter recovery accuracy
- If test fails, check for breaking changes in optimization or model physics

## Running Tests

```bash
# From peebee root directory
python tests/test_optimizer_physics.py
```

The test will output detailed parameter recovery statistics and indicate success/failure clearly.