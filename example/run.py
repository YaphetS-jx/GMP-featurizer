#!/usr/bin/env python3
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'python'))
import gmp_featurizer

# Compute features (GPU automatically initialized on import)
features = gmp_featurizer.compute_features('config.json')
print(f"Features shape: {features.shape}")

# Load reference data
print("\nLoading reference data from gmpFeatures.dat...")
ref_data = []
with open('gmpFeatures.dat', 'r') as f:
    for line in f:
        line = line.strip()
        if line:  # Skip empty lines
            values = [float(x.strip()) for x in line.split(',') if x.strip()]
            if values:  # Only add non-empty rows
                ref_data.append(values)

ref_features = np.array(ref_data)

# Compare features
print(f"\nComparison:")
print(f"Shapes match: {features.shape == ref_features.shape}")

if features.shape == ref_features.shape:
    # Calculate differences
    diff = np.abs(features - ref_features)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    # Check if they're close (within tolerance)
    tolerance = 1e-5
    close = np.allclose(features, ref_features, atol=tolerance)
    print(f"Features are close (tol={tolerance}): {close}")
    
    if not close:
        # Find where differences are largest
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"Largest difference at position {max_idx}: {diff[max_idx]:.8f}")
        print(f"  Computed: {features[max_idx]:.8f}")
        print(f"  Reference: {ref_features[max_idx]:.8f}")
else:
    print("ERROR: Shape mismatch - cannot compare features")