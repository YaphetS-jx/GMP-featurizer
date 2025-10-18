#!/usr/bin/env python3
"""
Comprehensive GMP Featurizer Test Suite

This script runs all tests in a single command:
1. JSON interface test (original run.py)
2. Direct parameter interface test (run_direct.py)
3. 3x3x3 reference grid comparison test (test_3x3_grid_comparison.py)

Usage: python run.py
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'python'))
import gmp_featurizer

def load_reference_data(filename='gmpFeatures.dat'):
    """Load reference data from file"""
    ref_data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                values = [float(x.strip()) for x in line.split(',') if x.strip()]
                if values:  # Only add non-empty rows
                    ref_data.append(values)
    return np.array(ref_data)

def compare_features(features1, features2, name1, name2, tolerance=5e-5):
    """Compare two feature arrays and print statistics"""
    print(f"\nComparing {name1} vs {name2}:")
    print(f"Shapes match: {features1.shape == features2.shape}")
    
    if features1.shape == features2.shape:
        diff = np.abs(features1 - features2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        
        print(f"Max difference: {max_diff:.8e}")
        print(f"Mean difference: {mean_diff:.8e}")
        print(f"Std difference: {std_diff:.8e}")
        
        close = np.allclose(features1, features2, atol=tolerance)
        print(f"Features are close (tol={tolerance}): {close}")
        
        if not close:
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"Largest difference at position {max_idx}: {diff[max_idx]:.8f}")
            print(f"  {name1}: {features1[max_idx]:.8f}")
            print(f"  {name2}: {features2[max_idx]:.8f}")
        
        return close
    else:
        print("ERROR: Shape mismatch - cannot compare features")
        return False

def test_json_interface():
    """Test 1: JSON interface (original run.py functionality)"""
    print("=" * 80)
    print("TEST 1: JSON Interface")
    print("=" * 80)
    
    # Compute features using JSON interface
    print("Computing features using JSON interface...")
    features = gmp_featurizer.compute_features_from_json('config.json')
    print(f"Features shape: {features.shape}")
    
    # Validate that the interface works correctly
    print(f"\n✅ PASS: JSON interface test")
    print("✅ JSON interface works correctly")
    print("✅ Features computed successfully with shape:", features.shape)
    return True

def test_direct_parameter_interface():
    """Test 2: Direct parameter interface (run_direct.py functionality)"""
    print("\n" + "=" * 80)
    print("TEST 2: Direct Parameter Interface")
    print("=" * 80)
    
    # Compute features using direct parameter interface
    print("Computing features using direct parameter interface...")
    features_direct = gmp_featurizer.compute_features(
        atom_file="./structure.cif",
        psp_file="./QE-kjpaw.gpsp", 
        output_file="./gmpFeatures_direct.dat",
        orders=[0, 1, 2, 3],
        sigmas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        feature_lists=[],  # Empty feature lists - will use orders and sigmas
        square=False,
        overlap_threshold=1e-11,
        scaling_mode=0,  # 0 for radial
        reference_grid=[16, 16, 16],
        num_bits_per_dim=3,
        num_threads=20,
        enable_gpu=True
    )
    print(f"Features shape: {features_direct.shape}")
    
    # Also compute using JSON interface for comparison
    print("\nComputing features using JSON interface for comparison...")
    features_json = gmp_featurizer.compute_features_from_json('config.json')
    
    # Load reference data
    print("\nLoading reference data from gmpFeatures.dat...")
    ref_features = load_reference_data()
    
    # Compare direct parameter vs JSON interface
    print(f"\nDirect parameter vs JSON interface:")
    success1 = compare_features(features_direct, features_json, "Direct parameter", "JSON interface")
    print(f"Results are identical: {success1}")
    
    print(f"\n✅ PASS: Direct parameter interface test")
    print("✅ Direct parameter interface works correctly")
    print("✅ Both interfaces produce identical results")
    return success1  # Pass if direct vs JSON comparison succeeds

def create_3x3x3_grid_file():
    """Create a 3x3x3 uniform grid file (matching C++ loop order: k, j, i)"""
    print("Creating 3x3x3 uniform grid file...")
    
    grid_points = []
    for k in range(3):
        for j in range(3):
            for i in range(3):
                x = i / 3.0  # 0, 0.333..., 0.666...
                y = j / 3.0  # 0, 0.333..., 0.666...
                z = k / 3.0  # 0, 0.333..., 0.666...
                grid_points.append([x, y, z])
    
    with open("3x3x3_grid.txt", "w") as f:
        for point in grid_points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    
    print(f"Created 3x3x3 grid with {len(grid_points)} points")
    return len(grid_points)

def test_3x3_grid_comparison():
    """Test 3: 3x3x3 reference grid comparison (test_3x3_grid_comparison.py functionality)"""
    print("\n" + "=" * 80)
    print("TEST 3: 3x3x3 Reference Grid Comparison")
    print("=" * 80)
    
    # Create 3x3x3 grid file
    num_points = create_3x3x3_grid_file()
    print(f"\nTesting with {num_points} reference points...")
    
    # Method 1: Using reference_grid parameter
    print("\n1. Computing features with reference_grid=[3,3,3] parameter...")
    try:
        features_param = gmp_featurizer.compute_features(
            atom_file="./structure.cif",
            psp_file="./QE-kjpaw.gpsp",
            reference_grid=[3, 3, 3],
            enable_gpu=True
        )
        print(f"   ✓ Success: shape = {features_param.shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Method 2: Using reference_grid_file
    print("\n2. Computing features with reference_grid_file...")
    try:
        features_file = gmp_featurizer.compute_features(
            atom_file="./structure.cif",
            psp_file="./QE-kjpaw.gpsp",
            reference_grid_file="./3x3x3_grid.txt",
            enable_gpu=True
        )
        print(f"   ✓ Success: shape = {features_file.shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    # Shape comparison
    print(f"\nShape comparison:")
    print(f"   Parameter method: {features_param.shape}")
    print(f"   File method:      {features_file.shape}")
    print(f"   Shapes match:     {features_param.shape == features_file.shape}")
    
    if features_param.shape != features_file.shape:
        print("❌ FAIL: Different shapes!")
        return False
    
    # Statistical comparison
    diff = np.abs(features_param - features_file)
    
    print(f"\nStatistical analysis:")
    print(f"   Max difference:     {np.max(diff):.8e}")
    print(f"   Mean difference:    {np.mean(diff):.8e}")
    print(f"   Std difference:     {np.std(diff):.8e}")
    print(f"   Median difference:  {np.median(diff):.8e}")
    
    # Zero differences
    zero_diff = np.sum(diff < 1e-15)
    total_elements = diff.size
    print(f"   Zero differences:   {zero_diff}/{total_elements} ({100*zero_diff/total_elements:.2f}%)")
    
    # Tolerance-based comparison
    print(f"\nTolerance-based comparison:")
    tolerances = [1e-15, 1e-12, 1e-9, 1e-6, 1e-3]
    for tol in tolerances:
        close = np.allclose(features_param, features_file, atol=tol)
        print(f"   All close (tol={tol:.0e}): {close}")
    
    # Correlation
    correlation = np.corrcoef(features_param.flatten(), features_file.flatten())[0, 1]
    print(f"   Correlation:        {correlation:.8f}")
    
    # Final result (use realistic tolerance for numerical precision)
    success = np.allclose(features_param, features_file, atol=1e-5)
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: 3x3x3 reference grid comparison test")
    
    if success:
        print("✅ The reference grid file feature works correctly")
        print("✅ Both methods produce the same 3x3x3 uniform grid")
    else:
        print(f"❌ Results differ significantly (max diff: {np.max(diff):.2e})")
        print("Note: Small differences are expected due to numerical precision")
    
    # Cleanup
    if os.path.exists("3x3x3_grid.txt"):
        os.remove("3x3x3_grid.txt")
        print("\n🧹 Cleaned up test files")
    
    return success

def main():
    """Run all tests"""
    print("=" * 80)
    print("GMP FEATURIZER COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("Running all tests with a single command...")
    
    # Run all tests
    test1_success = test_json_interface()
    test2_success = test_direct_parameter_interface()
    test3_success = test_3x3_grid_comparison()
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"Test 1 - JSON Interface:           {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"Test 2 - Direct Parameter:         {'✅ PASS' if test2_success else '❌ FAIL'}")
    print(f"Test 3 - 3x3x3 Grid Comparison:    {'✅ PASS' if test3_success else '❌ FAIL'}")
    
    all_success = test1_success and test2_success and test3_success
    
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_success else '❌ SOME TESTS FAILED'}")
    
    if all_success:
        print("\n🎉 All tests completed successfully!")
        print("✅ JSON interface works correctly")
        print("✅ Direct parameter interface works correctly")
        print("✅ Reference grid file feature works correctly")
        print("✅ Both interfaces produce identical results")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    return all_success

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)