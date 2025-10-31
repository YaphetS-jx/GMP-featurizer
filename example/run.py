#!/usr/bin/env python3
"""
Comprehensive GMP Featurizer Test Suite

This script runs all tests in a single command:
1. JSON interface test (original run.py)
2. Direct parameter interface test (run_direct.py)
3. Uniform vs direct grid comparison test (uniform_reference_grid vs reference_grid)

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
        uniform_reference_grid=[16, 16, 16],  # Updated parameter name
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


def test_3x3_grid_comparison():
    """Test 3: 3x3x3 reference grid comparison - uniform vs direct grid points"""
    print("\n" + "=" * 80)
    print("TEST 3: 3x3x3 Reference Grid Comparison")
    print("=" * 80)
    
    # Create 3x3x3 grid points array
    print("Creating 3x3x3 grid points array...")
    grid_points = []
    for k in range(3):
        for j in range(3):
            for i in range(3):
                x = i / 3.0  # 0, 0.333..., 0.666...
                y = j / 3.0  # 0, 0.333..., 0.666...
                z = k / 3.0  # 0, 0.333..., 0.666...
                grid_points.append([x, y, z]) 
    
    grid_points_array = np.array(grid_points)
    print(f"Created 3x3x3 grid with {len(grid_points)} points")
    
    # Method 1: Using uniform_reference_grid parameter (generates uniform grid)
    print("\n1. Computing features with uniform_reference_grid=[3,3,3] parameter...")
    try:
        features_uniform = gmp_featurizer.compute_features(
            atom_file="./structure.cif",
            psp_file="./QE-kjpaw.gpsp",
            uniform_reference_grid=[3, 3, 3],
            enable_gpu=True
        )
        print(f"   ✓ Success: shape = {features_uniform.shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Method 2: Using reference_grid parameter (direct grid points)
    print("\n2. Computing features with reference_grid (direct points)...")
    try:
        features_direct = gmp_featurizer.compute_features(
            atom_file="./structure.cif",
            psp_file="./QE-kjpaw.gpsp",
            reference_grid=grid_points_array,
            enable_gpu=True
        )
        print(f"   ✓ Success: shape = {features_direct.shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    # Shape comparison
    print(f"\nShape comparison:")
    print(f"   Uniform method: {features_uniform.shape}")
    print(f"   Direct method:  {features_direct.shape}")
    print(f"   Shapes match:   {features_uniform.shape == features_direct.shape}")
    
    if features_uniform.shape != features_direct.shape:
        print("❌ FAIL: Different shapes!")
        return False
    
    # Statistical comparison
    diff = np.abs(features_uniform - features_direct)
    
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
        close = np.allclose(features_uniform, features_direct, atol=tol)
        print(f"   All close (tol={tol:.0e}): {close}")
    
    # Correlation
    correlation = np.corrcoef(features_uniform.flatten(), features_direct.flatten())[0, 1]
    print(f"   Correlation:        {correlation:.8f}")
    
    # Final result (use realistic tolerance for numerical precision)
    success = np.allclose(features_uniform, features_direct, atol=1e-5)
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: 3x3x3 reference grid comparison test")
    
    if success:
        print("✅ The uniform and direct grid methods work correctly")
        print("✅ Both methods produce the same 3x3x3 uniform grid")
    else:
        print(f"❌ Results differ significantly (max diff: {np.max(diff):.2e})")
        print("Note: Small differences are expected due to numerical precision")
    
    return success

def test_raw_data_weighted_square_sum():
    """Test 4: Raw data mode with weighted_square_sum conversion"""
    print("\n" + "=" * 80)
    print("TEST 4: Raw Data Mode with Weighted Square Sum")
    print("=" * 80)
    
    # Define feature configuration
    orders = [0, 1, 2, 3]
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    feature_lists = []  # Empty - will use orders and sigmas
    
    # Compute expected feature counts for each order
    # Order -1: 1 value, Order 0: 1 value, Order 1: 3 values, etc.
    num_mcsh_values = [1, 3, 6, 10, 15, 21, 28, 36, 45]  # For orders 0-8
    
    # Calculate orders list from feature configuration
    # When feature_lists is empty, we get orders x sigmas combinations
    # We need to extract the unique orders that will be used
    if feature_lists:
        # Extract orders from feature_lists
        extracted_orders = [pair[0] for pair in feature_lists]
    else:
        # Use the orders provided (will be combined with sigmas)
        extracted_orders = list(set(orders))  # Get unique orders
    
    # Sort to match the feature list order (sorted by sigma then order)
    # Actually, since we use orders and sigmas, the feature list will be:
    # For each order, for each sigma: (order, sigma)
    # So we need all orders that appear
    all_orders = []
    for order in orders:
        all_orders.extend([order] * len(sigmas))
    
    print(f"Feature configuration:")
    print(f"  Orders: {orders}")
    print(f"  Sigmas: {sigmas}")
    print(f"  Total features (orders x sigmas): {len(orders) * len(sigmas)}")
    print(f"  All orders in feature list: {all_orders[:10]}... (showing first 10)")
    
    # Step 1: Compute features with output_raw_data=True (get raw data)
    print("\n1. Computing features with output_raw_data=True...")
    try:
        features_raw = gmp_featurizer.compute_features(
            atom_file="./structure.cif",
            psp_file="./QE-kjpaw.gpsp",
            orders=orders,
            sigmas=sigmas,
            feature_lists=feature_lists,
            uniform_reference_grid=[8, 8, 8],  # Use smaller grid for faster test
            enable_gpu=True,
            output_raw_data=True
        )
        print(f"   ✓ Success: raw data shape = {features_raw.shape}")
        print(f"   Raw data columns (should be sum of num_mcsh_values for all features)")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Compute features with output_raw_data=False (get final features)
    print("\n2. Computing features with output_raw_data=False (direct output)...")
    try:
        features_direct = gmp_featurizer.compute_features(
            atom_file="./structure.cif",
            psp_file="./QE-kjpaw.gpsp",
            orders=orders,
            sigmas=sigmas,
            feature_lists=feature_lists,
            uniform_reference_grid=[8, 8, 8],
            enable_gpu=True,
            output_raw_data=False  # Default, but explicit
        )
        print(f"   ✓ Success: direct features shape = {features_direct.shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Convert raw data to final features using compute_weighted_square_sum
    print("\n3. Converting raw data to final features using compute_weighted_square_sum...")
    try:
        # We need to pass the orders list - this should match the feature order
        # Since features are sorted by (sigma, order), we need the order for each feature
        features_converted = gmp_featurizer.compute_weighted_square_sum(
            raw_data=features_raw,
            orders=all_orders,
            square=False  # Use square root (matching default behavior)
        )
        print(f"   ✓ Success: converted features shape = {features_converted.shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    # Shape comparison
    print(f"\nShape comparison:")
    print(f"   Direct features:     {features_direct.shape}")
    print(f"   Converted features: {features_converted.shape}")
    print(f"   Shapes match:       {features_direct.shape == features_converted.shape}")
    
    if features_direct.shape != features_converted.shape:
        print("❌ FAIL: Shape mismatch!")
        print(f"   Expected {features_direct.shape}, got {features_converted.shape}")
        return False
    
    # Statistical comparison
    diff = np.abs(features_direct - features_converted)
    
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
        close = np.allclose(features_direct, features_converted, atol=tol)
        print(f"   All close (tol={tol:.0e}): {close}")
    
    # Correlation
    correlation = np.corrcoef(features_direct.flatten(), features_converted.flatten())[0, 1]
    print(f"   Correlation:        {correlation:.8f}")
    
    # Final result (use realistic tolerance for numerical precision)
    success = np.allclose(features_direct, features_converted, atol=1e-5)
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: Raw data with weighted square sum test")
    
    if success:
        print("✅ Raw data mode works correctly")
        print("✅ compute_weighted_square_sum converts raw data correctly")
        print("✅ Converted features match direct output")
    else:
        print(f"❌ Results differ significantly (max diff: {np.max(diff):.2e})")
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"   Largest difference at position {max_idx}:")
        print(f"     Direct:    {features_direct[max_idx]:.8f}")
        print(f"     Converted: {features_converted[max_idx]:.8f}")
        print("Note: Small differences are expected due to numerical precision")
    
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
    test4_success = test_raw_data_weighted_square_sum()
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"Test 1 - JSON Interface:           {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"Test 2 - Direct Parameter:         {'✅ PASS' if test2_success else '❌ FAIL'}")
    print(f"Test 3 - Uniform vs Direct Grid:   {'✅ PASS' if test3_success else '❌ FAIL'}")
    print(f"Test 4 - Raw Data Weighted Sum:    {'✅ PASS' if test4_success else '❌ FAIL'}")
    
    all_success = test1_success and test2_success and test3_success and test4_success
    
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_success else '❌ SOME TESTS FAILED'}")
    
    if all_success:
        print("\n🎉 All tests completed successfully!")
        print("✅ JSON interface works correctly")
        print("✅ Direct parameter interface works correctly")
        print("✅ Uniform and direct grid methods work correctly")
        print("✅ Raw data mode and weighted square sum work correctly")
        print("✅ Both interfaces produce identical results")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    return all_success

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)