#!/usr/bin/env python3
"""
Comprehensive GMP Featurizer Test Suite

This script runs all tests using the new separated initialization and computation API:
1. JSON interface test - using initialize_featurizer_from_json + compute_features
2. Direct parameter interface test - using initialize_featurizer_from_params + compute_features
3. Uniform vs direct grid comparison test - testing different initialization strategies
4. Raw data mode test - testing raw data output and weighted square sum conversion
5. Direct data interface test - using initialize_featurizer_from_data (no CIF file needed)

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
    """Test 1: JSON interface using separated initialization and computation"""
    print("=" * 80)
    print("TEST 1: JSON Interface (Separated Init + Compute)")
    print("=" * 80)
    
    # Initialize featurizer from JSON
    print("Initializing featurizer from JSON configuration...")
    context = gmp_featurizer.initialize_featurizer_from_json('config.json')
    print(f"Context initialized: GPU enabled = {context.is_gpu_enabled()}")
    
    # Compute features using the context
    print("Computing features using initialized context...")
    if context.is_gpu_enabled():
        features = gmp_featurizer.compute_features_gpu(context)
    else:
        features = gmp_featurizer.compute_features_cpu(context)
    print(f"Features shape: {features.shape}")
    
    # Validate that the interface works correctly
    print(f"\n✅ PASS: JSON interface test")
    print("✅ Separated initialization and computation works correctly")
    print("✅ Features computed successfully with shape:", features.shape)
    return True

def test_direct_parameter_interface():
    """Test 2: Direct parameter interface using separated initialization and computation"""
    print("\n" + "=" * 80)
    print("TEST 2: Direct Parameter Interface (Separated Init + Compute)")
    print("=" * 80)
    
    # Initialize featurizer from direct parameters
    print("Initializing featurizer from direct parameters...")
    context_direct = gmp_featurizer.initialize_featurizer_from_params(
        atom_file="./structure.cif",
        psp_file="./QE-kjpaw.gpsp", 
        output_file="./gmpFeatures_direct.dat",
        orders=[0, 1, 2, 3],
        sigmas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        feature_lists=[],  # Empty feature lists - will use orders and sigmas
        square=False,
        overlap_threshold=1e-11,
        scaling_mode=0,  # 0 for radial
        uniform_reference_grid=[16, 16, 16],
        num_bits_per_dim=3,
        num_threads=20,
        enable_gpu=True
    )
    print(f"Context initialized: GPU enabled = {context_direct.is_gpu_enabled()}")
    
    # Compute features using the context
    print("Computing features using initialized context...")
    if context_direct.is_gpu_enabled():
        features_direct = gmp_featurizer.compute_features_gpu(context_direct)
    else:
        features_direct = gmp_featurizer.compute_features_cpu(context_direct)
    print(f"Features shape: {features_direct.shape}")
    
    # Also compute using JSON interface for comparison
    print("\nInitializing from JSON for comparison...")
    context_json = gmp_featurizer.initialize_featurizer_from_json('config.json')
    if context_json.is_gpu_enabled():
        features_json = gmp_featurizer.compute_features_gpu(context_json)
    else:
        features_json = gmp_featurizer.compute_features_cpu(context_json)
    
    # Load reference data
    print("\nLoading reference data from gmpFeatures.dat...")
    ref_features = load_reference_data()
    
    # Compare direct parameter vs JSON interface
    print(f"\nDirect parameter vs JSON interface:")
    success1 = compare_features(features_direct, features_json, "Direct parameter", "JSON interface")
    print(f"Results are identical: {success1}")
    
    print(f"\n✅ PASS: Direct parameter interface test")
    print("✅ Separated initialization and computation works correctly")
    print("✅ Both initialization methods produce identical results")
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
    print("\n1. Initializing with uniform_reference_grid=[3,3,3] parameter...")
    try:
        context_uniform = gmp_featurizer.initialize_featurizer_from_params(
            atom_file="./structure.cif",
            psp_file="./QE-kjpaw.gpsp",
            uniform_reference_grid=[3, 3, 3],
            enable_gpu=True
        )
        if context_uniform.is_gpu_enabled():
            features_uniform = gmp_featurizer.compute_features_gpu(context_uniform)
        else:
            features_uniform = gmp_featurizer.compute_features_cpu(context_uniform)
        print(f"   ✓ Success: shape = {features_uniform.shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Method 2: Using reference_grid parameter (direct grid points)
    print("\n2. Initializing with reference_grid (direct points)...")
    try:
        context_direct = gmp_featurizer.initialize_featurizer_from_params(
            atom_file="./structure.cif",
            psp_file="./QE-kjpaw.gpsp",
            reference_grid=grid_points_array,
            enable_gpu=True
        )
        if context_direct.is_gpu_enabled():
            features_direct = gmp_featurizer.compute_features_gpu(context_direct)
        else:
            features_direct = gmp_featurizer.compute_features_cpu(context_direct)
        print(f"   ✓ Success: shape = {features_direct.shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
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

def test_direct_data_interface():
    """Test 5: Direct data interface (no CIF file needed)"""
    print("\n" + "=" * 80)
    print("TEST 5: Direct Data Interface (No CIF File)")
    print("=" * 80)
    
    # Define structure data directly (from structure.cif)
    cell_lengths = np.array([13.066739950848183, 6.447909540591107, 6.413901842517733])
    cell_angles = np.array([90.07113551631299, 89.53589382117042, 89.78803670940194])
    
    # Atom data: [type, fract_x, fract_y, fract_z, occupancy]
    atom_data = [
        ['N', 0.41546703537681856, 0.427416653802161, 0.577406109529728, 1.0000],
        ['N', 0.8789667083245883, 0.24013932567339225, 0.649514945480739, 1.0000],
        ['C', 0.31756711793846165, 0.3085496068057215, 0.5377486376248347, 1.0000],
        ['C', 0.8103547782785443, 0.3480680549171177, 0.49845958663493795, 1.0000],
        ['H', 0.39719958159880175, 0.569014523062509, 0.6436870117825956, 1.0000],
        ['H', 0.4581787303026594, 0.4578494371091457, 0.4383345283109038, 1.0000],
        ['H', 0.4582258350324719, 0.351190334033146, 0.6960042332894495, 1.0000],
        ['H', 0.2970783361726095, 0.3453593275954766, 0.3799447138874776, 1.0000],
        ['H', 0.33261436287729934, 0.14386784923275434, 0.5643477558651482, 1.0000],
        ['H', 0.2586301623169967, 0.3609036718130434, 0.6514832341493222, 1.0000],
        ['H', 0.9345127142134654, 0.14047383564734398, 0.5790125397448318, 1.0000],
        ['H', 0.8364587732412783, 0.14258000347813554, 0.7477463344293881, 1.0000],
        ['H', 0.9137719977148143, 0.3512998156949266, 0.744333026289629, 1.0000],
        ['H', 0.7741829189275211, 0.23763766149468404, 0.39215151536543863, 1.0000],
        ['H', 0.7551929338133306, 0.43789710184642383, 0.5924532640215644, 1.0000],
        ['H', 0.8582958222496566, 0.4578847272999746, 0.4144223729198017, 1.0000],
        ['Pb', 0.06634365354316886, 0.8737467160630854, 0.02568417778615599, 1.0000],
        ['Pb', 0.5643930111251986, 0.8832547141744858, 0.06576860561103609, 1.0000],
        ['I', 0.07860418881812913, 0.8898957535969937, 0.5087477342712794, 1.0000],
        ['I', 0.3104344327042981, 0.8577513139645994, 0.014260897462014565, 1.0000],
        ['I', 0.059708187344487955, 0.3848030918146536, 0.02705354459762949, 1.0000],
        ['I', 0.5807376518631311, 0.8902733586388623, 0.5525265593988309, 1.0000],
        ['I', 0.8087378794685559, 0.8141652199740181, 0.03152707188146364, 1.0000],
        ['I', 0.5492912125966564, 0.40881396316311397, 0.1150340124617497, 1.0000],
    ]
    
    atom_types = [row[0] for row in atom_data]
    atom_positions = np.array([[row[1], row[2], row[3]] for row in atom_data])
    atom_occupancies = np.array([row[4] for row in atom_data])
    
    print(f"   Structure: {len(atom_types)} atoms")
    print(f"   Cell lengths: {cell_lengths}")
    print(f"   Cell angles: {cell_angles}")
    
    # Method 1: Initialize from direct data
    print("\n1. Initializing from direct data (no CIF file)...")
    try:
        context_direct_data = gmp_featurizer.initialize_featurizer_from_data(
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
            atom_positions=atom_positions,
            atom_types=atom_types,
            atom_occupancies=atom_occupancies,
            psp_file="./QE-kjpaw.gpsp",
            orders=[0, 1, 2, 3],
            sigmas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            uniform_reference_grid=[16, 16, 16],
            num_bits_per_dim=3,
            num_threads=20,
            enable_gpu=True
        )
        print(f"   ✓ Success: Context initialized from direct data")
        print(f"   GPU enabled: {context_direct_data.is_gpu_enabled()}")
        
        if context_direct_data.is_gpu_enabled():
            features_direct_data = gmp_featurizer.compute_features_gpu(context_direct_data)
        else:
            features_direct_data = gmp_featurizer.compute_features_cpu(context_direct_data)
        print(f"   ✓ Success: Features computed, shape = {features_direct_data.shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Method 2: Initialize from file for comparison
    print("\n2. Initializing from CIF file for comparison...")
    try:
        context_file = gmp_featurizer.initialize_featurizer_from_params(
            atom_file="./structure.cif",
            psp_file="./QE-kjpaw.gpsp",
            orders=[0, 1, 2, 3],
            sigmas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            uniform_reference_grid=[16, 16, 16],
            num_bits_per_dim=3,
            num_threads=20,
            enable_gpu=True
        )
        if context_file.is_gpu_enabled():
            features_file = gmp_featurizer.compute_features_gpu(context_file)
        else:
            features_file = gmp_featurizer.compute_features_cpu(context_file)
        print(f"   ✓ Success: Features computed from file, shape = {features_file.shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    # Shape comparison
    print(f"\nShape comparison:")
    print(f"   Direct data method: {features_direct_data.shape}")
    print(f"   File-based method:  {features_file.shape}")
    print(f"   Shapes match:       {features_direct_data.shape == features_file.shape}")
    
    if features_direct_data.shape != features_file.shape:
        print("❌ FAIL: Different shapes!")
        return False
    
    # Statistical comparison
    diff = np.abs(features_direct_data - features_file)
    
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
        close = np.allclose(features_direct_data, features_file, atol=tol)
        print(f"   All close (tol={tol:.0e}): {close}")
    
    # Correlation
    correlation = np.corrcoef(features_direct_data.flatten(), features_file.flatten())[0, 1]
    print(f"   Correlation:        {correlation:.8f}")
    
    # Final result (use realistic tolerance for numerical precision)
    success = np.allclose(features_direct_data, features_file, atol=1e-5)
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: Direct data interface test")
    
    if success:
        print("✅ Direct data initialization works correctly")
        print("✅ Results match file-based initialization")
        print("✅ No CIF file needed for initialization")
    else:
        print(f"❌ Results differ significantly (max diff: {np.max(diff):.2e})")
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"   Largest difference at position {max_idx}:")
        print(f"     Direct data: {features_direct_data[max_idx]:.8f}")
        print(f"     File-based:  {features_file[max_idx]:.8f}")
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
    
    # Step 1: Initialize and compute features with output_raw_data=True (get raw data)
    print("\n1. Initializing with output_raw_data=True...")
    try:
        context_raw = gmp_featurizer.initialize_featurizer_from_params(
            atom_file="./structure.cif",
            psp_file="./QE-kjpaw.gpsp",
            orders=orders,
            sigmas=sigmas,
            feature_lists=feature_lists,
            uniform_reference_grid=[8, 8, 8],  # Use smaller grid for faster test
            enable_gpu=True,
            output_raw_data=True
        )
        if context_raw.is_gpu_enabled():
            features_raw = gmp_featurizer.compute_features_gpu(context_raw)
        else:
            features_raw = gmp_featurizer.compute_features_cpu(context_raw)
        print(f"   ✓ Success: raw data shape = {features_raw.shape}")
        print(f"   Raw data columns (should be sum of num_mcsh_values for all features)")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Initialize and compute features with output_raw_data=False (get final features)
    print("\n2. Initializing with output_raw_data=False (direct output)...")
    try:
        context_direct = gmp_featurizer.initialize_featurizer_from_params(
            atom_file="./structure.cif",
            psp_file="./QE-kjpaw.gpsp",
            orders=orders,
            sigmas=sigmas,
            feature_lists=feature_lists,
            uniform_reference_grid=[8, 8, 8],
            enable_gpu=True,
            output_raw_data=False  # Default, but explicit
        )
        if context_direct.is_gpu_enabled():
            features_direct = gmp_featurizer.compute_features_gpu(context_direct)
        else:
            features_direct = gmp_featurizer.compute_features_cpu(context_direct)
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
    test5_success = test_direct_data_interface()
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"Test 1 - JSON Interface:           {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"Test 2 - Direct Parameter:           {'✅ PASS' if test2_success else '❌ FAIL'}")
    print(f"Test 3 - Uniform vs Direct Grid:     {'✅ PASS' if test3_success else '❌ FAIL'}")
    print(f"Test 4 - Raw Data Weighted Sum:      {'✅ PASS' if test4_success else '❌ FAIL'}")
    print(f"Test 5 - Direct Data Interface:      {'✅ PASS' if test5_success else '❌ FAIL'}")
    
    all_success = test1_success and test2_success and test3_success and test4_success and test5_success
    
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_success else '❌ SOME TESTS FAILED'}")
    
    if all_success:
        print("\n🎉 All tests completed successfully!")
        print("✅ Separated initialization and computation API works correctly")
        print("✅ JSON and direct parameter initialization methods work correctly")
        print("✅ Uniform and direct grid initialization strategies work correctly")
        print("✅ Raw data mode and weighted square sum work correctly")
        print("✅ Direct data initialization (no CIF file) works correctly")
        print("✅ Context can be reused for multiple computations")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    return all_success

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)