#!/usr/bin/env python3
"""
Final test to verify the complete data pipeline works.
"""
import sys
import os
sys.path.insert(0, '/p/project1/training2529/marxen1/amm-risk-lab/python')

import pandas as pd
import json
import random
import h5py

def test_complete_pipeline():
    """Test the complete data pipeline."""
    
    try:
        from python.ml.PLV.data_io import LPsDataset
        
        # Configuration - ALL FEATURES CREATED BY FEATURE ENGINEERING
        FEATURES = [
            # Basic returns
            "price_return",
            
            # Volatility features
            "price_volatility_3h", "price_volatility_6h", "price_volatility_24h",
            "liquidity_volatility_3h", "liquidity_volatility_6h", "liquidity_volatility_24h",
            "volume_volatility_3h", "volume_volatility_6h", "volume_volatility_24h",
            
            # Moving averages
            "price_ma_3h", "price_ma_6h", "price_ma_24h",
            "liquidity_ma_3h", "liquidity_ma_6h", "liquidity_ma_24h",
            "volume_ma_3h", "volume_ma_6h", "volume_ma_24h",
            
            # Cross-asset features
            "price_volume_corr_6h", "price_volume_corr_24h", "price_liquidity_corr_6h",
            "volume_liquidity_return_ratio", "volume_liquidity_ratio_ma_6h", "volume_liquidity_ratio_volatility",
            
            # Microstructure features
            "price_volatility_to_volume_ratio", "liquidity_return_depth_proxy",
            "market_impact_proxy", "price_efficiency",
            
            # Momentum features
            "price_momentum_3h", "price_momentum_12h", "price_momentum_24h",
            "liquidity_momentum_6h", "liquidity_momentum_24h",
            "volume_momentum_6h", "volume_momentum_24h",
            "price_deviation_from_ma_6h", "price_deviation_from_ma_24h", "liquidity_deviation_from_ma_6h",
            
            # Volatility regime features
            "high_vol_regime_3h", "high_vol_regime_24h", "vol_cluster_indicator", "vol_term_structure",
            
            # Liquidity features
            "liquidity_return_concentration", "liquidity_return_percentile_6h", "liquidity_return_percentile_24h",
            "liquidity_stress", "liquidity_abundance", "liquidity_stability",
            
            # Temporal features
            "hour", "day_of_week", "month", "season",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "us_market_hours", "asia_market_hours", "weekend",
            
            # Technical indicators
            "price_rsi_6h", "price_rsi_24h", "liquidity_rsi_6h", "volume_rsi_6h",
            "price_bb_upper_6h", "price_bb_lower_6h", "price_bb_position",
            
            # Shock detection features
            "price_shock_3h", "price_shock_6h", "volume_spike", "volume_drought",
            "liquidity_drain", "liquidity_injection"
        ]
        TARGETS = ["liquidity_return", "volume_return"]
        
        SPLIT_DATES = {
            'train_start': '2023-01-01',
            'train_end': '2024-12-31',
            'val_start': '2025-01-01', 
            'val_end': '2025-05-31',
            'test_start': '2025-06-01',
            'test_end': '2025-06-30'
        }
        
        transformed_file = "/p/scratch/training2529/uniswap_pools_data_transformed.h5"
        
        print("=== Testing Complete Data Pipeline ===")
        
        # 1. Get available pool addresses from transformed file
        print("1. Getting available pool addresses...")
        
        # Directly extract pool addresses from HDF5 keys since meta might be empty
        with h5py.File(transformed_file, 'r') as h5f:
            available_pools = [k[5:] for k in h5f.keys() if k.startswith('pool_')]  # Remove 'pool_' prefix
        
        print(f"   ✓ Found {len(available_pools)} pools in transformed file")
        
        # 2. Sample a subset for testing
        print("2. Selecting sample pools for testing...")
        sample_size = min(500, len(available_pools))  # Use up to 50 pools for testing
        # sample_size = len(available_pools)  # Use all available pools for testing
        sample_pools = random.sample(available_pools, sample_size)
        print(f"   ✓ Selected {len(sample_pools)} pools for testing")
        
        # 3. Create training dataset
        print("3. Creating training PyTorch dataset...")
        train_dataset = LPsDataset(
            hdf5_path=transformed_file,
            pool_addresses=sample_pools,
            features=FEATURES,
            targets=TARGETS,
            n_lags=10,  # 24 hours of lag
            split='train',
            split_dates=SPLIT_DATES,
            verbose=1
        )
        print(f"   ✓ Training dataset created with {len(train_dataset)} samples")
        
        # 4. Create validation dataset
        print("4. Creating validation PyTorch dataset...")
        val_dataset = LPsDataset(
            hdf5_path=transformed_file,
            pool_addresses=sample_pools,
            features=FEATURES,
            targets=TARGETS,
            n_lags=10,
            split='val',
            split_dates=SPLIT_DATES,
            verbose=1
        )
        print(f"   ✓ Validation dataset created with {len(val_dataset)} samples")
        
        # 5. Test data access
        print("5. Testing data access...")
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"   ✓ Sample X shape: {sample[0].shape}")  # Features + target lags
            print(f"   ✓ Sample y_cls_1 shape: {sample[1].shape}")  # Classification target 1
            print(f"   ✓ Sample y_reg_1 shape: {sample[2].shape}")  # Regression target 1
            print(f"   ✓ Sample y_cls_2 shape: {sample[3].shape}")  # Classification target 2
            print(f"   ✓ Sample y_reg_2 shape: {sample[4].shape}")  # Regression target 2
            print(f"   ✓ Feature dtype: {sample[0].dtype}")
            print(f"   ✓ Target dtypes: {sample[1].dtype}, {sample[2].dtype}, {sample[3].dtype}, {sample[4].dtype}")
        
        print("\n=== SUCCESS: Complete pipeline working! ===")
        print(f"Available pools: {len(available_pools)}")
        print(f"Sample pools tested: {len(sample_pools)}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Features used: {len(FEATURES)}")
        print(f"Targets: {TARGETS}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    print(f"\nPipeline test: {'PASSED' if success else 'FAILED'}")
