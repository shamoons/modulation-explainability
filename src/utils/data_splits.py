# src/utils/data_splits.py

import numpy as np
from sklearn.model_selection import train_test_split


def create_stratified_split(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Create stratified train/validation/test split for multi-task dataset.
    
    Args:
        dataset: Dataset with modulation_labels_list and snr_labels_list attributes
        train_ratio: Proportion for training set (default: 0.8)
        val_ratio: Proportion for validation set (default: 0.1)
        test_ratio: Proportion for test set (default: 0.1)
        random_state: Random seed for reproducible splits
        
    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    print("Creating stratified train/val/test split...")
    indices = list(range(len(dataset)))
    
    # Get modulation and SNR labels directly from dataset (much faster than __getitem__ calls)
    if hasattr(dataset, 'modulation_labels_list') and hasattr(dataset, 'snr_labels_list'):
        all_mod_labels = dataset.modulation_labels_list
        all_snr_labels = dataset.snr_labels_list
        print(f"Using cached labels from dataset ({len(all_mod_labels)} samples)")
    else:
        # Fallback to individual lookups if labels not cached
        print("Dataset labels not cached, using individual lookups (slower)...")
        all_mod_labels = []
        all_snr_labels = []
        for i in indices:
            _, mod_label, snr_label = dataset[i]
            all_mod_labels.append(mod_label)
            all_snr_labels.append(snr_label)
    
    # Create combined labels for stratification (mod * num_snr_classes + snr)
    num_snr_classes = len(dataset.snr_labels)
    combined_labels = np.array([mod * num_snr_classes + snr for mod, snr in zip(all_mod_labels, all_snr_labels)])
    
    # First split: train vs (val + test)
    temp_ratio = val_ratio + test_ratio
    train_idx, temp_idx = train_test_split(
        indices, 
        test_size=temp_ratio, 
        stratify=combined_labels,
        random_state=random_state
    )
    
    # Second split: val vs test from temp
    test_ratio_adjusted = test_ratio / temp_ratio  # Adjust test ratio for the temp set
    temp_combined = combined_labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio_adjusted,
        stratify=temp_combined,
        random_state=random_state
    )
    
    # Print split statistics
    total_samples = len(dataset)
    print(f"Dataset split - Train: {len(train_idx)} ({len(train_idx)/total_samples*100:.1f}%), "
          f"Val: {len(val_idx)} ({len(val_idx)/total_samples*100:.1f}%), "
          f"Test: {len(test_idx)} ({len(test_idx)/total_samples*100:.1f}%)")
    
    # Verify stratification worked
    unique_combinations = len(np.unique(combined_labels))
    print(f"Total unique (mod, SNR) combinations: {unique_combinations}")
    
    return train_idx, val_idx, test_idx


def verify_stratification(dataset, train_idx, val_idx, test_idx):
    """
    Verify that stratification preserved class distributions.
    
    Args:
        dataset: Dataset object
        train_idx: Training indices
        val_idx: Validation indices  
        test_idx: Test indices
    """
    
    def get_class_distribution(indices):
        mod_counts = {}
        snr_counts = {}
        combined_counts = {}
        
        # Use cached labels for faster verification
        if hasattr(dataset, 'modulation_labels_list') and hasattr(dataset, 'snr_labels_list'):
            for idx in indices:
                mod_label = dataset.modulation_labels_list[idx]
                snr_label = dataset.snr_labels_list[idx]
                mod_type = dataset.inverse_modulation_labels[mod_label]
                snr_value = dataset.inverse_snr_labels[snr_label]
                combined_key = (mod_type, snr_value)
                
                mod_counts[mod_type] = mod_counts.get(mod_type, 0) + 1
                snr_counts[snr_value] = snr_counts.get(snr_value, 0) + 1
                combined_counts[combined_key] = combined_counts.get(combined_key, 0) + 1
        else:
            # Fallback to individual lookups
            for idx in indices:
                _, mod_label, snr_label = dataset[idx]
                mod_type = dataset.inverse_modulation_labels[mod_label]
                snr_value = dataset.inverse_snr_labels[snr_label]
                combined_key = (mod_type, snr_value)
                
                mod_counts[mod_type] = mod_counts.get(mod_type, 0) + 1
                snr_counts[snr_value] = snr_counts.get(snr_value, 0) + 1
                combined_counts[combined_key] = combined_counts.get(combined_key, 0) + 1
            
        return mod_counts, snr_counts, combined_counts
    
    print("\nVerifying stratification...")
    
    train_mod, train_snr, train_combined = get_class_distribution(train_idx)
    val_mod, val_snr, val_combined = get_class_distribution(val_idx)
    test_mod, test_snr, test_combined = get_class_distribution(test_idx)
    
    # Check that all splits have all classes
    all_mod_types = set(train_mod.keys()) | set(val_mod.keys()) | set(test_mod.keys())
    all_snr_values = set(train_snr.keys()) | set(val_snr.keys()) | set(test_snr.keys())
    
    print(f"Modulation types in train/val/test: {len(train_mod)}/{len(val_mod)}/{len(test_mod)} "
          f"(total: {len(all_mod_types)})")
    print(f"SNR values in train/val/test: {len(train_snr)}/{len(val_snr)}/{len(test_snr)} "
          f"(total: {len(all_snr_values)})")
    print(f"Combined classes in train/val/test: {len(train_combined)}/{len(val_combined)}/{len(test_combined)}")
    
    # Check for missing classes
    missing_mod_val = all_mod_types - set(val_mod.keys())
    missing_mod_test = all_mod_types - set(test_mod.keys())
    missing_snr_val = all_snr_values - set(val_snr.keys())
    missing_snr_test = all_snr_values - set(test_snr.keys())
    
    if missing_mod_val or missing_mod_test or missing_snr_val or missing_snr_test:
        print("⚠️  Warning: Some classes missing from splits:")
        if missing_mod_val: print(f"  Val missing mods: {missing_mod_val}")
        if missing_mod_test: print(f"  Test missing mods: {missing_mod_test}")
        if missing_snr_val: print(f"  Val missing SNRs: {missing_snr_val}")
        if missing_snr_test: print(f"  Test missing SNRs: {missing_snr_test}")
    else:
        print("✅ All classes present in all splits")