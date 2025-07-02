#!/usr/bin/env python3
"""
Analyze all sweep hyperparameters to find top 2 values for each
"""

import json
from collections import defaultdict

# Load completed runs configs
completed_runs = """
1. kind-sweep-13: 46.9% - resnet34, max_lr=5e-4, base_lr=1e-6, pretrained=true, snr_layer=bottleneck_128, dropout=0.5, batch=256
2. apricot-sweep-5: 44.4% - resnet34, max_lr=1e-4, base_lr=1e-5, pretrained=false, snr_layer=bottleneck_128, dropout=0.5, batch=256  
3. exalted-sweep-11: 43.2% - resnet34, max_lr=1e-4, base_lr=1e-4, pretrained=true, snr_layer=bottleneck_128, dropout=0.5, batch=64
4. zesty-sweep-10: 41.4% - resnet34, max_lr=1e-4, base_lr=1e-4, pretrained=false, snr_layer=dual_layer, dropout=0.5, batch=256
5. balmy-sweep-7: 39.9% - resnet34, max_lr=1e-4, base_lr=1e-4, pretrained=true, snr_layer=bottleneck_128, dropout=0.5, batch=256
6. lucky-sweep-1: 34.1% - resnet18, max_lr=5e-4, base_lr=1e-4, pretrained=true, snr_layer=bottleneck_64, dropout=0.25, batch=128
7. twilight-sweep-18: running - resnet34, max_lr=1e-3, base_lr=1e-4, pretrained=true, snr_layer=standard, dropout=0.25, batch=64
"""

# Parse run data manually
runs = [
    {'name': 'kind-sweep-13', 'acc': 46.9, 'model': 'resnet34', 'max_lr': '5e-4', 'pretrained': True, 
     'snr_layer': 'bottleneck_128', 'dropout': 0.5, 'batch_size': 256, 'weight_decay': '1e-5'},
    {'name': 'apricot-sweep-5', 'acc': 44.4, 'model': 'resnet34', 'max_lr': '1e-4', 'pretrained': False,
     'snr_layer': 'bottleneck_128', 'dropout': 0.5, 'batch_size': 256, 'weight_decay': '1e-3'},
    {'name': 'exalted-sweep-11', 'acc': 43.2, 'model': 'resnet34', 'max_lr': '1e-4', 'pretrained': True,
     'snr_layer': 'bottleneck_128', 'dropout': 0.5, 'batch_size': 64, 'weight_decay': '1e-4'},
    {'name': 'zesty-sweep-10', 'acc': 41.4, 'model': 'resnet34', 'max_lr': '1e-4', 'pretrained': False,
     'snr_layer': 'dual_layer', 'dropout': 0.5, 'batch_size': 256, 'weight_decay': '1e-4'},
    {'name': 'balmy-sweep-7', 'acc': 39.9, 'model': 'resnet34', 'max_lr': '1e-4', 'pretrained': True,
     'snr_layer': 'bottleneck_128', 'dropout': 0.5, 'batch_size': 256, 'weight_decay': '1e-3'},
    {'name': 'lucky-sweep-1', 'acc': 34.1, 'model': 'resnet18', 'max_lr': '5e-4', 'pretrained': True,
     'snr_layer': 'bottleneck_64', 'dropout': 0.25, 'batch_size': 128, 'weight_decay': '1e-3'},
]

# Analyze each hyperparameter
hyperparams = {
    'model': defaultdict(list),
    'max_lr': defaultdict(list),
    'pretrained': defaultdict(list),
    'snr_layer': defaultdict(list),
    'dropout': defaultdict(list),
    'batch_size': defaultdict(list),
    'weight_decay': defaultdict(list)
}

# Group runs by hyperparameter values
for run in runs:
    for param, value in run.items():
        if param in hyperparams:
            hyperparams[param][value].append((run['acc'], run['name']))

print("=== TOP 2 VALUES FOR EACH HYPERPARAMETER ===\n")

# Find top 2 for each hyperparameter
for param, values in hyperparams.items():
    print(f"\n{param.upper()}:")
    
    # Calculate best performance for each value
    value_performance = []
    for value, runs_list in values.items():
        best_acc = max(acc for acc, _ in runs_list)
        avg_acc = sum(acc for acc, _ in runs_list) / len(runs_list)
        value_performance.append((value, best_acc, avg_acc, len(runs_list), runs_list))
    
    # Sort by best performance
    value_performance.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 2
    print(f"Top 2 by best performance:")
    for i, (value, best, avg, count, runs_list) in enumerate(value_performance[:2]):
        print(f"  {i+1}. {value}: Best {best:.1f}%, Avg {avg:.1f}% ({count} runs)")
        # Show which runs used this value
        top_runs = sorted(runs_list, key=lambda x: x[0], reverse=True)[:2]
        for acc, name in top_runs:
            print(f"     - {name}: {acc:.1f}%")

print("\n=== KEY INSIGHTS ===")
print("\n1. MODEL:")
print("   - ResNet34 dominates (5/6 top runs)")
print("   - ResNet18 only appears in lowest performing run")

print("\n2. LEARNING RATE:")
print("   - Best: max_lr=5e-4 (rank 1)")
print("   - Most common: max_lr=1e-4 (ranks 2,3,4,5)")

print("\n3. PRETRAINED:")
print("   - Mixed results - both achieve >44%")
print("   - Pretrained: 46.9% best")
print("   - Random init: 44.4% (rank 2)")

print("\n4. SNR LAYER:")
print("   - bottleneck_128 dominates (4/5 top runs)")
print("   - bottleneck_64 and dual_layer underperform")

print("\n5. DROPOUT:")
print("   - 0.5 used in all top 5 runs")
print("   - 0.25 only in worst run")

print("\n6. BATCH SIZE:")
print("   - 256 most common (4/6 runs)")
print("   - Smaller batches (64, 128) can work but less optimal")