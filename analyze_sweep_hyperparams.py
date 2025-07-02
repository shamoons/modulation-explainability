#!/usr/bin/env python3
"""
Analyze all sweep runs to find top 2 values for each hyperparameter
"""

import json
from collections import defaultdict

# Load the sweep data
with open('sweep_latest_status.json', 'r') as f:
    data = json.load(f)

# Need to query each run individually to get full config
# For now, let's work with what we have in summaryMetrics

all_runs = data['project']['sweep']['runs']['edges']

# Collect data from all runs
runs_data = []
for run in all_runs:
    node = run['node']
    if node.get('summaryMetrics') and node['summaryMetrics'] != '{}':
        summary = json.loads(node['summaryMetrics'])
        
        # Get the best accuracy
        if node['state'] == 'finished':
            accuracy = summary.get('final_test_combined_accuracy', 0)
        else:
            accuracy = summary.get('val_combined_accuracy', 0)
        
        # Extract what we can from summary
        runs_data.append({
            'name': node['displayName'],
            'state': node['state'],
            'accuracy': accuracy * 100 if accuracy < 1 else accuracy,  # Convert to percentage
            'learning_rate': summary.get('learning_rate', None),
            'epoch': summary.get('epoch', summary.get('best_epoch', None))
        })

# Sort by accuracy
runs_data = [r for r in runs_data if r['accuracy'] > 0]
runs_data.sort(key=lambda x: x['accuracy'], reverse=True)

print('=== All Runs Ranked by Performance ===\n')
print('Rank | Run Name              | Accuracy | Learning Rate | Status')
print('-' * 70)
for i, run in enumerate(runs_data):
    lr_str = f"{run['learning_rate']:.5f}" if run['learning_rate'] else 'N/A'
    print(f"{i+1:2d}   | {run['name']:20s} | {run['accuracy']:5.1f}%   | {lr_str:13s} | {run['state']}")

# Analyze learning rates
print('\n=== Learning Rate Analysis ===')
lr_perfs = [(r['learning_rate'], r['accuracy'], r['name']) for r in runs_data if r['learning_rate']]
lr_perfs.sort(key=lambda x: x[1], reverse=True)

print('\nTop 2 Learning Rates by Best Performance:')
seen_lrs = set()
top_lrs = []
for lr, acc, name in lr_perfs:
    if lr not in seen_lrs:
        top_lrs.append((lr, acc, name))
        seen_lrs.add(lr)
    if len(top_lrs) >= 2:
        break

for i, (lr, acc, name) in enumerate(top_lrs[:2]):
    print(f"{i+1}. LR = {lr:.5f}: {acc:.1f}% ({name})")

# Group by unique learning rates
lr_groups = defaultdict(list)
for lr, acc, name in lr_perfs:
    lr_groups[lr].append((acc, name))

print('\nAll Learning Rates Performance:')
lr_summary = []
for lr, runs in lr_groups.items():
    avg_acc = sum(acc for acc, _ in runs) / len(runs)
    best_acc = max(acc for acc, _ in runs)
    lr_summary.append((lr, avg_acc, best_acc, len(runs)))

lr_summary.sort(key=lambda x: x[2], reverse=True)  # Sort by best accuracy

for lr, avg_acc, best_acc, count in lr_summary:
    print(f"LR {lr:.5f}: {count} runs, best {best_acc:.1f}%, avg {avg_acc:.1f}%")

# Note: To get full hyperparameter analysis, we'd need to query each run's config
print('\n=== Note ===')
print('To analyze model type, pretrained, bottleneck config, dropout, etc.')
print('we need to query each run\'s config individually.')
print('\nFrom the top run (kind-sweep-13) we know:')
print('- Model: ResNet34')
print('- Pretrained: Yes') 
print('- SNR Layer: bottleneck_128')
print('- Dropout: 0.5')
print('- Batch Size: 256')