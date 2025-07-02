#!/usr/bin/env python3
"""
Check wandb run metrics and look for F1 scores and confusion matrices
"""

import wandb
import json

def main():
    # Initialize wandb API
    api = wandb.Api()
    
    # Get the specific run
    run_path = 'shamoons/modulation-classification/nl5pyzud'
    print(f"Fetching run: {run_path}")
    
    try:
        run = api.run(run_path)
        
        print('\n=== Run Summary ===')
        print(f'Run name: {run.name}')
        print(f'State: {run.state}')
        print(f'Run ID: {run.id}')
        
        # Get config
        print('\n=== Configuration ===')
        config = dict(run.config)
        for key, value in config.items():
            print(f'{key}: {value}')
        
        # Get recent history
        history = run.history()
        if not history.empty:
            print('\n=== Recent Training Progress ===')
            # Get last 5 epochs
            recent = history.tail(5)
            
            # Check available columns
            print("Available metrics:", list(history.columns)[:20])
            
            # Display key metrics if available
            key_cols = ['epoch', 'val_combined_accuracy', 'val_modulation_accuracy', 
                       'val_snr_accuracy', 'val_loss', 'train_combined_accuracy']
            available_cols = [col for col in key_cols if col in history.columns]
            
            if available_cols:
                print("\nRecent epochs:")
                print(recent[available_cols].to_string())
        
        # Check for F1 or confusion matrix data in summary
        print('\n=== Checking for F1/Confusion Matrix Data ===')
        summary = dict(run.summary)
        
        f1_metrics = {k: v for k, v in summary.items() if 'f1' in k.lower()}
        if f1_metrics:
            print("F1 Scores found:")
            for key, value in f1_metrics.items():
                print(f'  {key}: {value}')
        else:
            print("No F1 scores found in summary metrics")
        
        # Check files
        print('\n=== Checking Run Files ===')
        files = list(run.files())
        relevant_files = [f for f in files if any(term in f.name.lower() for term in ['f1', 'confusion', 'matrix', 'csv'])]
        
        if relevant_files:
            print(f"Found {len(relevant_files)} relevant files:")
            for f in relevant_files[:10]:
                print(f'  - {f.name}')
        else:
            print("No F1 or confusion matrix files found")
            
    except Exception as e:
        print(f"Error accessing run: {e}")

if __name__ == "__main__":
    main()