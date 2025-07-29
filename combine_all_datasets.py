#!/usr/bin/env python3
"""
Combine all financial training datasets into one comprehensive file
"""

import json
import os
from datetime import datetime
from pathlib import Path

def load_dataset(file_path):
    """Load a dataset from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different dataset structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if 'training_data' in data:
                return data['training_data']
            elif 'data' in data:
                return data['data']
            elif 'samples' in data:
                return data['samples']
            else:
                # Try to find a list in the dict
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        # Check if it looks like training data
                        if isinstance(value[0], dict) and ('instruction' in value[0] or 'input' in value[0] or 'output' in value[0]):
                            return value
                
                print(f"Warning: Could not find training data in {file_path}")
                return []
        else:
            print(f"Warning: Unexpected data type in {file_path}")
            return []
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def combine_datasets():
    """Combine all datasets into one comprehensive dataset"""
    
    # For RunPod environment - use /workspace path
    base_path = Path("/workspace/clearledgr-runpod/data/financial_training_data")
    
    # Define all dataset files to combine
    dataset_files = [
        # Major datasets - these will be uploaded separately due to size
        base_path / "ULTIMATE_FINANCIAL_DATASET.json",  # Current 17K samples
        
        # Individual datasets that might be available
        base_path / "consolidated_financial_dataset.json",
        base_path / "comprehensive_financial_dataset.json",
        base_path / "comprehensive_synthetic_dataset.json",
        base_path / "enhanced_user_dataset.json",
        base_path / "ghana_reconciliation_dataset.json",
        base_path / "ghana_transaction_analysis_dataset.json",
        base_path / "financial_analysis_dataset.json",
        base_path / "financial_instruction_data.json",
        base_path / "enhanced_financial_instructions.json",
        base_path / "user_provided_dataset.json",
        base_path / "accounting_standards_dataset.json",
    ]
    
    combined_samples = []
    source_stats = {}
    
    print("🔄 Starting dataset combination on RunPod...")
    
    for file_path in dataset_files:
        if file_path.exists():
            print(f"📁 Loading {file_path.name}...")
            samples = load_dataset(file_path)
            
            if samples:
                combined_samples.extend(samples)
                source_stats[file_path.name] = len(samples)
                print(f"   ✅ Added {len(samples):,} samples")
            else:
                print(f"   ⚠️  No samples found")
        else:
            print(f"   ❌ File not found: {file_path}")
    
    # Create the comprehensive dataset
    comprehensive_dataset = {
        "metadata": {
            "name": "Clearledgr Complete Financial AI Training Dataset",
            "version": "1.0.0",
            "created": datetime.now().isoformat(),
            "total_samples": len(combined_samples),
            "total_sources": len(source_stats),
            "description": "Complete combined dataset with all available financial training data",
            "source_breakdown": source_stats
        },
        "training_data": combined_samples
    }
    
    # Save the combined dataset
    output_path = base_path / "RUNPOD_COMBINED_DATASET.json"
    print(f"\n💾 Saving combined dataset to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Dataset combination complete!")
    print(f"📊 Total samples: {len(combined_samples):,}")
    print(f"📁 Total sources: {len(source_stats)}")
    print(f"💾 Output file: {output_path}")
    
    # Calculate file size
    if output_path.exists():
        file_size_gb = output_path.stat().st_size / (1024*1024*1024)
        print(f"📏 File size: {file_size_gb:.2f} GB")
    
    return output_path, len(combined_samples)

if __name__ == "__main__":
    combine_datasets()
