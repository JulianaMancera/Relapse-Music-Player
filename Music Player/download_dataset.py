"""
Download and prepare ASL Alphabet dataset from Kaggle
Run this first to set up your training data
"""
import os
import shutil
import zipfile
import requests
from pathlib import Path

def download_asl_dataset():
    """
    Option 1: Manual download from Kaggle
    1. Go to: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
    2. Download the zip file
    3. Extract to: Music Player/datasets/asl-alphabet/
    """
    print("=" * 60)
    print("ASL ALPHABET DATASET SETUP")
    print("=" * 60)
    
    dataset_dir = Path("datasets/asl-alphabet")
    train_dir = dataset_dir / "asl_alphabet_train"
    
    if train_dir.exists():
        print(f"✓ Dataset already exists at: {train_dir}")
        return True
    
    print("\nDataset not found. Please follow these steps:")
    print("\n1. Go to: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
    print("2. Click 'Download' button")
    print("3. Extract the zip file to: Music Player/datasets/")
    print("\nExpected structure after extraction:")
    print("   Music Player/")
    print("   ├── datasets/")
    print("   │   └── asl-alphabet/")
    print("   │       ├── asl_alphabet_train/")
    print("   │       │   ├── A/")
    print("   │       │   ├── B/")
    print("   │       │   └── ...")
    print("   │       └── asl_alphabet_test/")
    print("   └── ...")
    
    print("\nAlternatively, you can use API authentication:")
    print("   pip install kaggle")
    print("   kaggle datasets download -d grassknoted/asl-alphabet")
    
    return False

if __name__ == "__main__":
    download_asl_dataset()
