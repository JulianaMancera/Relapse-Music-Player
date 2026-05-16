"""
Verify ML Setup
Check if all components are ready for training
"""
import os
import sys
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("🐍 Python Version Check")
    version = sys.version_info
    print(f"   Version: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print("   ✓ OK (3.8+)")
        return True
    print("   ✗ FAILED (Need 3.8+)")
    return False

def check_dependencies():
    """Check if ML dependencies are installed"""
    print("\n📦 Dependency Check")
    dependencies = {
        'tensorflow': 'TensorFlow',
        'keras': 'Keras',
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'mediapipe': 'MediaPipe',
        'PIL': 'Pillow',
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"   ✓ {name}")
        except ImportError:
            print(f"   ✗ {name} NOT INSTALLED")
            all_ok = False
    
    if not all_ok:
        print("\n   To install: pip install -r requirements_ml.txt")
    
    return all_ok

def check_dataset():
    """Check if dataset exists"""
    print("\n📂 Dataset Check")
    dataset_dir = Path("datasets/asl-alphabet")
    train_dir = dataset_dir / "asl_alphabet_train"
    
    if train_dir.exists():
        # Count classes
        classes = [d for d in train_dir.iterdir() if d.is_dir()]
        print(f"   ✓ Dataset found")
        print(f"   ✓ Classes: {len(classes)}/26")
        
        # Count total images
        total_images = sum(len(list(d.glob("*"))) for d in classes)
        print(f"   ✓ Total images: {total_images}")
        
        if len(classes) == 26 and total_images > 70000:
            print("   ✓ Dataset complete!")
            return True
        else:
            print("   ⚠ Dataset incomplete. Download again.")
            return False
    else:
        print("   ✗ Dataset not found")
        print("   Run: python download_dataset.py")
        return False

def check_models_dir():
    """Check if models directory exists"""
    print("\n📁 Models Directory Check")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"   ✓ Models directory ready: {models_dir.absolute()}")
    return True

def check_gpu():
    """Check if GPU is available"""
    print("\n🖥️ GPU Check")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   ✓ GPU found: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"      - {gpu}")
            return True
        else:
            print("   ⚠ No GPU detected (CPU training will be slower)")
            return False
    except Exception as e:
        print(f"   ✗ GPU check failed: {e}")
        return False

def main():
    print("="*60)
    print("ML SETUP VERIFICATION")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Models Directory", check_models_dir),
        ("Dataset", check_dataset),
        ("GPU", check_gpu),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Checks passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ ALL CHECKS PASSED!")
        print("You're ready to train: python train_asl_model.py")
    else:
        print("\n⚠ Some checks failed. See above for details.")
        failed = [name for name, result in results if not result]
        print(f"Failed: {', '.join(failed)}")
    
    print("="*60)

if __name__ == "__main__":
    main()
