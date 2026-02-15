#!/usr/bin/env python3
"""
Simple Training Starter Script
Performs pre-flight checks and starts training
"""
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print(f"✗ Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    required = ['torch', 'numpy', 'matplotlib', 'opencv-python', 'scikit-learn']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  - Device: {torch.cuda.get_device_name(0)}")
            print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠ CUDA not available (will use CPU - training will be slower)")
            return True
    except:
        print("✗ Cannot check CUDA")
        return False


def check_data():
    """Check if data is organized correctly"""
    print("\nChecking data...")
    data_dir = Path('data')
    
    if not data_dir.exists():
        print("✗ 'data/' directory not found")
        print("\nRun: python organize_data.py")
        return False
    
    db1 = data_dir / 'DorsalHandVeins_DB1'
    db2 = data_dir / 'DorsalHandVeins_DB2'
    
    total_images = 0
    
    if db1.exists():
        db1_count = len(list(db1.glob("*.tif")))
        print(f"✓ Database 1: {db1_count} images")
        total_images += db1_count
    else:
        print("⚠ Database 1 not found")
    
    if db2.exists():
        db2_count = len(list(db2.glob("*.tif")))
        print(f"✓ Database 2: {db2_count} images")
        total_images += db2_count
    else:
        print("⚠ Database 2 not found")
    
    if total_images == 0:
        print("✗ No images found")
        print("\nOrganize your data with: python organize_data.py")
        return False
    
    print(f"✓ Total images: {total_images}")
    return True


def check_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    dirs = [
        'checkpoints',
        'results/generated',
        'results/xai/gradcam',
        'results/xai/shap',
        'results/xai/lime',
        'results/metrics',
        'results/fairness',
        'results/comparisons',
        'results/training_progress',
        'logs'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ All directories created")
    return True


def start_training():
    """Start the training process"""
    print("\n" + "="*70)
    print("Starting Training...")
    print("="*70 + "\n")
    
    # Run main.py with train mode
    try:
        subprocess.run([sys.executable, 'main.py', '--mode', 'train'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        return False
    
    return True


def main():
    """Main function"""
    print("="*70)
    print("XAI-Enhanced Vein GAN - Training Starter")
    print("="*70 + "\n")
    
    # Run pre-flight checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("CUDA", check_cuda),
        ("Data", check_data),
        ("Directories", check_directories)
    ]
    
    for check_name, check_func in checks:
        if not check_func():
            print(f"\n✗ Pre-flight check failed: {check_name}")
            print("\nPlease fix the above issues before training.")
            sys.exit(1)
    
    print("\n" + "="*70)
    print("✓ All pre-flight checks passed!")
    print("="*70)
    
    # Ask user confirmation
    print("\nReady to start training.")
    response = input("Continue? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Training cancelled.")
        sys.exit(0)
    
    # Start training
    success = start_training()
    
    if success:
        print("\n" + "="*70)
        print("✓ Training completed successfully!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Check results in: results/training_progress/")
        print("  2. Evaluate model: python main.py --mode evaluate --checkpoint checkpoints/best_model.pth")
        print("  3. Run XAI: python main.py --mode xai --checkpoint checkpoints/best_model.pth")
    else:
        print("\n" + "="*70)
        print("✗ Training did not complete")
        print("="*70)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
