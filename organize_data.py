#!/usr/bin/env python3
"""
Data Organization Helper Script
Helps organize dorsal hand vein dataset into the correct structure
"""
import os
import shutil
from pathlib import Path


def check_and_organize_data():
    """Check data organization and help organize if needed"""
    
    print("="*70)
    print("Data Organization Helper")
    print("="*70)
    print()
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    print()
    
    # Look for data directories
    possible_data_dirs = [
        'dorsalhandveins',
        'dorsal_hand_veins',
        'DorsalHandVeins',
        'data',
        'dataset'
    ]
    
    found_dirs = []
    for dir_name in possible_data_dirs:
        dir_path = current_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            found_dirs.append(dir_path)
            print(f"✓ Found directory: {dir_path}")
            
            # Count .tif files
            tif_files = list(dir_path.rglob("*.tif"))
            print(f"  - Contains {len(tif_files)} .tif files")
    
    print()
    
    if not found_dirs:
        print("⚠ No data directories found!")
        print()
        print("Expected data structure:")
        print("  data/")
        print("  ├── DorsalHandVeins_DB1/")
        print("  │   └── person_*.tif files")
        print("  └── DorsalHandVeins_DB2/")
        print("      └── person_*.tif files")
        return
    
    # Check if data directory already exists with correct structure
    data_dir = current_dir / 'data'
    db1_dir = data_dir / 'DorsalHandVeins_DB1'
    db2_dir = data_dir / 'DorsalHandVeins_DB2'
    
    if db1_dir.exists() or db2_dir.exists():
        print("✓ Data already organized correctly!")
        print()
        if db1_dir.exists():
            db1_files = list(db1_dir.glob("*.tif"))
            print(f"  DB1: {len(db1_files)} images in {db1_dir}")
        if db2_dir.exists():
            db2_files = list(db2_dir.glob("*.tif"))
            print(f"  DB2: {len(db2_files)} images in {db2_dir}")
        return
    
    # Suggest organization
    print("Organizing data...")
    print()
    
    # Create data directory
    data_dir.mkdir(exist_ok=True)
    
    # Look for database folders
    for found_dir in found_dirs:
        # Check for DB1 and DB2 subdirectories
        for item in found_dir.iterdir():
            if item.is_dir():
                tif_files = list(item.glob("*.tif"))
                if tif_files:
                    print(f"Found {len(tif_files)} .tif files in: {item}")
                    
                    # Try to determine if it's DB1 or DB2
                    sample_file = tif_files[0].name
                    if 'db1' in sample_file.lower():
                        target = db1_dir
                        print(f"  → Identified as Database 1")
                    elif 'db2' in sample_file.lower():
                        target = db2_dir
                        print(f"  → Identified as Database 2")
                    else:
                        # Check if it's already named correctly
                        if 'db1' in item.name.lower():
                            target = db1_dir
                        elif 'db2' in item.name.lower():
                            target = db2_dir
                        else:
                            print(f"  → Cannot determine database version")
                            continue
                    
                    # Copy or move files
                    print(f"  → Moving to: {target}")
                    target.mkdir(parents=True, exist_ok=True)
                    
                    for tif_file in tif_files:
                        shutil.copy2(tif_file, target / tif_file.name)
                    
                    print(f"  ✓ Copied {len(tif_files)} files")
                    print()
    
    # Final verification
    print()
    print("="*70)
    print("Verification")
    print("="*70)
    
    if db1_dir.exists():
        db1_count = len(list(db1_dir.glob("*.tif")))
        print(f"✓ Database 1: {db1_count} images")
    else:
        print("✗ Database 1: Not found")
    
    if db2_dir.exists():
        db2_count = len(list(db2_dir.glob("*.tif")))
        print(f"✓ Database 2: {db2_count} images")
    else:
        print("✗ Database 2: Not found")
    
    print()
    print("Data organization complete!")
    print()
    print("Next step: Run 'python main.py --mode train' to start training")


if __name__ == "__main__":
    try:
        check_and_organize_data()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
