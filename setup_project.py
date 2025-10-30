#!/usr/bin/env python3
"""
Setup script for Human Action Recognition Project
Automatically configures paths and environment for any machine
"""

import os
import sys
import json
from pathlib import Path

def setup_project():
    """
    One-click project setup for any environment
    """
    print("🚀 HUMAN ACTION RECOGNITION - PROJECT SETUP")
    print("="*60)
    
    # Add src to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    try:
        from environment_config import setup_environment
        from path_utils import get_project_root, resolve_path
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure you're running this from the project root directory")
        return False
    
    # Setup environment
    print("🔍 Detecting environment...")
    config = setup_environment()
    config.print_summary()
    
    # Detect project structure
    print("\n📁 Detecting project structure...")
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    
    # Validate critical paths
    critical_paths = {
        'processed_data': 'processed_data_auto',
        'dataset_json': 'processed_data_auto/dataset.json',
        'source_code': 'Source/src',  # Fixed: removed duplicate Human_action/
        'models_dir': 'outputs'
    }
    
    print("\n🔍 Validating project structure...")
    all_good = True
    
    for name, rel_path in critical_paths.items():
        abs_path = resolve_path(rel_path, must_exist=False)
        exists = os.path.exists(abs_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {abs_path}")
        
        if not exists and name in ['processed_data', 'source_code']:
            all_good = False
    
    # Create missing directories
    if config.get('path_resolution.auto_create_dirs', True):
        print("\n📁 Creating missing directories...")
        dirs_to_create = ['outputs', 'logs', '.cache']
        for dir_name in dirs_to_create:
            dir_path = os.path.join(project_root, dir_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"  ✅ Created: {dir_path}")
    
    # Validate dataset
    dataset_path = resolve_path('processed_data_auto/dataset.json', must_exist=False)
    if os.path.exists(dataset_path):
        print("\n📊 Validating dataset...")
        try:
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
            
            print(f"  📊 Total frames: {len(dataset)}")
            
            # Check sample paths
            sample_count = min(10, len(dataset))
            missing_files = 0
            
            for i in range(sample_count):
                item = dataset[i]
                for path_key in ['rgb_path', 'depth_path', 'pose_path', 'mask_path']:
                    if path_key in item:
                        file_path = resolve_path(item[path_key], must_exist=False)
                        if not os.path.exists(file_path):
                            missing_files += 1
            
            if missing_files == 0:
                print(f"  ✅ Sample validation passed ({sample_count} samples checked)")
            else:
                print(f"  ⚠️  Found {missing_files} missing files in sample")
                
        except Exception as e:
            print(f"  ❌ Dataset validation failed: {e}")
    else:
        print(f"\n⚠️  Dataset not found: {dataset_path}")
        print("Please run data processing first!")
    
    # Generate run commands
    print("\n🚀 READY TO RUN!")
    print("="*60)
    
    base_cmd = f"cd {project_root}/Source && python"
    
    print("Training commands:")
    print(f"  Stage A: {base_cmd} src/train_stageA.py --dataset_json {dataset_path} --epochs 10 --batch 8 --out_dir outputs/stageA")
    print(f"  Stage B: {base_cmd} src/train_stageB.py --dataset_json {dataset_path} --epochs 10 --batch 4 --gnn_type gcn --out_dir outputs/stageB")
    print(f"  Stage C: {base_cmd} src/train_stageC.py --dataset_json {dataset_path} --epochs 10 --batch 4 --out_dir outputs/stageC")
    print(f"  Stage D: {base_cmd} src/train_stageD.py --dataset_json {dataset_path} --epochs 5 --batch 2 --out_dir outputs/stageD")
    
    # Save setup info
    setup_info = {
        'project_root': project_root,
        'dataset_path': dataset_path,
        'environment': config.config['environment'],
        'device': config.get_device(),
        'setup_time': str(Path(__file__).stat().st_mtime),
        'paths_validated': all_good
    }
    
    setup_file = os.path.join(project_root, '.project_setup.json')
    with open(setup_file, 'w') as f:
        json.dump(setup_info, f, indent=2)
    
    print(f"\n💾 Setup info saved to: {setup_file}")
    
    # Environment variables for convenience
    print(f"\n🔧 Environment variables:")
    print(f"export PROJECT_ROOT='{project_root}'")
    print(f"export DATASET_PATH='{dataset_path}'")
    print(f"export PYTHONPATH='{project_root}/Source/src:$PYTHONPATH'")
    
    return all_good

def check_requirements():
    """Check if required packages are installed"""
    print("\n📦 Checking requirements...")
    
    required_packages = [
        ('torch', 'torch'), 
        ('torchvision', 'torchvision'), 
        ('numpy', 'numpy'), 
        ('opencv-python', 'cv2'), # Đã sửa: kiểm tra gói 'cv2'
        ('mediapipe', 'mediapipe'), 
        ('tqdm', 'tqdm'), 
        ('tensorboard', 'tensorboard')
    ]
    
    missing = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name}")
            missing.append(package_name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

if __name__ == "__main__":
    print("Starting project setup...")
    
    # Check requirements first
    if not check_requirements():
        print("\n❌ Please install missing packages first!")
        sys.exit(1)
    
    # Run setup
    success = setup_project()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("You can now run training scripts.")
    else:
        print("\n⚠️  Setup completed with warnings.")
        print("Some paths may need manual verification.")
    
    print("\n" + "="*60)
