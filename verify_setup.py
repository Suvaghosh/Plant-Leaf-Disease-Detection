"""
Setup Verification Script
=========================
Run this script to verify that your environment is set up correctly
before training the model or running the web application.
"""

import os
import sys

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = {
        'tensorflow': 'TensorFlow',
        'flask': 'Flask',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'sklearn': 'scikit-learn'
    }
    
    missing = []
    for module, name in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {name} is installed")
        except ImportError:
            print(f"❌ {name} is NOT installed")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Install them using: pip install -r requirements.txt")
        return False
    return True

def check_dataset():
    """Check if dataset path exists."""
    dataset_path = r"C:\Users\subha\OneDrive\Desktop\Dataset"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        print("   Please ensure your dataset is in the correct location")
        return False
    
    # Check if it's a directory
    if not os.path.isdir(dataset_path):
        print(f"❌ Dataset path is not a directory: {dataset_path}")
        return False
    
    # Check for subdirectories (classes)
    subdirs = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    if len(subdirs) == 0:
        print(f"⚠️  Dataset directory found but no class subdirectories detected")
        print("   Expected structure: Dataset/class_name/images.jpg")
        return False
    
    print(f"✅ Dataset found at: {dataset_path}")
    print(f"   Found {len(subdirs)} class directories")
    return True

def check_directories():
    """Check if required project directories exist."""
    required_dirs = ['model', 'static', 'templates', 'static/uploads']
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"❌ Directory missing: {dir_path}")
            all_exist = False
    
    return all_exist

def check_files():
    """Check if required project files exist."""
    required_files = [
        'train.py',
        'predict.py',
        'app.py',
        'requirements.txt',
        'README.md',
        'templates/index.html',
        'templates/result.html',
        'static/style.css'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ File missing: {file_path}")
            all_exist = False
    
    return all_exist

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Plant Leaf Disease Detection - Setup Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Directories", check_directories),
        ("Project Files", check_files),
        ("Dataset", check_dataset),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 40)
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    all_passed = all(result for _, result in results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print()
    if all_passed:
        print("🎉 All checks passed! You're ready to start.")
        print("\nNext steps:")
        print("1. Train the model: python train.py")
        print("2. Run the web app: python app.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

