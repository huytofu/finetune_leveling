import os
from pathlib import Path

# Base directory structure
base_dir = Path('NLP/classes')

# Class directories to create
class_dirs = [
    'trainers',
    'managers',
    'others'
]

# Create directories
for dir_path in class_dirs:
    full_path = base_dir / dir_path
    full_path.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {full_path}")

print("\nClass directory structure created successfully!") 
