import os
import shutil
from pathlib import Path

# Base directory
base_dir = Path('classes')

# File mappings
class_files = {
    'trainers': [
        'trainers.py',
        'trainers_with_lightning.py',
        'accelerated_trainers.py',
        'accelerated_trainers_with_lightning.py',
        'trainer_utils.py'
    ],
    'managers': [
        'checkpoint_manager.py',
        'quantization_manager.py',
        'config_manager.py'
    ],
    'others': [
        'peft_callbacks.py',
        'type_utils.py',
        'utils.py'
    ]
}

# Move files
for folder, files in class_files.items():
    for file in files:
        source = base_dir / file
        dest = base_dir / folder / file
        if source.exists():
            print(f"Moving {file} to {folder}/")
            shutil.move(str(source), str(dest))
        else:
            print(f"Source file not found: {source}")

print("\nFile reorganization complete!") 
