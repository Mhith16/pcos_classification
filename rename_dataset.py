import os
import shutil
from pathlib import Path
import re

def rename_images_in_directory(directory, prefix=""):
    """
    Rename all image files in a directory to a sequential pattern.
    
    Args:
        directory (str): Path to the directory containing images
        prefix (str): Optional prefix for filenames
    """
    # Create a backup directory
    backup_dir = os.path.join(directory, "original_files_backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(Path(directory).glob(f"*{ext}")))
        image_files.extend(list(Path(directory).glob(f"*{ext.upper()}")))
    
    # Skip files that are already in the desired format
    image_files = [f for f in image_files if not re.match(r'\d{5}\.', f.name)]
    
    print(f"Found {len(image_files)} images to rename in {directory}")
    
    # Sort files to ensure consistent ordering
    image_files.sort()
    
    # Rename files
    for i, file_path in enumerate(image_files, 1):
        # Get file extension
        _, ext = os.path.splitext(file_path)
        
        # Create new filename with 5-digit numbering
        new_filename = f"{prefix}{i:05d}{ext.lower()}"
        new_path = os.path.join(directory, new_filename)
        
        # Backup original file
        backup_path = os.path.join(backup_dir, file_path.name)
        shutil.copy2(file_path, backup_path)
        
        # Rename file
        os.rename(file_path, new_path)
        print(f"Renamed: {file_path.name} → {new_filename}")

def rename_pcos_dataset(base_dir="PCOS"):
    """Rename images in the PCOS dataset."""
    # Process infected directory
    infected_dir = os.path.join(base_dir, "infected")
    if os.path.exists(infected_dir):
        print(f"\nProcessing infected directory...")
        rename_images_in_directory(infected_dir)
    
    # Process noninfected directory
    noninfected_dir = os.path.join(base_dir, "noninfected")
    if os.path.exists(noninfected_dir):
        print(f"\nProcessing noninfected directory...")
        rename_images_in_directory(noninfected_dir)
    
    print("\nRenaming complete! Original files have been backed up in 'original_files_backup' directories.")

if __name__ == "__main__":
    # You can change the base directory if needed
    rename_pcos_dataset("PCOS")