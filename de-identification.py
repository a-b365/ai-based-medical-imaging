import os
import shutil
from pathlib import Path

# Root directory containing the nested folders
SOURCE_ROOT = r"Chest X-rays (Dhulikhel Hospital by Amir) February 20 2026"

# Destination directory where all DICOM files will be moved
DESTINATION = r"Extracted from February 20 2026"

# Create destination folder if it doesn't exist
os.makedirs(DESTINATION, exist_ok=True)

moved_count = 0

for root, dirs, files in os.walk(SOURCE_ROOT):
    for file in files:
        if file.lower().endswith(".dcm"):
            src_file = os.path.join(root, file)

            # Handle duplicate filenames
            dst_file = os.path.join(DESTINATION, file)
            # if os.path.exists(dst_file):
            #     stem = Path(file).stem
            #     suffix = Path(file).suffix
            #     counter = 1

            #     while os.path.exists(dst_file):
            #         dst_file = os.path.join(
            #             DESTINATION,
            #             f"{stem}_{counter}{suffix}"
            #         )
            #         counter += 1

            shutil.move(src_file, dst_file)
            moved_count += 1
            print(f"Moved: {src_file} -> {dst_file}")

print(f"\nTotal DICOM files moved: {moved_count}")