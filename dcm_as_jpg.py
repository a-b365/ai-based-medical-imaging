import os
from pathlib import Path

import cv2
import numpy as np
import pydicom

input_dir = r"D:/ai-based-medical-imaging/PID Removal from February 20 2026/"
output_dir = r"D:/ai-based-medical-imaging/Chest X-rays Jpeg/"

os.makedirs(output_dir, exist_ok=True)

for dcm_file in Path(input_dir).glob("*.dcm"):
    # Read DICOM
    ds = pydicom.dcmread(dcm_file)

    # Get pixel data
    image = ds.pixel_array.astype(np.float32)

    # Apply rescale slope/intercept if present
    slope = float(ds.get("RescaleSlope", 1))
    intercept = float(ds.get("RescaleIntercept", 0))
    image = image * slope + intercept

    # Normalize to 0-255
    image -= image.min()
    if image.max() > 0:
        image = image / image.max()
    image = (image * 255).astype(np.uint8)

    # Invert if MONOCHROME1
    if ds.PhotometricInterpretation == "MONOCHROME1":
        image = 255 - image

    # Save as JPG
    output_path = Path(output_dir) / (dcm_file.stem + ".jpg")
    cv2.imwrite(str(output_path), image)

print("Conversion completed.")