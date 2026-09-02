import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def preprocess_cxr_dicom(dicom_path):

    # Read DICOM
    ds = pydicom.dcmread(dicom_path)

    # Extract pixel array
    image = ds.pixel_array.astype(np.float32)

    # Apply rescale transformation
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))

    image = image * slope + intercept


    # Handle MONOCHROME1
    if ds.PhotometricInterpretation == "MONOCHROME1":
        image = np.max(image) - image


    # Percentile clipping
    lower = np.percentile(image, 1)
    upper = np.percentile(image, 99)

    image = np.clip(image, lower, upper)


    # Normalize to 0-255
    image = (
        (image - image.min()) /
        (image.max() - image.min())
    )

    image = (image * 255).astype(np.uint8)


    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    image = clahe.apply(image)


    # Gamma correction
    gamma = 1.2

    image = np.power(
        image / 255.0,
        gamma
    )

    image = (image * 255).astype(np.uint8)


    return image


def preprocess_cxr_jpeg(jpeg_path):

    # Read JPEG
    image = Image.open(jpeg_path)

    # Percentile clipping
    lower = np.percentile(image, 1)
    upper = np.percentile(image, 99)

    image = np.clip(image, lower, upper)


    # Normalize to 0-255
    image = (
        (image - image.min()) /
        (image.max() - image.min())
    )

    image = (image * 255).astype(np.uint8)


    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    image = clahe.apply(image)


    # Gamma correction
    gamma = 1.2

    image = np.power(
        image / 255.0,
        gamma
    )

    image = (image * 255).astype(np.uint8)


    return image



if __name__ == "__main__":

    image = preprocess_cxr_jpeg(r"Chest X-rays Jpeg/1.2.840.113564.54.20260201094458264.8004.jpg")

    # Image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("DICOM image after preprocessing")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    pixels = image.ravel()

    # Histogram
    plt.hist(
        pixels,
        bins=256,
        range=(
            pixels.min(),
            pixels.max()
        )
    )

    plt.title("Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.25)
    plt.show()
