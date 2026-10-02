import cv2
import numpy as np

def zoom_image(image, zoom_factor=1.2, center=None):
    h, w = image.shape[:2]

    if center is None:
        center_x, center_y = w // 2, h // 2
    else:
        center_x, center_y = center

    # Smaller crop = larger apparent zoom
    crop_w = int(w / zoom_factor)
    crop_h = int(h / zoom_factor)

    x1 = max(0, center_x - crop_w // 2)
    y1 = max(0, center_y - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    cropped = image[y1:y2, x1:x2]

    # Resize back to original dimensions
    zoomed = cv2.resize(
        cropped,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    return zoomed

def detect_chest_center(image):

    # Temporary normalization
    temp = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Smooth image
    blur = cv2.GaussianBlur(
        temp,
        (51, 51),
        0
    )

    # Threshold
    _, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological cleanup
    kernel = np.ones((15, 15), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Find body region
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        h, w = image.shape
        return w // 2, h // 2

    largest = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(largest)

    chest_center_x = x + w // 2
    chest_center_y = y + h // 2

    return chest_center_x, chest_center_y

if __name__ == "__main__":
    
    cx, cy = detect_chest_center(image)

    zoomed = zoom_image(
        image,
        zoom_factor=1.2,
        center=(cx, cy)
    )