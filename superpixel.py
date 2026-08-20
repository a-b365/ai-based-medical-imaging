import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float


img_array = imread("notebooks/xray_text_removed.jpg")

segments_slic = slic(img_array, n_segments=250, compactness=10, sigma=1, start_label=1)
print(f"SLIC number of segments: {len(np.unique(segments_slic))}")

fig, ax =  plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(mark_boundaries(img_array, segments_slic))
ax.set_title("SLIC")
ax.set_axis_off()

plt.tight_layout()
plt.show()