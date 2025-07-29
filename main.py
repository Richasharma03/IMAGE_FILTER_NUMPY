from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from filters import grayscale_filter, blur_filter, edge_filter

# Load image
img = Image.open("input_image.jpg").convert("RGB")
img_array = np.array(img)

# Show original
plt.imshow(img_array)
plt.title("Original Image")
plt.axis("off")
plt.show()

# Apply filters
gray = grayscale_filter(img_array)
blur = blur_filter(img_array)
edge = edge_filter(img_array)

# Show outputs
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")
plt.show()

plt.imshow(blur)
plt.title("Blur")
plt.axis("off")
plt.show()

plt.imshow(edge, cmap="gray")
plt.title("Edge Detection")
plt.axis("off")
plt.show()
