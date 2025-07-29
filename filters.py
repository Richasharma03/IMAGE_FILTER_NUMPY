import numpy as np
from scipy.signal import convolve2d

# Grayscale filter
def grayscale_filter(img_array):
    grayscale = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    return grayscale.astype(np.uint8)

# Blur filter (Box blur)
def blur_filter(img_array, k=3):
    kernel = np.ones((k, k)) / (k**2)
    blurred_channels = []

    for i in range(3):
        channel = img_array[:, :, i]
        blurred = convolve2d(channel, kernel, mode="same", boundary="symm")
        blurred_channels.append(blurred)

    blurred_img = np.stack(blurred_channels, axis=2)
    return np.clip(blurred_img, 0, 255).astype(np.uint8)

# Edge detection (Sobel filter)
def edge_filter(img_array):
    gray = grayscale_filter(img_array)

    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Ix = convolve2d(gray, Kx, mode="same", boundary="symm")
    Iy = convolve2d(gray, Ky, mode="same", boundary="symm")

    G = np.sqrt(Ix**2 + Iy**2)
    return np.clip(G, 0, 255).astype(np.uint8)
