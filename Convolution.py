import cv2
import numpy as np
from matplotlib import pyplot as plt

def detect_edges(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    laplace = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, -1, 0]])

    edges_x = cv2.filter2D(gray, -1, sobel_x)
    edges_y = cv2.filter2D(gray, -1, sobel_y)
    edges_laplace = cv2.filter2D(gray, -1, laplace)

    edges_combined = np.abs(edges_x) + np.abs(edges_y) + np.abs(edges_laplace)

    return edges_combined
def blur_image(image, num_iterations):
    # Zauważyłem, że użycie 1/14 zamiast 1/16 pozwala lepiej odwzorować oryginalną jasność obrazu
    #kernel = (1/14) * np.array([[1, 2, 1],
    #                            [1, 4, 1],
    #                            [1, 2, 1]])

    # Blur jest mało zauważalny przy jednorazowym nałożeniu, aby go zwiększyć, nakładam go wielokrotnie
    # for _ in range(num_iterations):
    #    image = cv2.filter2D(image, -1, kernel)

    # Implementacja z wiekszym rozmiarem jądra
    kernel = (1 / 256) * np.array([[1, 4, 6, 4, 1],
                                   [4, 16, 24, 16, 4],
                                   [6, 24, 36, 24, 6],
                                   [4, 16, 24, 16, 4],
                                   [1, 4, 6, 4, 1]])

    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image


def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image

image = cv2.cvtColor(cv2.imread('windows.jpg'), cv2.COLOR_BGR2RGB)

edges = detect_edges(image)
blurred_image = blur_image(image, 50)
sharpened_image = sharpen_image(image)

plt.figure(figsize=(16, 6))

plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(1, 4, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')

plt.subplot(1, 4, 3)
plt.imshow(sharpened_image)
plt.title('Sharpened Image')

plt.subplot(1, 4, 4)
plt.imshow(blurred_image)
plt.title('Blurred Image')

plt.show()
