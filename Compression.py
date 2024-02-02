import cv2
import numpy as np
from scipy.fft import fft2, ifft2

def compress_image(image_path, compression_ratio):
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (600, 600))

    # Dzielenie obrazów na składowe RGB
    blue_channel, green_channel, red_channel = cv2.split(original_image)

    # Przeprowadzenie transoformacji fouriera dla każdego koloru
    blue_channel_fft = fft2(blue_channel)
    green_channel_fft = fft2(green_channel)
    red_channel_fft = fft2(red_channel)

    # Określ ilość współczynników do zachowania (zgodnie z kompresją)
    num_coefficients_to_keep = int(np.prod(original_image.shape) * (1 - compression_ratio))

    # Upewnij się, że num_coefficients_to_keep nie przekracza liczby współczynników w danym kanale
    num_coefficients_to_keep = min(num_coefficients_to_keep, blue_channel.size - 1)

    # Ustaw współczynniki poza zakresem do usunięcia
    flat_blue_fft = blue_channel_fft.flatten()
    flat_blue_fft[np.argsort(np.abs(flat_blue_fft))[:-num_coefficients_to_keep]] = 0
    blue_channel_fft = flat_blue_fft.reshape(blue_channel_fft.shape)

    flat_green_fft = green_channel_fft.flatten()
    flat_green_fft[np.argsort(np.abs(flat_green_fft))[:-num_coefficients_to_keep]] = 0
    green_channel_fft = flat_green_fft.reshape(green_channel_fft.shape)

    flat_red_fft = red_channel_fft.flatten()
    flat_red_fft[np.argsort(np.abs(flat_red_fft))[:-num_coefficients_to_keep]] = 0
    red_channel_fft = flat_red_fft.reshape(red_channel_fft.shape)

    # Przeprowadzenie odwrotnej transformacji fouriera
    compressed_blue_channel = np.abs(ifft2(blue_channel_fft)).astype(np.uint8)
    compressed_green_channel = np.abs(ifft2(green_channel_fft)).astype(np.uint8)
    compressed_red_channel = np.abs(ifft2(red_channel_fft)).astype(np.uint8)

    # Ponowne złączenie składowych w cały obraz
    compressed_image = cv2.merge([compressed_blue_channel, compressed_green_channel, compressed_red_channel])

    # Sprawdzenie czy wartosci są pomiędzy 0 a 255
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)

    return compressed_image

original_image_path = "windows.jpg"
compression_ratio = 0.99
compressed_image_result = compress_image(original_image_path, compression_ratio)

cv2.imshow("Original Image", cv2.resize(cv2.imread(original_image_path), (600,600)))
cv2.imshow("Compressed Image", compressed_image_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
