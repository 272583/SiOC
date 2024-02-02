import numpy as np
import cv2

def add_noise(image, noise_level):
    noisy_image = image + noise_level * np.random.normal(size=image.shape)
    return np.clip(noisy_image, 0, 1)

def compute_2d_fft(image):
    return np.fft.fftshift(np.fft.fft2(image))

def compute_2d_ifft(fourier_image):
    return np.abs(np.fft.ifft2(np.fft.ifftshift(fourier_image)))

def apply_filter(fourier_image, radius):
    im_height, im_width = fourier_image.shape[0:2]
    im_y, im_x = np.ogrid[:im_height, :im_width]
    image_center = (im_height / 2, im_width / 2)

    low_pass_mask = (im_y - image_center[0]) ** 2 + (im_x - image_center[1]) ** 2 <= radius ** 2

    filtered_fourier = np.zeros_like(fourier_image, dtype=complex)
    for i in range(3):
        filtered_fourier[:, :, i] = fourier_image[:, :, i] * low_pass_mask

    return filtered_fourier

image = cv2.imread('windows.jpg')

input_image = image / 255.0

noise_ratio = 0.2
noisy_image = add_noise(input_image, noise_ratio)

fourier_image = np.zeros_like(noisy_image, dtype=complex)
for i in range(3):
    fourier_image[:, :, i] = compute_2d_fft(noisy_image[:, :, i])

borderline_radius = 45
filtered_fourier = apply_filter(fourier_image, borderline_radius)

denoised_image = np.zeros_like(noisy_image)
for i in range(3):
    denoised_image[:, :, i] = compute_2d_ifft(filtered_fourier[:, :, i])


# Display images using cv2.imshow
cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow('Denoised Image', denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
