import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_atmospheric_light(image, transmission, top_percent=0.1):
    flat_image = image.reshape(-1, 3)
    flat_transmission = transmission.reshape(-1)
    num_pixels = int(flat_image.shape[0] * top_percent)
    indices = np.argsort(flat_transmission)[-num_pixels:]
    atmospheric_light = np.mean(flat_image[indices], axis=0)
    return atmospheric_light

def estimate_transmission(image, atmospheric_light, omega=0.95, window_size=15):
    norm_image = image / atmospheric_light
    transmission = 1 - omega * np.min(norm_image, axis=2)
    transmission = cv2.blur(transmission, (window_size, window_size))
    return transmission

def guided_filter(I, p, r, eps):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * I + mean_b
    return q

def recover_image(image, transmission, atmospheric_light, t_min=0.1):
    transmission = np.maximum(transmission, t_min)
    transmission = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)
    J = (image - atmospheric_light) / transmission + atmospheric_light
    J = np.clip(J, 0, 1)
    return J

def dehaze_image(image_path):
    image = cv2.imread(image_path).astype(np.float32) / 255
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Estimate transmission
    transmission = 1 - gray_image

    # Estimate atmospheric light
    atmospheric_light = get_atmospheric_light(image, transmission)

    # Refine transmission with guided filter
    transmission = guided_filter(gray_image, transmission, 60, 0.0001)

    # Recover the image
    dehazed_image = recover_image(image, transmission, atmospheric_light)

    # Convert to 8-bit image
    dehazed_image = (dehazed_image * 255).astype(np.uint8)

    return dehazed_image

def remove_salt_pepper_noise(image):
    return cv2.medianBlur(image, 5)

def apply_high_pass_filter(image):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def upscale_image(image, scale_percent=150):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

def increase_contrast(image, alpha=0.5, beta=0):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

if __name__ == '__main__':
    import sys
    try:
        image_path = sys.argv[1]
    except:
        image_path = 'a.jpg'

    dehazed_image = dehaze_image(image_path)
    dehazed_image = remove_salt_pepper_noise(dehazed_image)
    dehazed_image = apply_high_pass_filter(dehazed_image)

    # Upscale image
    dehazed_image = upscale_image(dehazed_image, scale_percent=150)

    # Increase contrast
    dehazed_image = increase_contrast(dehazed_image, alpha=0.9, beta=0)

    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Hazy Image')
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title('Dehazed Image')
    plt.imshow(cv2.cvtColor(dehazed_image, cv2.COLOR_BGR2RGB))

    plt.show()

    # Save the dehazed image
    cv2.imwrite('dehazed_image.png', dehazed_image)
