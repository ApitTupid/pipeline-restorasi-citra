import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# ==============================
# 1. LOAD / CREATE IMAGE
# ==============================
def create_image():
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(img, (30,30), (100,100), 200, -1)
    cv2.circle(img, (180,80), 40, 150, -1)
    cv2.putText(img, 'TEST', (80,150), cv2.FONT_HERSHEY_SIMPLEX, 1, 180, 2)
    return img

# ==============================
# 2. DEGRADASI
# ==============================
def motion_blur_psf(length=15, angle=30):
    psf = np.zeros((length, length))
    center = length // 2
    angle = np.deg2rad(angle)

    x1 = int(center - (length/2)*np.cos(angle))
    y1 = int(center - (length/2)*np.sin(angle))
    x2 = int(center + (length/2)*np.cos(angle))
    y2 = int(center + (length/2)*np.sin(angle))

    cv2.line(psf, (x1,y1), (x2,y2), 1, 1)
    psf /= psf.sum()
    return psf

def gaussian_noise(img, sigma=20):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def sp_noise(img, prob=0.05):
    noisy = img.copy()
    num = int(prob * img.size)

    coords = [np.random.randint(0, i, num) for i in img.shape]
    noisy[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i, num) for i in img.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

# ==============================
# 3. RESTORASI
# ==============================
def inverse_filter(g, psf, eps=1e-3):
    G = np.fft.fft2(g)
    H = np.fft.fft2(psf, s=g.shape)

    F_hat = G / (H + eps)
    f = np.abs(np.fft.ifft2(F_hat))
    return np.clip(f,0,255).astype(np.uint8)

def wiener_filter(g, psf, K=0.01):
    G = np.fft.fft2(g)
    H = np.fft.fft2(psf, s=g.shape)

    H_conj = np.conj(H)
    F_hat = (H_conj / (np.abs(H)**2 + K)) * G

    f = np.abs(np.fft.ifft2(F_hat))
    return np.clip(f,0,255).astype(np.uint8)

def richardson_lucy(g, psf, iter=20):
    g = g.astype(np.float32)
    f = g.copy()

    psf_flip = np.flip(psf)

    for i in range(iter):
        conv = convolve2d(f, psf, mode='same')
        conv[conv == 0] = 1e-8
        ratio = g / conv
        f = f * convolve2d(ratio, psf_flip, mode='same')

    return np.clip(f,0,255).astype(np.uint8)

# ==============================
# 4. METRIK
# ==============================
def mse(img1, img2):
    return np.mean((img1 - img2)**2)

def psnr(img1, img2):
    m = mse(img1, img2)
    return 10*np.log10(255**2/m)

def ssim(img1, img2):
    C1 = (0.01*255)**2
    C2 = (0.03*255)**2

    mu1 = cv2.GaussianBlur(img1.astype(np.float32),(11,11),1.5)
    mu2 = cv2.GaussianBlur(img2.astype(np.float32),(11,11),1.5)

    sigma1 = cv2.GaussianBlur(img1**2,(11,11),1.5) - mu1**2
    sigma2 = cv2.GaussianBlur(img2**2,(11,11),1.5) - mu2**2
    sigma12 = cv2.GaussianBlur(img1*img2,(11,11),1.5) - mu1*mu2

    return np.mean((2*mu1*mu2 + C1)*(2*sigma12 + C2) /
                   ((mu1**2 + mu2**2 + C1)*(sigma1 + sigma2 + C2)))

# ==============================
# 5. MAIN PIPELINE
# ==============================
if __name__ == "__main__":

    img = cv2.imread('foto motor.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256))

    # PSF (estimasi = diketahui dari parameter)
    psf = motion_blur_psf(15, 30)

    # Variasi degradasi
    blur = cv2.filter2D(img, -1, psf)
    g1 = blur
    g2 = gaussian_noise(blur, 20)
    g3 = sp_noise(blur, 0.05)

    # Restorasi
    inv = inverse_filter(g2, psf, eps=1e-1)
    wnr = wiener_filter(g2, psf, K=0.05)
    rl  = richardson_lucy(g2, psf, 10)

    # Evaluasi
    methods = {"Inverse":inv, "Wiener":wnr, "RL":rl}

    for name, res in methods.items():
        print(f"{name}:")
        print("PSNR:", psnr(img, res))
        print("MSE :", mse(img, res))
        print("SSIM:", ssim(img, res))
        print()

    # Visualisasi
    plt.figure(figsize=(12,8))
    plt.subplot(2,3,1); plt.imshow(img,cmap='gray'); plt.title("Original")
    plt.subplot(2,3,2); plt.imshow(g2,cmap='gray'); plt.title("Degraded")
    plt.subplot(2,3,3); plt.imshow(psf,cmap='gray'); plt.title("PSF")
    plt.subplot(2,3,4); plt.imshow(inv,cmap='gray'); plt.title("Inverse")
    plt.subplot(2,3,5); plt.imshow(wnr,cmap='gray'); plt.title("Wiener")
    plt.subplot(2,3,6); plt.imshow(rl,cmap='gray'); plt.title("RL")

    for i in range(1,7):
        plt.subplot(2,3,i).axis('off')

    plt.tight_layout()
    plt.show()