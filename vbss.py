import cv2
import numpy as np
import math


def normalize(np_array:np.array) -> np.array:
    return (np_array - np_array.min())/(np_array.max() - np_array.min())

def normalize_save(float_image:np.ndarray, savename:str) -> None:
    normalized_image = normalize(float_image)*255.0
    if len(normalized_image.shape) == 3:
        normalized_image = cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR)
    normalized_image = np.clip(normalized_image, 0, 255).astype(np.uint8)
    print(normalized_image)
    cv2.imwrite(savename, normalized_image)

def valid_convolve(xx, size):
    b = np.ones(size)/size
    xx_mean = np.convolve(xx, b, mode="same")

    n_conv = math.ceil(size/2)

    xx_mean[0] *= size/n_conv
    for i in range(1, n_conv):
        xx_mean[i] *= size/(i+n_conv)
        xx_mean[-i] *= size/(i + n_conv - (size % 2)) 

    return xx_mean


def main():
    blur_path = '/mnt/d/results/20240522/000637_B.png'
    sharp_path = '/mnt/d/results/20240522/000637_GT.png'
    np_path = '/mnt/d/results/20240522/000637_diff.png'

    blur = cv2.imread(blur_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
    sharp = cv2.imread(sharp_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0

    # percep = np.load(np_path)
    percep = cv2.imread(np_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0

    blur_fft = np.fft.fft2(blur)
    percep_fft = np.fft.fft2(percep)
    
    # blur_fft = np.fft.fftshift(blur_fft)
    # percep_fft = np.fft.fftshift(percep_fft)
    # blur_mag  = 20*np.log(np.abs(blur_fft))
    # percep_mag = 20*np.log(np.abs(percep_fft))

    # normalize_save(blur_mag, '/mnt/d/results/20240522/000637_fft.png')
    # normalize_save(blur_mag, '/mnt/d/results/20240522/000637_percep_fft.png')
    h, w = blur_fft.shape
    blur_fft = blur_fft.reshape((1,-1))
    percep_fft = percep_fft.reshape((1,-1))
    # y = normalize(np.concatenate([blur_fft, percep_fft], axis=0))
    y = np.concatenate([blur_fft, percep_fft], axis=0)

    y_hat = np.zeros_like(y)
    for i in range(y_hat.shape[0]):
        y_hat[i] = valid_convolve(y[i], 5)
    
    p = np.matmul(y, np.conjugate(y.T.copy()))
    # p = np.matmul(y, y.T.copy())
    q = np.matmul(y_hat - y, np.conjugate((y_hat - y).T.copy()))
    # q = np.matmul(y_hat - y, (y_hat - y).T.copy())

    q_inv_p = np.matmul(np.linalg.inv(q), p)

    eigenvalues, eigenvectors = np.linalg.eig(q_inv_p)

    f = np.matmul(eigenvectors, y)
    
    print(f)
    print(f.shape)

    freq_1 = f[0].reshape((h,w))
    freq_2 = f[1].reshape((h,w))

    inv_1 = np.fft.ifft2(freq_1)
    inv_2 = np.fft.ifft2(freq_2)

    freq_1 = np.fft.fftshift(freq_1).real
    freq_2 = np.fft.fftshift(freq_2).real

    freq1_mag  = 20*np.log(np.abs(freq_1))
    freq2_mag = 20*np.log(np.abs(freq_2))

    print(inv_1)
    print(inv_1.min(), inv_1.max())
    print(inv_2)
    print(inv_2.min(), inv_2.max())
    normalize_save(freq1_mag, '/mnt/d/results/20240522/freq_1.png')
    normalize_save(freq2_mag, '/mnt/d/results/20240522/freq_2.png')
    normalize_save(inv_1, '/mnt/d/results/20240522/inv_1.png')
    normalize_save(inv_2, '/mnt/d/results/20240522/inv_2.png')





if __name__ == '__main__':
    main()