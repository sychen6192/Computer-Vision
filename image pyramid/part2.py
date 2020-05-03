import numpy
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import math
import numpy as np
import imageio
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

cat = imageio.imread("Jagermeister.jpg", as_gray=True)


def makeGaussianFilter(n_row, n_col, sigma, highPass=True):
    if n_row % 2 == 1:
        center_x = int(n_row / 2) + 1
    else:
        center_x = int(n_row / 2)
    if n_col % 2 == 1:
        center_y = int(n_col / 2) + 1
    else:
        center_y = int(n_col / 2)

    def gaussian(i, j):
        coefficient = math.exp(-1.0 * ((i - center_x) ** 2 + (j - center_y) ** 2) / (2 * sigma ** 2))
        return 1 - coefficient if highPass else coefficient

    return numpy.array([[gaussian(i, j) for j in range(n_col)] for i in range(n_row)])


def DFT(Matrix_img, sigma, state):
    # Compute Fourier transform of input image
    shiftedDFT = fftshift(fft2(Matrix_img))
    # mutiply F by a filter function H(u, v)
    filteredDFT = shiftedDFT * makeGaussianFilter(Matrix_img.shape[0], Matrix_img.shape[1], sigma, highPass=state)
    # inverse
    res = ifft2(ifftshift(filteredDFT))

    return numpy.real(res)


def laplacian(img_matrix):
    def expand(img_matrix):
        return np.repeat(np.repeat(img_matrix, 2, axis=1), 2, axis=0)
    sub = DFT(img_matrix, 20, False)[::2, ::2]
    result = img_matrix - expand(sub)[:img_matrix.shape[0], :]
    return result


def imagePyramid(img_matrix, sigma, num=5, state=False):
    total_layer = []
    total_layer.append(img_matrix)
    plt.figure(figsize=(8, 8))
    plt.subplot(251)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_matrix, cmap="gray")
    plt.title('Layer_{}'.format(num))
    for i in range(num-1):
        img_matrix = DFT(img_matrix, sigma, state)[::2, ::2] # smooth -> subsampling
        plt.subplot(252+i)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img_matrix, cmap="gray")
        total_layer.append(img_matrix)
        plt.title("Layer_{}".format(num-i-1))

    return total_layer


def laplacianPyramid(gaussian_list):
    plt.subplot(256)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(gaussian_list[-1], cmap="gray")
    plt.title("Layer_{}".format(1))
    def expand(img_matrix):
        return np.repeat(np.repeat(img_matrix, 2, axis=1), 2, axis=0)
    for i in range(len(gaussian_list)-1):
        result = gaussian_list[i] - expand(gaussian_list[i+1])[:gaussian_list[i].shape[0],:gaussian_list[i].shape[1]]
        plt.subplot(2,5,7+i)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(result, cmap="gray")
        plt.title("Layer_{}".format(i+2))


total_layer = imagePyramid(cat, 20)
laplacianPyramid(total_layer)
plt.show()
