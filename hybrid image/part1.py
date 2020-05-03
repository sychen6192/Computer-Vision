#!/usr/bin/env python
# coding: utf-8

import numpy
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import math
import numpy as np
import imageio
import matplotlib.pyplot as plt
import warnings
from skimage.transform import resize
warnings.filterwarnings('ignore')

def makeGaussianFilter(n_row, n_col, sigma, highPass=True):
    if n_row % 2 == 1:
        center_x = int(n_row/2) + 1 
    else:
        center_x = int(n_row/2)
    if n_col % 2 == 1:
        center_y = int(n_col/2) + 1
    else:
        center_y =int(n_col/2)
            
    def gaussian(i, j):
        coefficient = math.exp(-1.0 * ((i - center_x)**2 + (j - center_y)**2) / (2 * sigma**2))
        return 1 - coefficient if highPass else coefficient

    return numpy.array([[gaussian(i,j) for j in range(n_col)] for i in range(n_row)])

def idealFilter(n_row, n_col, sigma, highPass=True):
    if n_row % 2 == 1:
        center_x = int(n_row/2) + 1 
    else:
        center_x = int(n_row/2)
    if n_col % 2 == 1:
        center_y = int(n_col/2) + 1
    else:
        center_y =int(n_col/2)
        
    def ideal(i, j):
        D = math.sqrt((i - center_x)**2 + (j - center_y)**2)
        if highPass:
            return 0 if D <= sigma else 1
        else:
            return 1 if D <= sigma else 0

    return numpy.array([[ideal(i,j) for j in range(n_col)] for i in range(n_row)])  

def DFT(Matrix_img, sigma, isHigh, real):
    # Compute Fourier transform of input image
    shiftedDFT = fftshift(fft2(Matrix_img))
    # mutiply F by a filter function H(u, v)
    filteredDFT = shiftedDFT * makeGaussianFilter(Matrix_img.shape[0], Matrix_img.shape[1], sigma, highPass=isHigh)
    # inverse
    res = ifft2(ifftshift(filteredDFT))
    
    if(real):
        return numpy.real(res)
    else:
        return numpy.imag(res)

def DFT_ideal(Matrix_img, sigma, isHigh, real):
    # Compute Fourier transform of input image
    shiftedDFT = fftshift(fft2(Matrix_img))
    # mutiply F by a filter function H(u, v)
    filteredDFT = shiftedDFT * idealFilter(Matrix_img.shape[0], Matrix_img.shape[1], sigma, highPass=isHigh)
    # inverse
    res = ifft2(ifftshift(filteredDFT))
    
    if(real):
        return numpy.real(res)
    else:
        return numpy.imag(res)

girl = imageio.imread("./girl.jpg", as_gray=True);

view = imageio.imread("./view.png", as_gray=True);
view = resize(view, (1080, 1080))

plt.imshow(view, cmap='gray')

plt.imshow(girl, cmap='gray')
plt.show(plt.imshow(DFT_ideal(girl, 15, isHigh=True, real=True), cmap='gray'))
