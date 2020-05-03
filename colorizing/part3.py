import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
from tqdm import tqdm

class Image_Align:
    '''
        perform normalized cross correlation 
        Args:
            target: target image
            input: template image
        Return:
            (w, h): offset of template image at top left corner with target image
    '''
    def __init__(self, target):
        self.target = target

    def __call__(self, input):
        score = float('-inf')
        for i in tqdm(range(-args.step//2, args.step//2)):
            for j in range(-args.step//2, args.step//2):
                similarity = self.ncc(self.target, np.roll(input, [i,j],axis=(0,1)))
                if similarity > score:
                    offset = [i, j]
                    score = similarity
        return offset

    def ncc(self, f, g):
        '''
            f: target image
            g: template
        '''
        g_mean = np.mean(g)
        f_mean = np.mean(f)
        
        numerator = np.sum((g - g_mean) * (f - f_mean))
        denominator = np.sqrt(np.linalg.norm(g) * np.linalg.norm(f))
        return numerator / denominator
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('path', help='path of input image')
    parser.add_argument('--step', type=int, default=20, help='number of steps start from top left corner')
    args = parser.parse_args()    

    image = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)

    # remove border
    h, w = image.shape[:2]
    image = image[int(h*0.01): int(h*0.99), int(w*0.01): int(w*0.99)]

    # divide image into three subimages
    image = image[: image.shape[0]//3*3, ...]
    h, w = image.shape[:2]
    print(h, w)

    B, G, R = np.split(image, 3)

    target = Image_Align(B)
    print("aligning G channel...")
    G_offset = target(G)
    print("aligning R channel...")
    R_offset = target(R)

    # shift the images and stack together
    G = np.roll(G, G_offset, axis=(0, 1))
    R = np.roll(R, R_offset, axis=(0, 1))

    result = np.dstack((R, G, B)).astype(np.uint8)

    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    ax.imshow(result)
    ax.axis('off')
    plt.show()
