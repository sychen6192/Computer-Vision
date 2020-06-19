import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math
from scipy import ndimage
from scipy.spatial import distance
from tqdm import tqdm



def get_tiny_images(path):
    total_pic = {}
    for doc in os.listdir(path):
        tmp = []
        for file in os.listdir(os.path.join(path, doc)):
            if file.endswith(".jpg"):
                img = cv2.imread(os.path.join(path, doc, file), cv2.IMREAD_GRAYSCALE)
                pic = cv2.resize(img, (16, 16))
                pic = (pic - np.mean(pic))/np.std(pic)
                tmp.append(pic)
        total_pic[doc] = tmp
    return total_pic

train_image_feats = get_tiny_images('./hw5_data/train/')
test_image_feats = get_tiny_images('./hw5_data/test/')

def knn(train, test, n):
    res = []
    for category, images in train.items():
        for img in images:
            res.append(np.linalg.norm(img - test))
    dis = np.array(res)

    count = np.argsort(dis)[:n] // len(images)

    bc = np.bincount(count)

    max_num = 0
    min_dis = 10000
    final_idx = 0
    for idx, item in enumerate(bc):
      if item > max_num and dis[idx] < min_dis:
        max_num = item
        min_dis = dis[idx]
        final_idx = idx
      
    classes = list(train_image_feats.keys())
    return classes[final_idx]


def evaluation(train, test, n):
    num_correct = 0
    for category, images in test.items():
        for img in images:
            if knn(train, img, n) == category:
                num_correct += 1
    accuracy_score = num_correct / (len(images) * len(test))
    return accuracy_score

results = evaluation(train_image_feats, test_image_feats, 4)

print("Tiny images representation and nearest neighbor classifier\nAccuracy score: {:.1%}".format(results))