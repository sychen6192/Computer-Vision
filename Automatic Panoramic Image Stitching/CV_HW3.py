#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
import random


# In[2]:


# Read image and convert them to gray!! we need to use rgb2gray not bgr2gray
def read_image(path):
    img = cv2.imread(path)
    img_gray= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img_gray, img, img_rgb


# In[3]:


left, left_rgb, leftt = read_image('images/1.jpg')
right, right_rgb, rightt = read_image('images/2.jpg')


# In[4]:


def SIFT(img):
    siftDetector= cv2.xfeatures2d.SIFT_create()
    kp, des = siftDetector.detectAndCompute(img, None)
    return kp, des

def plot_sift(img, kp):
    tmp = img.copy()
    cv2.drawKeypoints(img, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(tmp)
    cv2.imwrite('data/sift.jpg', cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
    return tmp


# In[5]:


kp_left, des_left = SIFT(left)
kp_right, des_right = SIFT(right)


# In[6]:


def sqeuclidean(des_left, des_right):
    res = []
    for i in des_left:        
        res.append([np.sum(np.square(i-j)) for j in des_right])
    return np.array(res)


# In[7]:


def match_points(kp_left, kp_right, des_left, des_right, threshold=7000):
    print("Matching ...")
    
    dist = sqeuclidean(des_left, des_right)
    desp1_idx, desp2_idx = np.where(dist < threshold)[0],  np.where(dist < threshold)[1]

    coords = []
    for i, j in zip(desp1_idx, desp2_idx):
        coords.append(list(kp_left[i].pt + kp_right[j].pt))
    
    match_coords = np.array(coords)
    return match_coords


# In[8]:


match =  match_points(kp_left, kp_right, des_left, des_right, 7000)


# In[9]:


def homomat(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[len(V)-1].reshape(3, 3)
    H = H/H[2, 2]
    return H


# In[10]:


def random_point(coords, k=4):
    idx = random.sample(range(coords.shape[0]), k)
    point = []
    for i in idx:
        point.append(coords[i])
    return np.array(point)


# In[11]:


def get_error(point, H):
    num_points = len(point)
    all_p1 = np.concatenate((point[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = point[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error.
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

    return errors


# In[12]:


def ransac(coords, threshold, iters):
    print("RANSAC PROCESSING...")
    num_inliers = 0
    num_best_inliers = 0
    for i in range(iters):
        points = random_point(coords)
        H = homomat(points)
        #  divide by zero avoid
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        errors = get_error(coords, H)
        idx = np.where(errors < threshold)[0]
        inliers = coords[idx]
        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    print("Number of inliers: {}".format(num_best_inliers))
    return best_inliers, best_H


# In[13]:


inliers, H = ransac(match, 0.5, 1000)


# In[14]:




# In[15]:


total_img = np.concatenate((leftt, rightt), axis=1)


# In[16]:


# I've tried to use CV2 plot the points but it need int type


# In[65]:


def plot_inlier_matches(inliers, total_img):
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    
    ax.plot(inliers[:,0], inliers[:,1], 'xr')
    ax.plot(inliers[:,2]+offset, inliers[:,3], 'xr')
    
    ax.plot([inliers[:,0], inliers[:,2]+offset],[inliers[:,1], inliers[:,3]], 'r', linewidth=0.2)
    
    plt.show()
    plt.savefig('foo.png')


# In[66]:


plot_inlier_matches(inliers,total_img)


# In[29]:


def stitch_img(left, right, H):
    
        # Convert to double and normalize. 避免雜訊
    left = cv2.normalize(left.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
        # Convert to double and normalize.
    right = cv2.normalize(right.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    
    # left image
    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T 
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)
    
    # Get height, width
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    # right image
    warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)

    height_r, width_r, channel_r = right.shape
    
    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)
    

    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)
    black = np.zeros(3)  # Black pixel.
    
    # Stitching procedure, store results in warped_l.
    for i in range(warped_r.shape[0]):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]
            
            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                pass
                  
    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    return stitch_image


# In[50]:


res = stitch_img(leftt, rightt, H)


# In[51]:


plt.imshow(res)
plt.show()
