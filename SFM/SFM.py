#!/usr/bin/env python
# coding: utf-8

# In[282]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm.notebook import tqdm
from math import sqrt
import mpl_toolkits.mplot3d.axes3d as p3
plt.rcParams['figure.figsize'] = [15, 15]


# In[283]:


# Read image and convert them to gray!!
def read_image(path):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray, img, img_rgb


# In[284]:


def sift(img):
    siftDetector= cv2.xfeatures2d.SIFT_create()

    kp, des = siftDetector.detectAndCompute(img, None)
    return kp, des


# In[285]:


def plot_matches(matches, total_img):
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
     
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)

    plt.show()


# In[286]:


def plot_sift(gray, rgb, kp):
    tmp = rgb.copy()
    img = cv2.drawKeypoints(gray, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img


# In[287]:


def matcher(kp1, des1, img1, kp2, des2, img2, threshold):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])

    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

    matches = np.array(matches)
    return matches


# In[288]:


def normalization(points):
    # De-mean to center the origin at mean.
    mean = np.mean(points, axis=0)
    
    # Rescale.
    std_x = np.std(points[:, 0])
    std_y = np.std(points[:, 1])

    # Matrix for transforming points to do normalization.
    transform = np.array([[sqrt(2)/std_x, 0, -sqrt(2)/std_x*mean[0]], 
                          [0, sqrt(2)/std_y, -sqrt(2)/std_y*mean[1]], 
                          [0, 0, 1]])

    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    normalized = np.dot(transform, points.T).T

    return normalized[:, 0:2], transform


# In[289]:


def fundamental_matrix(pairs):
    all_p1 = matches[:, 0:2]  # All points of image1 in matching pairs.
    all_p2 = matches[:, 2:4]  # All points of image2 in matching pairs.
    
    # Normalization
    all_p1, T1 = normalization(all_p1)
    all_p2, T2 = normalization(all_p2)
    
    # Solving F
    A_rows = []  # Every row in A is a sublist of A_row.
    
    for i in range(all_p1.shape[0]):
        p1 = all_p1[i]
        p2 = all_p2[i]
        
        row = [p2[0]*p1[0], p2[0]*p1[1], p2[0], 
               p2[1]*p1[0], p2[1]*p1[1], p2[1], p1[0], p1[1], 1]
        A_rows.append(row)

    A = np.array(A_rows)

    U, s, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    F = F / F[2, 2]
    
    # Enforce rank-2 constraint.
    U, s, Vh = np.linalg.svd(F)
    s_prime = np.diag(s)
    s_prime[-1] = 0

    F = np.dot(U, np.dot(s_prime, Vh))
    
    # Denormalization
    F = np.dot(np.dot(T2.T, F), T1)
    
    return F


# In[290]:


def get_errors(matches, F):
    # Compute average geometric distances between epipolar line and its 
    # corresponding point in both images

    ones = np.ones((matches.shape[0], 1))

    all_p1 = np.concatenate((matches[:, 0:2], ones), axis=1)
    all_p2 = np.concatenate((matches[:, 2:4], ones), axis=1)
    
    # Epipolar lines.
    F_p1 = np.dot(F, all_p1.T).T  # F*p1, dims [#points, 3].
    F_p2 = np.dot(F.T, all_p2.T).T  # (F^T)*p2, dims [#points, 3].
    # Geometric distances.
    p1_line2 = np.sum(all_p1 * F_p2, axis=1)[:, np.newaxis]
    p2_line1 = np.sum(all_p2 * F_p1, axis=1)[:, np.newaxis]
    d1 = np.absolute(p1_line2) / np.linalg.norm(F_p2, axis=1)[:, np.newaxis]
    d2 = np.absolute(p2_line1) / np.linalg.norm(F_p1, axis=1)[:, np.newaxis]

    return (d1 + d2) / 2


# In[291]:


def random_pairs(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    pairs = [matches[i] for i in idx ]
    return np.array(pairs)


# In[292]:


def ransac(matches, threshold, iters):
    print("running ransac ...")
    num_best_inliers = 0
    
    for i in tqdm(range(iters)):
        pairs = random_pairs(matches)
        F = fundamental_matrix(pairs)

        errors = get_errors(matches, F)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_F = F.copy()
            
    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_F


# In[293]:


def triangulate(P1, P2, matches):
    # Don't know why needs to transpose V, but it just works..
    U, s, V = np.linalg.svd(P1)
    center1 = V.T[:, -1]
    center1 = center1/center1[-1]
    
    U, s, V = np.linalg.svd(P2)
    center2 = V.T[:, -1]
    center2 = center2/center2[-1]
    
    # Convert on homogeneous.
    ones = np.ones((matches.shape[0], 1))
    points1 = np.concatenate((matches[:, 0:2], ones), axis=1)
    points2 = np.concatenate((matches[:, 2:4], ones), axis=1) 

    # Reconstruct 3D points.
    X_3d = np.zeros((matches.shape[0], 4))
    for i in range(matches.shape[0]):
        x1_cross_P1 = np.array([[0, -points1[i,2], points1[i,1]], 
                          [points1[i,2], 0, -points1[i,0]], 
                          [-points1[i,1], points1[i,0], 0]])
        x2_cross_P2 = np.array([[0, -points2[i,2], points2[i,1]], 
                          [points2[i,2], 0, -points2[i,0]], 
                          [-points2[i,1], points2[i,0], 0]])

        x_cross_P = np.concatenate((x1_cross_P1.dot(P1), x2_cross_P2.dot(P2)), 
                                   axis=0)
        
        # X_3d will become inf when I don't use the tmp var, I don't know why.
        U, s, V = np.linalg.svd(x_cross_P)
        temp = V.T[:, -1]
        temp = temp / temp[-1]
        X_3d[i] = temp

    return center1, center2, X_3d


# In[294]:


def reconstruct(K1, K2, F):
    
    E = np.dot(np.dot(K2.T, F), K1)
    U, s, Vh = np.linalg.svd(E)
    
    W = np.array([0, -1, 0, 1, 0, 0, 0, 0, 1]).reshape(3, 3)
    
    R1 = np.dot(np.dot(U, W), Vh)
    R2 = np.dot(np.dot(U, W.T), Vh)
    T1 = (U[:, 2]).reshape(3, 1)
    T2 = -T1
 
    P1 = np.concatenate((R1, T1), axis=1)
    P2 = np.concatenate((R1, T2), axis=1)
    P3 = np.concatenate((R2, T1), axis=1)
    P4 = np.concatenate((R2, T2), axis=1)
    
    return K2.dot(P1), K2.dot(P2), K2.dot(P3), K2.dot(P4)


# In[295]:


def plot_epipolar(matches, F, image):
    # Display second image with epipolar lines reprojected 
    # from the first image.

    # first, fit fundamental matrix to the matches
    N = len(matches)
    M = np.c_[matches[:,0:2], np.ones((N,1))].transpose()
    L1 = np.matmul(F, M).transpose() # transform points from 
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:,0]**2 + L1[:,1]**2)
    L = np.divide(L1,np.kron(np.ones((3,1)),l).transpose())# rescale the line
    pt_line_dist = np.multiply(L, np.c_[matches[:,2:4], np.ones((N,1))]).sum(axis = 1)
    closest_pt = matches[:,2:4] - np.multiply(L[:,0:2],np.kron(np.ones((2,1)), pt_line_dist).transpose())

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - np.c_[L[:,1], -L[:,0]]*10# offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:,1], -L[:,0]]*10

    # display points and segments of corresponding epipolar lines
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(image).astype('uint8'))
    ax.plot(matches[:,2],matches[:,3], 'or', markersize=2)
    ax.plot([matches[:,2], closest_pt[:,0]],[matches[:,3], closest_pt[:,1]], 'r')
    ax.plot([pt1[:,0], pt2[:,0]],[pt1[:,1], pt2[:,1]], 'g', linewidth=1)
    plt.axis('off')
    plt.show()


# In[296]:


def plot_3d(center1, center2, X_3d):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    
    ax.scatter(X_3d[:,0], X_3d[:,1], X_3d[:,2], c='b', marker='o', alpha=0.6)
    ax.scatter(center1[0], center1[1], center1[2], c='r', marker='+', s=200)
    ax.scatter(center2[0], center2[1], center2[2], c='g', marker='+', s=200)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# In[297]:


is_box = 1
is_statue = 0
is_house = 0
is_library =0


# In[298]:


# box
if is_box:
    left_gray, left_origin, left_rgb = read_image('1.jpeg')
    right_gray, right_origin, right_rgb = read_image('2.jpeg')


# In[299]:


if is_statue:
    left_gray, left_origin, left_rgb = read_image('3.jpg')
    right_gray, right_origin, right_rgb = read_image('4.jpg')


# In[300]:


if is_house:
    left_gray, left_origin, left_rgb = read_image('house1.jpg')
    right_gray, right_origin, right_rgb = read_image('house2.jpg')


# In[301]:


if is_library:
    left_gray, left_origin, left_rgb = read_image('library1.jpg')
    right_gray, right_origin, right_rgb = read_image('library2.jpg')


# In[302]:


# SIFT only can use gray
kp_left, des_left = sift(left_gray)
kp_right, des_right = sift(right_gray)


# In[303]:


kp_left_img = plot_sift(left_gray, left_rgb, kp_left)
kp_right_img = plot_sift(right_gray, right_rgb, kp_right)
total_kp = np.concatenate((kp_left_img, kp_right_img), axis=1)
plt.imshow(total_kp)


# In[304]:


matches = matcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb, 0.5)


# In[305]:


total_img = np.concatenate((left_rgb, right_rgb), axis=1)
plot_matches(matches, total_img) # Good mathces


# In[306]:


inliers, F = ransac(matches, 0.5, 2000)


# In[307]:


plot_matches(inliers, total_img) # show inliers matches


# In[308]:


plot_epipolar(matches, F, right_rgb)


# In[318]:


if is_box:
    K1 = np.array([1.4219, 0.0005, 0.5092, 0, 1.4219, 0.3802, 0, 0, 0.0010]).reshape(3,3)
    K1 = K1 * 1000

    # the first camera matrix
    P0 = np.array([1, 0 ,0 ,0, 0, 1, 0, 0, 0, 0, 1, 0]).reshape(3, 4)
    # possible camera matrix for the second one
    P1, P2, P3, P4 = reconstruct(K1, K1, F)


# In[319]:


# case 1
if is_box:
    center1, center2, X_3D = triangulate(P0, P1, matches)
    plot_3d(center1, center2, X_3D)


# In[320]:


# case 2
if is_box:
    center1, center2, X_3D = triangulate(P0, P2, matches)
    plot_3d(center1, center2, X_3D)


# In[321]:


# case 3
if is_box:
    center1, center2, X_3D = triangulate(P0, P3, matches)
    plot_3d(center1, center2, X_3D)


# In[322]:


# case 4
if is_box:
    center1, center2, X_3D = triangulate(P0, P4, matches)
    plot_3d(center1, center2, X_3D)


# In[323]:


if is_statue:
    K1 = np.array([5426.566895, 0.678017, 330.096680,
    0.000000, 5423.133301, 648.950012,
    0.000000, 0.000000, 1.000000
    ]).reshape(3,3)
    E1 = np.array([0.140626, 0.989027, -0.045273, -1.71427019,
    0.475766, -0.107607, -0.872965, 2.36271724,
    -0.868258, 0.101223, -0.485678, 78.73528449]).reshape(3, 4)

    K2 = np.array([5426.566895, 0.678017, 387.430023,
    0.000000, 5423.133301, 620.616699,
    0.000000, 0.000000, 1.000000
    ]).reshape(3,3)
    E2 = np.array([0.336455, 0.940689, -0.043627, 0.44275193,
    0.446741, -0.200225, -0.871970, 3.03985054,
    -0.828988, 0.273889, -0.487611, 77.67276126]).reshape(3, 4)

    camera1 = np.dot(K1, E1)
    camera2 = np.dot(K2, E2)

    center1, center2, X_3D = triangulate(camera1, camera2, matches)
    plot_3d(center1, center2, X_3D)


# In[324]:


'''
reference:
https://cmsc426.github.io/sfm/
http://www.cs.cmu.edu/~16385/s17/Slides/12.4_8Point_Algorithm.pdf
http://www.cs.cmu.edu/~16385/s17/Slides/12.5_Reconstruction.pdf
'''


# In[325]:


if is_house:

    camera1 = np.array([  1.6108033e+001,  1.3704159e+001 ,-6.7351564e+001 ,-1.8838024e+002,
  8.2886212e-001 ,-6.1257005e+001 ,-2.7985739e+001 ,-7.4190016e+000,
  1.6739784e-001 ,-4.5720139e-002 ,-8.4811075e-002  ,5.6548906e-001
]).reshape(3, 4)

    camera2 = np.array([  1.0571624e+001 , 4.0812730e+000 ,-2.2538413e+001, -5.9593366e+001,
  3.1827253e-001 ,-2.1616617e+001, -9.8820962e+000, -2.7146868e+000,
  6.1142503e-002, -2.0656640e-002,-2.0701037e-002 , 2.5211789e-001]).reshape(3, 4)

    center1, center2, X_3D = triangulate(camera1, camera2, matches)
    plot_3d(center1, center2, X_3D)


# In[326]:


if is_library:

    camera1 = np.array([  -4.5250208e+001,  4.8215478e+002 , 4.0948922e+002,  3.4440464e+003,
  4.8858466e+002 , 2.7346374e+002 ,-1.3977268e+002 , 4.8030231e+003,
 -1.9787463e-001 , 8.8042214e-001 ,-4.3093212e-001 , 2.8032556e+001

]).reshape(3, 4)

    camera2 = np.array([  -5.9593834e+001 , 5.5643970e+002  ,2.3093716e+002,  3.5683545e+003,
  4.6419679e+002 , 2.2628430e+002, -1.9605278e+002 , 4.8734171e+003,
 -1.9116708e-001 , 7.2057697e-001, -6.6650130e-001 , 2.8015392e+001
]).reshape(3, 4)

    center1, center2, X_3D = triangulate(camera1, camera2, matches)
    plot_3d(center1, center2, X_3D)

