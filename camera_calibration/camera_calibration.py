import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image
from scipy.spatial.transform import Rotation
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data/*.jpg')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
num_img = 0
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # If found, add object points, image points
    if ret == True:
        num_img += 1
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)

#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
img_size = (img.shape[1], img.shape[0])
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
'''
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
Vr = np.array(rvecs)
Tr = np.array(tvecs)
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
'''
# solve H
H = np.zeros((num_img, 3, 3))
H_ = np.zeros((num_img, 3, 3))
for i in range(num_img):
    L = np.zeros((2*corner_x*corner_y, 9))
    
    # M is objpoints on same chessboard
    M = [coord for coord in objpoints[i]]
    M = np.array(M)
    M[:, 2] = 1
    u = imgpoints[i][:, 0, 0]
    v = imgpoints[i][:, 0, 1]
    
    # rewrite imgpoint = H * objpoint to Lx = 0, where x is [h1 h2 h3]. Here h1 h2 h3 are row vectors of H
    for j in range(corner_x*corner_y):
        L[j*2] = np.array([M[j, 0], M[j, 1], M[j, 2], 0, 0, 0, -u[j]*M[j, 0], -u[j]*M[j, 1], -u[j]*M[j, 2]])
        L[j*2+1] = np.array([0, 0, 0, M[j, 0], M[j, 1], M[j, 2], -v[j]*M[j, 0], -v[j]*M[j, 1], -v[j]*M[j, 2]])
    
    # x is nullspace of L
    u, w, vh = np.linalg.svd(L)
    x = vh[-1]
    H[i, 0] = x[0:3]
    H[i, 1] = x[3:6]
    H[i, 2] = x[6:9]

    # find rho
    # third row vector of H dot product objpoint and then multiply rho is equal to 1
    # use np.sum divide num_corners to compute mean value of rho
    rho = np.sum([1 / H[i, 2].dot(objpoint) for objpoint in M]) / (corner_x * corner_y)
    H[i] *= rho

# find intrinsic matrix
# first solve B
# define b = (b11, b12, b13, b22, b23, b33)
# by expanding h1^T*B*h2 = 0 and h1^T*B*h1 = h2^T*B*h2, we get Vb = 0
V = np.zeros((num_img*2, 6))
for i in range(num_img):
    # each plane gives two equations
    # h1^T*B*h2 = 0
    v00 = H[i, 0, 0] * H[i, 0, 1]
    v01 = H[i, 0, 0] * H[i, 1, 1] + H[i, 1, 0] * H[i, 0, 1]
    v02 = H[i, 0, 0] * H[i, 2, 1] + H[i, 2, 0] * H[i, 0, 1]
    v03 = H[i, 1, 0] * H[i, 1, 1]
    v04 = H[i, 1, 0] * H[i, 2, 1] + H[i, 2, 0] * H[i, 1, 1]
    v05 = H[i, 2, 0] * H[i, 2, 1]
    v0 = np.array([v00, v01, v02, v03, v04, v05])
    # h1^T*B*h1 = h2^T*B*h2
    v10 = H[i, 0, 0] * H[i, 0, 0] - H[i, 0, 1] * H[i, 0, 1]
    v11 = H[i, 0, 0] * H[i, 1, 0] + H[i, 1, 0] * H[i, 0, 0] - H[i, 0, 1] * H[i, 1, 1] - H[i, 1, 1] * H[i, 0, 1]
    v12 = H[i, 0, 0] * H[i, 2, 0] + H[i, 2, 0] * H[i, 0, 0] - H[i, 0, 1] * H[i, 2, 1] - H[i, 2, 1] * H[i, 0, 1]
    v13 = H[i, 1, 0] * H[i, 1, 0] - H[i, 1, 1] * H[i, 1, 1]
    v14 = H[i, 1, 0] * H[i, 2, 0] + H[i, 2, 0] * H[i, 1, 0] - H[i, 1, 1] * H[i, 2, 1] - H[i, 2, 1] * H[i, 1, 1]
    v15 = H[i, 2, 0] * H[i, 2, 0] - H[i, 2, 1] * H[i, 2, 1]
    v1 = np.array([v10, v11, v12, v13, v14, v15])

    V[i*2] = v0
    V[i*2+1] = v1

# b is nullspace of V
u, s, vh = (np.linalg.svd(V))
b = vh[-1]
B = np.array([[b[0], b[1], b[2]],
              [b[1], b[3], b[4]],
              [b[2], b[4], b[5]]])

# B = lambda * K^-T K^-1
# find lambda
# refer to https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf?fbclid=IwAR392h7MPA70qZsKxoMVYaWex5GpEwH0J8LogG2adZkcVJk1-uEaSzIZt8E
v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1] ** 2)
lamb = B[2, 2] - (B[0, 2] ** 2 + v0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
B /= lamb

# find intrinsic matrix K using B = K^-T K^-1
K_inverse = np.linalg.cholesky(B).T
K = np.linalg.inv(K_inverse)

# initialize extrinsic parameters
ext = np.zeros((num_img, 6))
for i in range(num_img):
    # rotation matrix
    R = np.zeros((3,3))    
    # lambda = 1 / norm(K^-1 h1)
    lamb2 = 1 / np.linalg.norm(np.linalg.inv(K).dot(H[i, :, 0]))
    # r1 = lambda * K^-1 h1
    r1 = lamb2 * np.linalg.inv(K).dot(H[i, :, 0])
    R[:, 0] = r1
    # r2 = lambda * K^-1 h2
    r2 = lamb2 * np.linalg.inv(K).dot(H[i, :, 1])
    R[:, 1] = r2
    # r3 = r1 x r2
    r3 = np.cross(r1, r2)
    R[:, 2] = r3
    # t = lambda * K^-1 h3
    t = lamb2 * np.linalg.inv(K).dot(H[i, :, 2])
    
    # calculate rotation vector from rotation matrix
    r = Rotation.from_matrix(R)
    rvec = np.array(r.as_rotvec())
    ext[i] = np.concatenate((rvec, t))

# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
# camera setting
camera_matrix = K
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, ext, board_width,
                                                board_height, square_size, True)
X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()

#animation for rotating plot
'''
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(0.001)
'''
