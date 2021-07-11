# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('TkAgg')
from vgg_KR_From_P import vgg_KR_from_P
from vgg_KR_From_P import rq
%matplotlib qt
import math
import cv2
import math

#

XYZ = np.array([
    [0,35,21],
    [0,28,14],
    [0,21,7],
    [0,7,14],
    [21,35,0],
    [14,28,0],
    [7,21,0],
    [14,7,0],
    [0,0,0],
    [7,0,21],
    [21,0,21],
    [21,0,7]
])

img = cv2.imread('stereo2012a.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

uv1 = np.load('uv1.npy')

uv_3D = (np.append(uv1,np.ones((12,1)),axis=1))
XYZ_4D = (np.append(XYZ,np.ones((12,1)),axis=1))

# plt.imshow(img)
plt.figure()

plt.scatter(uv1[:,0],uv1[:,1],c='r',s=30,marker='o',zorder=1)
plt.title('Check the selected points in the image')
plt.imshow(img)

#####################################################################
def T_norm(img):
    img_copy = img.copy()
    H = img_copy.shape[0]
    W = img_copy.shape[1]
    matrix = np.array([
        [H + W, 0, W / 2],
        [0, H + W, H / 2],
        [0, 0, 1]
    ])
    T_norm = np.linalg.inv(matrix)
    return T_norm

def S_norm(uv):
# calculate the coovariance matrix of uv
    Cov = np.cov(uv.T)
# generate the eigenvalues and eigenvectors by the Cov
    EigenValues, EigenVectors = np.linalg.eig(Cov)
# generate the diagonal matrix with the eigenvalues
    Diag = np.zeros((EigenValues.shape[0],EigenValues.shape[0]))
    for i in range(EigenValues.shape[0]):
        if EigenValues[i] != 0:
            Diag[i][i] = 1 / EigenValues[i]
        else:
            Diag[i][i] = EigenValues[i]
# generate the S_norm matrix according to the formulas
    S_norm = np.zeros((EigenValues.shape[0] + 1, EigenVectors.shape[1] + 1))
    S_norm[:EigenVectors.shape[0], :EigenVectors.shape[1]] = EigenVectors
    S_norm[:EigenVectors.shape[0], - 1] = EigenValues
    S_norm[-1, -1] = 1
    return S_norm


def generate_x(img, XYZ, uv):
    t_norm = T_norm(img)
    s_norm = S_norm(uv)
    x_i =  uv @ t_norm.T # 12*3
    X_i = XYZ @ s_norm.T # 12*4
    return X_i, x_i

def calibrate(img, XYZ, uv):
    uv_3D = (np.append(uv, np.ones((uv.shape[0], 1)), axis=1))
    XYZ_4D = (np.append(XYZ, np.ones((XYZ.shape[0], 1)), axis=1))
    X_i, x_i = generate_x(img, XYZ_4D, uv_3D)
    i_range = uv.shape[0]
    A = []
# according to the formulas, generate the A matrix
    for i in range(i_range):
        x = list(X_i[i][:])
        u = x_i[i][0]
        v = x_i[i][1]
        
        ux = [a*-u for a in x]
        vx = [b*-v for b in x]
        
        A.append(np.array(x + [0, 0, 0, 0] + ux))
        A.append(np.array([0, 0, 0, 0] + x + vx))

    A = np.stack(A)
    s, v, d = np.linalg.svd(A)
    P = d.T[:, -1].reshape(3, 4)
    return P

def denormalize_DLT(img, P, uv):
    uv_3D = (np.append(uv,np.ones((uv.shape[0],1)),axis=1))
    t_norm = T_norm(img)
    s_norm = S_norm(uv_3D)
    C = np.linalg.inv(t_norm) @ P @ s_norm
    return C



P = calibrate(img, XYZ, uv1)
C = denormalize_DLT(img, P, uv1)
print(C)


new_xyz = XYZ_4D @ C.T

new_points = np.zeros((12,2))
new_points[:,0] = new_xyz[:,0]/new_xyz[:,2]
new_points[:,1] = new_xyz[:,1]/new_xyz[:,2]

error = np.sqrt(np.mean((uv1 - new_points) * (uv1 - new_points)))
print('RMSE is ' + str(error))

plt.figure()

plt.scatter(uv_3D[:,0],uv_3D[:,1],c='r',s=30,marker='o',zorder=1)
plt.scatter(new_points[:,0],new_points[:,1],c='y',s=25,marker='+',zorder=2)
plt.title('Show the points with normalized DLT')

plt.imshow(img)

K,R,t = vgg_KR_from_P(C)
print("K matrix is \n", K)
print("R matrix is \n", R)
print("t matrix is \n", t)

error = np.sqrt(np.mean((uv1 - new_points) * (uv1 - new_points)))
print('RMSE is ' + str(error))

alpha = K[0][0]
gamma = K[0][1]
beta = K[1][1]

fx = alpha
theta = math.atan(-alpha / gamma)
fy = beta * math.sin(theta)
f = np.sqrt(fx**2 + fy**2)

print('Focal length is ' + str(f))

Pitch_angle = math.asin(-R[2][0])
print("pitch angle =",math.degrees(Pitch_angle))



############################################################################


