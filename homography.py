
# I1 = Image.open('left.jpg')
# plt.title('select 6 points in the origin image')
# plt.imshow(I1)

# base = np.asarray(plt.ginput(6,timeout=9999))
# np.save('base.npy', base)


# I2 = Image.open('right.jpg')
# plt.title('select 6 coresponding points in the trans image')
# plt.imshow(I2)

# trans = np.asarray(plt.ginput(6,timeout=9999))
# np.save('trans.npy', trans)

img2 = cv2.imread('left.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.imread('right.jpg')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)


base = np.load('base.npy')
plt.figure()

plt.scatter(base[:,0],base[:,1],c='r',s=30,marker='o',zorder=1)
plt.title('Check the selected points in the image')
plt.imshow(img2)


trans = np.load('trans.npy')
plt.figure()

plt.scatter(trans[:,0],trans[:,1],c='r',s=30,marker='o',zorder=1)
plt.title('Check the coresponding points in the image')
plt.imshow(img3)


plt.figure()
plt.subplot(121)
plt.scatter(base[:,0],base[:,1],c='r',s=30,marker='o',zorder=1)
plt.title('The selected points in the left image')
plt.imshow(img2)

plt.subplot(122)
plt.scatter(trans[:,0],trans[:,1],c='r',s=30,marker='o',zorder=1)
plt.title('Check the corresponding points in right the image')
plt.imshow(img3)


# plt.figure()
# plt.imshow(img2)
# plt.title('select 6 points from the image')
# uv2 = np.asarray(plt.ginput(6,timeout=9999)) # Graphical user interface to get 6 points

# np.save('uv2.npy', uv2)


uv2 = np.load('uv2.npy')
plt.figure()

plt.scatter(uv2[:,0],uv2[:,1],c='r',s=30,marker='o',zorder=1)
plt.title('Check the selected points in the image')
plt.imshow(img2)

def T_norm(img):
    img_copy = img.copy()
    H = img_copy.shape[0]
    W = img_copy.shape[1]
    matrix = np.array([
        [H + W, 0, W/2],
        [0, H + W, H/2],
        [0, 0, 1]
    ])
    T_norm = np.linalg.inv(matrix)
    return T_norm

def homography(u2Trans, v2Trans, uBase, vBase):
    A = []
    for i in range(len(u2Trans)):
        x = uBase[i]
        y = vBase[i]
        u = u2Trans[i]
        v = v2Trans[i]
        
        
        A.append(np.array([0, 0, 0, -x, -y, -1, v * x, v * y, v]))
        A.append(np.array([x, y, 1, 0, 0, 0, -u * x, -u * y, -u]))
        
    A = np.stack(A)
    s, v, d = np.linalg.svd(A)
    H = d[-1,:].reshape(3, 3)
#     H = H/H[-1,-1]
    return H

def denormalize_H(T_norm2, H, T_norm1):
    new_H = np.linalg.inv(T_norm2) @ H @ T_norm1
    return new_H


base_3D = (np.append(base,np.ones((6,1)),axis=1))
trans_3D = (np.append(trans,np.ones((6,1)),axis=1))
uv2_3D = (np.append(uv2,np.ones((6,1)),axis=1))

t_norm_1 = T_norm(img2)
t_norm_2 = T_norm(img3)

normalized_base = base_3D @ t_norm_1.T
normalized_trans = trans_3D @ t_norm_2.T

H = homography(normalized_trans[:,0], normalized_trans[:,1], normalized_base[:,0], normalized_base[:,1])
H = denormalize_H(t_norm_2, H, t_norm_1)
print(H)


hz = (H @ uv2_3D.T)

new = np.zeros((2,6))
new[0] = hz[0]/hz[2]
new[1] = hz[1]/hz[2]

plt.figure()
plt.subplot(122)
plt.imshow(img3)
plt.title('Show the normalized homography points')
plt.scatter(new[0],new[1],c='r',s=30,marker='o',zorder=2)


plt.subplot(121)
plt.scatter(uv2[:,0],uv2[:,1],c='r',s=30,marker='o',zorder=1)
plt.title('Check the selected points in the image')
plt.imshow(img2)

plt.figure()
warp_img = cv2.warpPerspective(img2, H, (img2.shape[1],img2.shape[0]))
plt.imshow(warp_img)
plt.title('Show the wrapping image result')
plt.show()

# test the distance of the selected points affect the result on the warpping image


# plt.figure()
# plt.imshow(img2)
# test_base_uv = np.asarray(plt.ginput(6,timeout=9999)) # Graphical user interface to get 6 points

# np.save('test_base_uv.npy', test_base_uv)


# plt.figure()
# plt.imshow(img3)
# test_trans_uv = np.asarray(plt.ginput(6,timeout=9999)) # Graphical user interface to get 6 points

# np.save('test_trans_uv.npy', test_trans_uv)


test_base_uv = np.load('test_base_uv.npy')
test_trans_uv = np.load('test_trans_uv.npy')

test_base_3D = (np.append(test_base_uv,np.ones((6,1)),axis=1))
test_trans_3D = (np.append(test_trans_uv,np.ones((6,1)),axis=1))
uv2_3D = (np.append(uv2,np.ones((6,1)),axis=1))

t_norm_1 = T_norm(img2)
t_norm_2 = T_norm(img3)

norm_test_base = test_base_3D @ t_norm_1.T
norm_test_trans = test_trans_3D @ t_norm_2.T

test_H = homography(norm_test_trans[:,0], norm_test_trans[:,1], norm_test_base[:,0], norm_test_base[:,1])
test_H = denormalize_H(t_norm_2, test_H, t_norm_1)
print(test_H)


new_hz = (test_H @ uv2_3D.T)

test_new = np.zeros((2,6))
test_new[0] = new_hz[0]/new_hz[2]
test_new[1] = new_hz[1]/new_hz[2]

# plt.figure()
# plt.subplot(122)
# plt.imshow(img3)
# plt.title('Show the normalized homography points')
# plt.scatter(test_new[0],test_new[1],c='r',s=30,marker='o',zorder=2)


# plt.subplot(121)
# plt.scatter(uv2[:,0],uv2[:,1],c='r',s=30,marker='o',zorder=1)
# plt.title('Check the selected points in the image')
# plt.imshow(img2)
plt.figure()
plt.scatter(test_base_uv[:,0],test_base_uv[:,1],c='r',s=30,marker='o',zorder=1)
plt.imshow(img2)


plt.figure()
test_warp_img = cv2.warpPerspective(img2, test_H, (img2.shape[1],img2.shape[0]))
plt.imshow(test_warp_img)
plt.title('Show the test1 wrapping image result')
plt.show()


# plt.figure()
# plt.imshow(img2)
# test2_base_uv = np.asarray(plt.ginput(6,timeout=9999)) # Graphical user interface to get 6 points

# np.save('test2_base_uv.npy', test2_base_uv)


# plt.figure()
# plt.imshow(img3)
# test2_trans_uv = np.asarray(plt.ginput(6,timeout=9999)) # Graphical user interface to get 6 points

# np.save('test2_trans_uv.npy', test2_trans_uv)



test2_base_uv = np.load('test2_base_uv.npy')
test2_trans_uv = np.load('test2_trans_uv.npy')

test2_base_3D = (np.append(test2_base_uv,np.ones((6,1)),axis=1))
test2_trans_3D = (np.append(test2_trans_uv,np.ones((6,1)),axis=1))
uv2_3D = (np.append(uv2,np.ones((6,1)),axis=1))

t_norm_1 = T_norm(img2)
t_norm_2 = T_norm(img3)

norm_test2_base = test2_base_3D @ t_norm_1.T
norm_test2_trans = test2_trans_3D @ t_norm_2.T

test2_H = homography(norm_test2_trans[:,0], norm_test2_trans[:,1], norm_test2_base[:,0], norm_test2_base[:,1])
test2_H = denormalize_H(t_norm_2, test2_H, t_norm_1)
print(test_H)


plt.figure()
test2_warp_img = cv2.warpPerspective(img2, test2_H, (img2.shape[1],img2.shape[0]))
plt.imshow(test2_warp_img)
plt.title('Show the test2 wrapping image result')
plt.show()

plt.figure()
plt.scatter(test2_base_uv[:,0],test2_base_uv[:,1],c='r',s=30,marker='o',zorder=1)
plt.imshow(img2)
