# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:24:12 2018

@author: Ouch
"""

#from skimage import data_dir
from skimage import io, transform
import numpy as np
import sys

imageFile = sys.argv[1]
targetFile = sys.argv[2]
idx = int(targetFile[:-4])
#path = 'C:/Users/chich/Desktop/ML2017FALL/6/Report/'
img = io.ImageCollection(imageFile+'/*.jpg')
print(len(img))

shape = img[0].shape
print(shape)

imgList = []
for i in img:
    imgList.append(i.flatten())

def image_show(img):
    io.imshow(img)
    io.show()
    
#Compute Average
def computeAvg(imgList, shape):
    avg = np.mean(imgList, axis = 0).astype(np.uint8)
    avg_reshaped = avg.reshape(shape)
    return avg_reshaped

#img_avg = computeAvg(imgList, shape)
#image_show(img_avg)
#io.imsave(path + 'average.jpg', img_avg)

#SVD
mean = np.mean(imgList, axis = 0)
i_m = imgList - mean
U, s, V = np.linalg.svd( i_m.T, full_matrices=False)
#print(s.shape)
#for i in range(4):
#    summation = s.sum()
#    ans = s[i]/summation
#    print(ans, s[i], summation)

#new_img = transform.resize(old_img, new_shape)
#600*600*3
#EigenFace
def computeEigen(eigen):
    eigen = (-1)*eigen
    eigen_min = np.min(eigen)
    eigen = eigen - eigen_min
    eigen_max = np.max(eigen)
    eigen = eigen / eigen_max
    eigen = (eigen * 255).astype(np.uint8)
    return eigen

##0~3 EigenFace
for i in range(4):
    eigen = np.copy(U[:, i].reshape(shape))
    eigen = computeEigen(eigen)
    #image_show(eigen)
    #io.imsave(path+'Eigen Face' + str(i) +'.jpg', eigen)

img_weights = np.dot( (imgList - mean), U)

def reconstruct_old(img, idx, weights, mean, U, shape):
    #image_show(img[idx])
    #io.imsave('oldIMG' + str(idx) + '.jpg', img[idx])
    
    #EigenFace
    new2old_img = mean + np.dot(weights[idx, :4], U[:,:4].T)
    new2old_img = new2old_img.reshape(shape)
    new2old_img = (-1) * new2old_img
    new2old_img = computeEigen(new2old_img)
    #image_show(new2old_img)
    return new2old_img

new2old_img = reconstruct_old(img, idx, weights=img_weights, mean = mean, U = U, shape = shape)
io.imsave( 'reconstruction.jpg',new2old_img)


#new2old_img = reconstruct_old(img, idx = 7, weights=img_weights, mean = mean, U = U, shape = shape)
#io.imsave(path + 'newTOoldIMG' + str(7) + '.jpg', new2old_img)
#
#new2old_img = reconstruct_old(img, idx = 17, weights=img_weights, mean = mean, U = U, shape = shape)
#io.imsave(path + 'newTOoldIMG' + str(17) + '.jpg', new2old_img)
#
#new2old_img = reconstruct_old(img, idx = 70, weights=img_weights, mean = mean, U = U, shape = shape)
#io.imsave(path + 'newTOoldIMG' + str(70) + '.jpg', new2old_img)
#
#new2old_img = reconstruct_old(img, idx = 77, weights=img_weights, mean = mean, U = U, shape = shape)
#io.imsave(path + 'newTOoldIMG' + str(77) + '.jpg', new2old_img)

  
    
    