import mnist
import numpy as np
from numpy.random import multivariate_normal
import gzip
import os
import urllib.request
#from PIL import Image
import matplotlib.pyplot as plt

#modified formular 4   
def robust_cond_prob(x,y,z):
    norm_fact = ( 2 * np.pi * sigma**2 ) ** 14
    expo = np.e ** (-(np.linalg.norm(y - x) ** 2) / (2 * sigma**2) - z)
    return expo / norm_fact

#formular 6
def cond_mean(y_list, x):
    enum = 0
    denom = 0
    for y in y_list:
        c = robust_cond_prob(x, y, z)
        enum += c * y
        denom += c
    enum *= np.e **(z)
    denom *= np.e **(z)
    return enum/denom

#formular 7
def map(y_list, x):
    prob_list = []
    
    for y in y_list:
        cp = robust_cond_prob(x, y, 0)
        prob_list.append(cp)
    
    y_max = prob_list.index(max(prob_list))
    return y_list[y_max]
 
#helper function for formular 8 
def get_maxn_rn(y_list, x):
    max_rn = np.NINF
    for y in y_list:
       rn = -(np.linalg.norm(y - x) ** 2) / (2 * sigma**2)
       if rn > max_rn:
          max_rn = rn
          
    return max_rn
       
train = mnist.load_data("train-images-idx3-ubyte.gz")
test = mnist.load_data("t10k-images-idx3-ubyte.gz")
labels = mnist.load_labels()

#normalize each array by dividing by 255
y = train / 255

#str(len(y)) = 60000
#y[0].shape = 28x28 = DxD
d = 28
sigma = 0.25
#sigma = 0.5
#sigma = 1

#add pixelwise noise to each image
x = np.empty(y.shape)
mean = np.full((d), 0)
cov = np.zeros((d,d), float)
np.fill_diagonal(cov, sigma**2)

for idx, image in enumerate(y):
    noise = multivariate_normal(mean, cov, (d)) #returns noise in shape dxd
    x[idx] = image + noise

#randomly select indices from training and testing set
n = 10000
m = 100
n_idces = np.random.choice(x.shape[0], n, replace=False)
n_list = y[n_idces]   
m_idces = np.random.choice(x.shape[0], m, replace=False)  
m_list = x[m_idces]   

res_cm = np.empty(m_list.shape)
res_map = np.empty(m_list.shape)
for idx, m in enumerate(m_list):
    z = get_maxn_rn(n_list, m)
    res_cm[idx] = cond_mean(n_list, m)
    res_map[idx] = map(n_list, m)

#print res images    
fig = plt.figure(figsize=(d, d))  # width, height in inches
fig2 = plt.figure(figsize=(d, d))  # width, height in inches
res_cm = res_cm * 255
for i in range(len(res_cm)):
    sub = fig.add_subplot(10, 10, i + 1)
    sub.imshow(res_cm[i], interpolation='nearest')
    
for i in range(len(res_map)):
    sub2 = fig2.add_subplot(10, 10, i + 1)
    sub2.imshow(res_map[i], interpolation='nearest')

plt.show()

#displays a noisy image
#img = Image.fromarray(x[1] * 255)
#img.show()
#img = Image.fromarray(res_cm[0] * 255)
#img.show()

