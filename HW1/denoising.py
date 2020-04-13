import mnist
import numpy as np
from numpy.random import multivariate_normal
import gzip
import os
import urllib.request
from PIL import Image

#formular 4
def cond_prob(x,y):
    norm_fact = ( 2 * np.pi * sigma**2 ) ** 14
    expo = np.e ** (-(np.linalg.norm(y - x) ** 2) / (2 * sigma**2))
    #print(expo / norm_fact)
    return expo / norm_fact

#formular 6
def cond_mean(y_list, x):
    enum = 0
    denom = 0
    for y in y_list:
        c = cond_prob(x, y)
        enum += c * y
        denom += c
    return enum/denom

#formular 7
def map(y_list, x):
    prob_list = []
    
    for y in y_list:
        cp = cond_prob(x, y)
        prob_list.append(cp)
    
    y_max = prob_list.index(max(prob_list))
    return y_list[y_max]
 
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
'''
n = 50000
m = 1
n_idces = np.random.choice(x.shape[0], n, replace=False)
n_list = y[n_idces]   
m_idces = np.random.choice(x.shape[0], m, replace=False)  
m_list = x[m_idces]   
'''
m_list = []
m_list.append(x[1])
n_list = []
n_list.append(y[0])
n_list.append(y[1])
n_list.append(y[2])
n_list.append(y[3])

res_cm = []
res_map = []
for m in m_list:
    res_cm.append(cond_mean(n_list, m))
    #res_map.append(map(n_list, m))

print(res_cm)

#displays a noisy image
img = Image.fromarray(x[1] * 255)
img.show()
img = Image.fromarray(res_cm[0] * 255)
img.show()

