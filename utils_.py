import requests
import os
from scipy.io import loadmat
import numpy as np


def lire_alpha_digit(focus_list, path=None):
    if path == None: 
        alphadigs_url = 'https://cs.nyu.edu/~roweis/data/binaryalphadigs.mat'
        r = requests.get(alphadigs_url, allow_redirects=True)
        filename = 'binaryalphadigs.mat'
        open('data/' + filename, 'wb').write(r.content)
        path = 'data/binaryalphadigs.mat'
        print('Download completed, data available in /data/binaryalphadigs.mat')
    
    elif os.path.exists(path):
        print('Path correct')

    data_dic = loadmat(path)

    digit2idx = {}
    for i, digit in enumerate(data_dic['classlabels'][0]):
        digit2idx[digit[0]] = i

    focus_idx =[]
    for digit in focus_list: 
        focus_idx.append(digit2idx[digit])
    
    data = np.stack(np.concatenate(data_dic['dat'][focus_idx])).reshape(-1, 16*20)
    print(data)
    return data 


def load_mnist():
    list_name = ['train-images', 'train-labels', 't10k-images', 't10kimages']
    for name in list_name: 
        mnist_url = f'http://yann.lecun.com/exdb/mnist/{name}-idx3-ubyte.gz'
        r = requests.get(mnist_url, allow_redirects=True)
        filename = f'mnist-{name}.gz'
        open('data/' + filename, 'wb').write(r.content)
        print(f'Download completed, data available in /data/{name}-idx3-ubyte.gz')
