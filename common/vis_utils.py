"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def show_batch(batch):
    npimg = batch.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()


def show_stereo_batch(l_batch, r_batch):
    l_npimg = np.transpose(l_batch.numpy(), (1, 2, 0))
    r_npimg = np.transpose(r_batch.numpy(), (1, 2, 0))
    plt.imshow(
        np.concatenate(
            (l_npimg,
             r_npimg),
            axis=1),
        interpolation='nearest')
    plt.show()
    
rgb_colors = [np.array([0.0, 0.0, 255.0]),
              np.array([0.0, 255.0, 0.0]),
              np.array([255.0, 0.0, 0.0]),
              np.array([0.0,255.0,255.0]),
              np.array([255.0,255.0,0.0]),
              np.array([192.0,192.0,192.0]),
              np.array([0.0, 0.0, 128.0]),
              np.array([0.0, 128.0, 128.0]),
              np.array([0.0, 128.0, 0.0]),
              np.array([128.0, 0.0, 128.0])
             ]

names = ['Background', 'Sky', 'Road', 'Sidewalk', 'Grass', 'Vegetation', 
            'Building', 'Poles', 'Dynamic', 'Unknown']

def sem_label_to_name_deeploc(label):
    global names
    return names[label]

def sem_labels_to_rgb_deeploc(img):
    """
    Assuming getting input of shape (WxHx10)
    
    Background	0	0	255	0
    Sky		    0	255	0	1
    Road		255	0	0	2
    Sidewalk	0	255	255	3
    Grass		255	255	0	4
    Vegetation	192	192	192	5
    Building	0	0	128	6
    Poles		0	128	128	7
    Dynamic		0	128	0	8
    Unknown		128	0	128	9
    """
    global rgb_colors
    s = img.shape
    out = np.empty((s[0], s[1], 3))
    for i in range(s[0]):
        for j in range(s[1]):
            if len(s) > 2:
                index = np.argmax(img[i,j])
            else:
                index = img[i,j]
            out[i,j] = rgb_colors[index]
    return out

def sem_labels_to_rgb(img):
    img = img/np.max(img)
    cm = plt.get_cmap('rainbow')
    return cm(img)[:,:,0:3]

"""
Transforms tensor of shape (10, x, y) to shape (x, y)
"""
def one_hot_to_one_channel(img):
    return np.argmax(img, axis=0)

def normalize(out):
    return (out-np.min(out))/abs(np.max(out)-np.min(out))


def vis_tsne(embedding, images, ax=None):
    """

    :param embedding:
    :param images: list of PIL images
    :param ax:
    :return:
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    embedding -= np.min(embedding, axis=0)
    embedding /= np.max(embedding, axis=0)
    S = 5000
    s = 250
    canvas = np.zeros((S, S, 3), dtype=np.uint8)
    for pos, im in zip(embedding[:, [1, 2]], images):
        x, y = (pos * S).astype(np.int)
        im.thumbnail((s, s), Image.ANTIALIAS)
        x = min(x, S - im.size[0])
        y = min(y, S - im.size[1])
        canvas[y:y + im.size[1], x:x + im.size[0], :] = np.array(im)

    ax.imshow(canvas)
    plt.show(block=True)
    
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()





if __name__ == '__main__':
    import pickle
    with open('../data/embedding_data_robotcar_vidvoo.pkl', 'rb') as f:
        print('Reading data...')
        embedding, images = pickle.load(f)
        print('done')

    vis_tsne(embedding, images)
