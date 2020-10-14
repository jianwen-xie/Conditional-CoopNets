from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
from PIL import Image
# import scipy.misc

from six.moves import xrange

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def read_images_from_folder(data_path, img_width=224, img_height=224, low=-1, high=1):
    img_list = [f for f in os.listdir(data_path) if any(f.lower().endswith(ext) for ext in IMG_EXTENSIONS)]
    img_list.sort()
    images = np.zeros(shape=(len(img_list), img_width, img_height, 3))
    print('Reading images from: {}'.format(data_path))
    for i in xrange(len(img_list)):
        image = Image.open(os.path.join(data_path, img_list[i])).convert('RGB')
        image = image.resize((img_width, img_height))
        image = np.asarray(image, dtype=float)
        cmax = image.max()
        cmin = image.min()
        image = (image - cmin) / (cmax - cmin) * (high - low) + low
        images[i] = image
    print('Images loaded, shape: {}'.format(images.shape))
    return images

def labels_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)

    if not os.path.exists(path):
        os.mkdir(path)

def cell2img(cell_image, image_size=100, margin_syn=2):
    num_cols = cell_image.shape[1] // image_size
    num_rows = cell_image.shape[0] // image_size
    images = np.zeros((num_cols * num_rows, image_size, image_size, 3))
    for ir in range(num_rows):
        for ic in range(num_cols):
            temp = cell_image[ir*(image_size+margin_syn):image_size + ir*(image_size+margin_syn),
                   ic*(image_size+margin_syn):image_size + ic*(image_size+margin_syn),:]
            images[ir*num_cols+ic] = temp
    return images

def clip_by_value(input_, low=0, high=1):
    return np.minimum(high, np.maximum(low, input_))

def img2cell(images, row_num=10, col_num=10, low=-1, high=1, margin_syn=2):
    [num_images, image_size] = images.shape[0:2]
    num_cells = int(math.ceil(num_images / (col_num * row_num)))
    cell_image = np.zeros((num_cells, row_num * image_size + (row_num-1)*margin_syn,
                           col_num * image_size + (col_num-1)*margin_syn, images.shape[3]), dtype=np.uint8)
    for i in range(num_images):
        cell_id = int(math.floor(i / (col_num * row_num)))
        idx = i % (col_num * row_num)
        ir = int(math.floor(idx / col_num))
        ic = idx % col_num
        temp = (images[i] + 1.) * 127.5
        temp = np.clip(temp, 0., 255.)
        # cmin = temp.min()
        # cmax = temp.max()
        # temp = (temp - cmin) / (cmax - cmin)
        cell_image[cell_id, (image_size+margin_syn)*ir:image_size + (image_size+margin_syn)*ir,
                    (image_size+margin_syn)*ic:image_size + (image_size+margin_syn)*ic,:] = temp
    if images.shape[3] == 1:
        cell_image = np.squeeze(cell_image, axis=3)
    return cell_image

def saveSampleImages(sample_results, filename, postfix='', row_num=10, col_num=10, margin_syn=2, save_all=False, flip_color=False):
    cell_image = img2cell(sample_results, row_num=row_num, col_num=col_num, margin_syn=margin_syn)
    if flip_color:
        cell_image = 1 - cell_image

    if save_all:
        for ci in range(len(cell_image)):
            Image.fromarray(cell_image[ci]).save(filename[:-4] + '_%03d%s.png' % (ci, postfix))
    else:
        Image.fromarray(cell_image[0]).save(filename)
        # scipy.misc.imsave(filename, cell_image[0])


if __name__ == '__main__':
    # data = DataLoader('../../data/CUHK/train/cropped_sketches', '../../data/CUHK/train/cropped_photos')
    from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
    db = read_data_sets('../../data/mnist', one_hot=True)
    x, y = db.train.next_batch(1)
    x = np.reshape(x, [1, 28, 28, 1])
    saveSampleImages(x, 'test.png', 1, 1)
    print(y)