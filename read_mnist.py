import os
import sys
import struct
from array import array
import random
import numpy as np
from sklearn.model_selection import train_test_split

class MNIST():

    def __init__(self, path, random_seed):
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

        self.num_classes = 10

        self.random_seed = random_seed

        np.random.seed(42)
        random.seed(42)

    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                                os.path.join(self.path, self.test_lbl_fname))

        self.test_images = self.process_images(ims)
        self.test_labels = self.process_labels(labels)

        return self.test_images, self.test_labels

    def load_training(self):
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                                os.path.join(self.path, self.train_lbl_fname))

        self.train_images = self.process_images(ims)
        self.train_labels = self.process_labels(labels)

        return self.train_images, self.train_labels

    def process_images(self, images):
        images_np = np.array(images) / 255.0
        return images_np

    def process_labels(self, labels):
        one_hot_labels = np.eye(self.num_classes, dtype=float)[labels]
        one_hot_labels[one_hot_labels == 0] = -1
        return np.array(one_hot_labels)

    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels

    def get_data(self, train_percent=0.6, valid_percent=0.2):
        assert 0.0 < train_percent < 1.0
        assert 0.0 < valid_percent < 1.0
        assert train_percent + valid_percent < 1.0

        test_percent = 1.0 - train_percent - valid_percent

        self.load_training()
        self.load_testing()

        self.images_full = np.concatenate((self.train_images, self.test_images), axis=0)
        self.labels_full = np.concatenate((self.train_labels, self.test_labels), axis=0)

        x_train, x_test, y_train, y_test = train_test_split(self.images_full, self.labels_full, test_size=test_percent, random_state=self.random_seed)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_percent / (valid_percent + train_percent), random_state=self.random_seed)

        self.data = {
            'train': {
                'x': x_train,
                'y': y_train
            },
            'valid': {
                'x': x_valid,
                'y': y_valid
            },
            'test': {
                'x': x_test,
                'y': y_test
            }
        }

        return self.data