from zipfile import ZipFile
import tensorflow as tf
import pandas as pd
import numpy as np
import json

Y = pd.read_csv('labels.csv', index_col=0)


def get_processed_imgs(file, img_size):
    """This function processes images from zip file and returns list of scaled and resized matrices"""
    with ZipFile(file) as zip:
        objects = zip.namelist()

    archive = ZipFile(file, 'r')

    data = []
    for item in objects:
        decoded = tf.io.decode_jpeg(archive.read(item), channels=3)
        scaled = tf.image.convert_image_dtype(decoded, tf.float32)
        img = tf.image.resize(scaled, size=(img_size, img_size))
        data.append(img)

    return data


def one_hot(label, labels):
    """This function returns one-hot encoded labels"""
    masked = np.zeros(shape=(len(labels)))
    masked[labels.index(label)] = 1
    return masked






