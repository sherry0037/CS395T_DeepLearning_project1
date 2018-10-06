from os import path
import util
import numpy as np
import argparse
from skimage.io import imread
from util import *
import csv

import tensorflow as tf


SRC_PATH = path.dirname(path.abspath(__file__))
MODEL_PATH = path.join(SRC_PATH, '../model/')
print MODEL_PATH


def load(image_path):
    # TODO:load image and process if you want to do any
    #img=imread(image_path)
    img = tf.gfile.FastGFile(image_path, 'rb').read()
    return img

label_lines = [line.rstrip() for line
                   in tf.gfile.GFile(path.join(MODEL_PATH, "trained_labels.txt"))]

class Predictor:
    DATASET_TYPE = 'yearbook'

    def __init__(self, model_name = "trained_graph.pb"):
        self.model_name = model_name[:-3]
        with tf.gfile.FastGFile(path.join(MODEL_PATH, model_name), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    # baseline 1 which calculates the median of the train data and return each time
    def yearbook_baseline(self):
        # Load all training data
        train_list = listYearbook(train=True, valid=False)

        # Get all the labels
        years = np.array([float(y[1]) for y in train_list])
        med = np.median(years, axis=0)

        return [med]


    def yearbook_tf_inception(self, image_path, loss_type = 'cross_entropy'):
        image_data = load(image_path)

        with tf.Session() as sess:
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor, \
                                   {'DecodeJpeg/contents:0': image_data})

            if loss_type == 'cross_entropy':
                rst = predictions[0].argsort()[-len(predictions[0]):][0]
                rst = np.asscalar(rst)
                print rst
                return [rst+1900]
            elif loss_type == 'MSE':
                rst = int(round(predictions[0][0]))
                print rst
                return [rst + 1900]

    # We do this in the projective space of the map instead of longitude/latitude,
    # as France is almost flat and euclidean distances in the projective space are
    # close enough to spherical distances.
    def streetview_baseline(self):
        # Load all training data
        train_list = listStreetView(train=True, valid=False)

        # Get all the labels
        coord = np.array([(float(y[1]), float(y[2])) for y in train_list])
        xy = coordinateToXY(coord)
        med = np.median(xy, axis=0, keepdims=True)
        med_coord = np.squeeze(XYToCoordinate(med))
        return med_coord

    def predict(self, image_path):

        # TODO: load model

        # TODO: predict model and return result either in geolocation format or yearbook format
        # depending on the dataset you are using
        if self.DATASET_TYPE == 'geolocation':
            result = self.streetview_baseline()  # for geolocation
        elif self.DATASET_TYPE == 'yearbook':
            #result = self.yearbook_baseline()  # for yearbook
            result = self.yearbook_tf_inception(image_path)
        return result
