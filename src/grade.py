from __future__ import print_function
import os
from math import sin, cos, atan2, sqrt, pi
from run import *
from util import *
import random
random.seed(42)

SRC_PATH = path.dirname(path.abspath(__file__))
DATA_PATH = path.join(SRC_PATH, '..', 'data')
YEARBOOK_PATH = path.join(DATA_PATH, 'yearbook')
YEARBOOK_VALID_PATH = path.join(YEARBOOK_PATH, 'valid')
YEARBOOK_TEST_PATH = path.join(YEARBOOK_PATH, 'test')
YEARBOOK_TRAIN_PATH = path.join(YEARBOOK_PATH, 'train')
YEARBOOK_TEST_LABEL_PATH = path.join(SRC_PATH, '..', 'output', 'yearbook_test_label.txt')
STREETVIEW_PATH = path.join(DATA_PATH, 'geo')
STREETVIEW_VALID_PATH = path.join(STREETVIEW_PATH, 'valid')
STREETVIEW_TEST_PATH = path.join(STREETVIEW_PATH, 'test')
STREETVIEW_TEST_LABEL_PATH = path.join(SRC_PATH, '..', 'output', 'geo_test_label.txt')


def numToRadians(x):
    return x / 180.0 * pi


# Calculate L1 score: max, min, mean, standard deviation
def maxScore(ground_truth, predicted_values):
    diff = np.absolute(np.array(ground_truth) - np.array(predicted_values))
    diff_sum = np.sum(diff, axis=1)
    return np.max(diff_sum), np.min(diff_sum), np.mean(diff_sum), np.std(diff_sum)


# Calculate distance (km) between Latitude/Longitude points
# Reference: http://www.movable-type.co.uk/scripts/latlong.html
EARTH_RADIUS = 6371


def dist(lat1, lon1, lat2, lon2):
    lat1 = numToRadians(lat1)
    lon1 = numToRadians(lon1)
    lat2 = numToRadians(lat2)
    lon2 = numToRadians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2.0) * sin(dlat / 2.0) + cos(lat1) * cos(lat2) * sin(dlon / 2.0) * sin(dlon / 2.0)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    d = EARTH_RADIUS * c
    return d


# Evaluate L1 distance on valid data for yearbook dataset
def evaluateYearbook(predictor):
    test_list = util.listYearbook(False, True)
    #predictor = Predictor()
    predictor.DATASET_TYPE = 'yearbook'

    total_count = len(test_list)
    l1_dist = 0.0
    print("Total validation data", total_count)
    count = 0

    folder_path = path.join(SRC_PATH, '..', 'output', predictor.model_name)
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    output_path = path.join(folder_path, 'yearbook_validation_label.txt')

    with open(output_path, 'a') as f:
        for image_gr_truth in test_list:
            image_path = path.join(YEARBOOK_VALID_PATH, image_gr_truth[0])
            pred_year = predictor.predict(image_path)
            truth_year = int(image_gr_truth[1])
            l1_dist += abs(pred_year[0] - truth_year)
            count = count + 1
            print('%s,%i,%i,%i\n'%(image_gr_truth[0], pred_year[0], truth_year, abs(pred_year[0] - truth_year)))
            f.write('%s,%i,%i,%i\n'%(image_gr_truth[0], pred_year[0], truth_year, abs(pred_year[0] - truth_year)))

        l1_dist /= total_count
        print("L1 distance", l1_dist)
        f.write(str(l1_dist))
    return l1_dist


# Evaluate L1 distance on valid data for geolocation dataset
def evaluateStreetview(Predictor):
    test_list = listStreetView(False, True)
    predictor = Predictor()
    predictor.DATASET_TYPE = 'geolocation'

    total_count = len(test_list)
    l1_dist = 0
    print("Total validation data", total_count)
    for image_gr_truth in test_list:
        image_path = path.join(STREETVIEW_VALID_PATH, image_gr_truth[0])
        pred_lat, pred_lon = predictor.predict(image_path)
        truth_lat, truth_lon = float(image_gr_truth[1]), float(image_gr_truth[2])
        l1_dist += dist(pred_lat, pred_lon, truth_lat, truth_lon)
    l1_dist /= total_count
    print("L1 distance", l1_dist)
    return l1_dist


# Predict label for test data on yearbook dataset
def predictTestYearbook(predictor):
    test_list = util.testListYearbook()
    #predictor = Predictor()
    """predictor.DATASET_TYPE = 'yearbook'

    total_count = len(test_list)
    print("Total test data: ", total_count)

    folder_path = path.join(SRC_PATH, '..', 'output', predictor.model_name)
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    output_path = path.join(folder_path,'yearbook_test_label.txt')
    with open(output_path, 'w') as f:
        for image in test_list:
            image_path = path.join(YEARBOOK_TEST_PATH, image[0])
            pred_year = predictor.predict(image_path)
            #out_string = image[0] + '\t' + str(pred_year[0]) + '\n'
            # output.write(out_string)
            f.write('%s,%i\n' % (image[0], pred_year[0]))"""

    predictor.DATASET_TYPE = 'yearbook'

    total_count = len(test_list)
    l1_dist = 0.0
    print("Total test data", total_count)
    count = 0

    folder_path = path.join(SRC_PATH, '..', 'output', predictor.model_name)
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    output_path = path.join(folder_path, 'yearbook_test_label.txt')

    with open(output_path, 'a') as f:
        for image_gr_truth in test_list:
            if random.random < 0.5:
                continue
            image_path = path.join(YEARBOOK_TEST_PATH, image_gr_truth[0])
            pred_year = predictor.predict(image_path)
            truth_year = int(image_gr_truth[1])
            l1_dist += abs(pred_year[0] - truth_year)
            count = count + 1
            print('%s,%i,%i,%i\n' % (image_gr_truth[0], pred_year[0], truth_year, abs(pred_year[0] - truth_year)))
            f.write('%s,%i,%i,%i\n' % (image_gr_truth[0], pred_year[0], truth_year, abs(pred_year[0] - truth_year)))

        l1_dist /= count
        print("L1 distance", l1_dist)
        f.write(str(l1_dist))
    return l1_dist

# Predict label for test data for geolocation dataset
def predictTestStreetview(Predictor):
    test_list = testListStreetView()
    predictor = Predictor()
    predictor.DATASET_TYPE = 'geolocation'

    total_count = len(test_list)
    print("Total test data", total_count)

    output = open(STREETVIEW_TEST_LABEL_PATH, 'w')
    for image in test_list:
        image_path = path.join(STREETVIEW_TEST_PATH, image[0])
        pred_lat, pred_lon = predictor.predict(image_path)
        out_string = image[0] + '\t' + str(pred_lat) + '\t' + str(pred_lon) + '\n'
        output.write(out_string)
    output.close()


if __name__ == "__main__":
    import importlib
    from argparse import ArgumentParser

    parser = ArgumentParser("Evaluate a model on the validation set")
    parser.add_argument("--DATASET_TYPE", dest="dataset_type",
                        help="Dataset: yearbook/geolocation", required=True)
    parser.add_argument("--type", dest="type",
                        help="Dataset: valid/test", required=True)
    parser.add_argument("--model", dest="model",
                        help="Model name", default='trained_graph.pb')
    parser.add_argument("--label_path", dest="label_path",
                        default="trained_labels.txt")
    parser.add_argument("--data_type", dest="data_type",
                        default="year")

    args = parser.parse_args()

    predictor = Predictor(model_name=args.model, label_path=args.label_path, type=args.data_type)
    if args.dataset_type == 'yearbook':
        print("Yearbook")
        if (args.type == 'valid'):
            evaluateYearbook(predictor)
        elif (args.type == 'test'):
            predictTestYearbook(predictor)
        else:
            print("Unknown type '%s'", args.type)
    elif args.dataset_type == 'geolocation':
        print("Geolocation")
        if (args.type == 'valid'):
            evaluateStreetview(Predictor)
        elif (args.type == 'test'):
            predictTestStreetview(Predictor)
        else:
            print("Unknown type '%s'", args.type)
    else:
        print("Unknown dataset type '%s'", Predictor.DATASET_TYPE)
        exit(1)
