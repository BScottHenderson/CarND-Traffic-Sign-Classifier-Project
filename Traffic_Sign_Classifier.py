# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 09:47:22 2018

@author: henders

Self-Driving Car Engineer Nanodegree

Project: Traffic Sign Classifier
"""

import os
import sys
import logging
import datetime
import random
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow.contrib.layers import flatten

from tensorflow.keras.datasets import cifar10

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


DATA_DIR            = './data'
MODEL_DIR           = './model'
MODEL_NAME          = 'lenet'
VISUAL_DIR          = './visualization'
TEST_IMAGE_DIR      = './test_images'

# Run option flags.
TEST_ON_CIFAR10     = False
DATA_VISUALIZATION  = False
TEST_MODEL          = True  # Test a saved model.
TOP_K_SOFTMAX       = True


#
# Hyperparameters
#

# Data prep parameters
MAX_IMAGE_SCALE         = 2.0   # Max. image scale factor.
MAX_IMAGE_TRANSLATION   = 3     # Max. +/- image x/y translation in pixels.
MAX_IMAGE_ROTATION      = 10.0  # Max. +/- image rotation in degrees.
MAX_IMAGE_INTENSITY_MOD = 50.0  # Max +/- image intensity modification.

# Model parameters: arguments used for tf.truncated_normal() call to
# initialze weights and biases.
MU    = 0
SIGMA = 0.15

# Training parameters
DROPOUT_KEEP_PROB = 0.5
LEARNING_RATE     = 0.001
EPOCHS            = 25
BATCH_SIZE        = 128


def init_logging(log_file_base='LogFile', logging_level=logging.INFO):
    """
    Application initialization.

    Set up a log file object.

    Args:
        log_file_base (str): Base log file name (date will be appended).
        log_level (Level): Log message level

    Returns:
        Logger: Object to be used for logging
    """
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    string_date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = '{}_{}.log'.format(log_file_base, string_date)
    log = setup_logger(log_dir, log_file, log_level=logging_level)

    return log


# Setup Logging
def setup_logger(log_dir=None,
                 log_file=None,
                 log_format=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                              datefmt='%Y-%m-%d %H:%M:%S'),
                 log_level=logging.INFO):
    """
    Setup a logger.

    Args:
        log_dir (str): Log file directory
        log_file (str): Log file name
        log_format (Formatter): Log file message format
        log_level (Level): Log message level

    Returns:
        Logger: Object to be used for logging
    """
    # Get logger
    logger = logging.getLogger('')
    # Clear logger
    logger.handlers = []
    # Set level
    logger.setLevel(log_level)
    # Setup screen logging (standard out)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(log_format)
    logger.addHandler(sh)
    # Setup file logging
    if log_dir and log_file:
        fh = logging.FileHandler(os.path.join(log_dir, log_file))
        fh.setFormatter(log_format)
        logger.addHandler(fh)

    return logger


def write(msg, log=None):
    if log:
        log.info(msg)
    else:
        print(msg)


def grayscale_and_normalize(X):
    """
    Prepare image data for further processing. Steps:
        1. Convert to grayscale.
        2. Normalize the image data so that it has mean zero and
        equal variance.
    Assume that the images have dimensions (32, 32, 3) on input.
    After processing the images will have dimensions (32, 32, 1).

    Args:
        X: An array of images.

    Returns:
        A grayscale, normalized array of images.
    """
    # Convert from RGB to grayscale and normalize.
#    X = [normalize(grayscale(img)) for img in X]
#    X = [normalize_min_max(grayscale(img)) for img in X]
    # Using the either of the above steps to convert to grayscale and
    # normalize does not work well and causes the model accuracy to be
    # extremely low. I don't know why but the following two simple lines
    # of code yield much better results.
    X = np.sum(X / 3, axis=3, keepdims=True)
    X = (X - 128) / 128

    """
    Subtract the mean
    Then divide by the standard deviation

    Normalize the layer by subtracting its mean and dividing by its standard deviation.
    """
    print(X.shape)
    mu = np.mean(X)
    print('mean: {}'.format(mu))
    sigma = np.std(X)
    print('stddev: {}'.format(sigma))

    return X


def grayscale(image):
    """
    Convert an RGB image to grayscale.

    Args:
        image: Image to be converted.

    Returns:
        A grayscale image.
    """
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img= img[:, :, np.newaxis]
    return img


def normalize(image):
    """
    Normalize an image array.

    Assume the image is stored as an array or matrix of uint8. Since the max
    value of uint8 is 255 we use 128 (255 / 2) as the constant in the
    formula below.

    Args:
        image: input image

    Returns:
        Normalized image
    """
    return image - 128 / 128


def normalize_min_max(image):
    """
    Normalize the image with Min-Max scaling to a range of [0.1, 0.9]

    Args:
        image: The image to be normalized

    Returns:
        Normalized image data
    """
    # Normalize image data to the range [a, b]
    a = 0.1
    b = 0.9
    d = b - a  # delta

    # Assume image data is in grayscale, with the current values in
    # the range [0, 255] (uint8).
    x_min = 0
    x_max = 255
    x_del = x_max - x_min  # delta

    # x' = a + ((x - x_min) * (b - a)) / (x_max - x_min)
    normalized_image = [a + ((x.astype(float) - x_min) * d) / x_del for x in image]

    return normalized_image


def add_modified_images(X, y, n_classes, min_sample_size):
    """
    Given an array of images, ensure that each image class has a minimum
    number of instance.

    In cases where there are less than minimum number of instances of a
    particular image class, we add new images to make up the difference.

    The new images are modified copies of the original images. The
    modification consists of:
        1. Random intensity modification.
        2. Random scaling.
        3. Random rotation.
        4. Random translation.

    Args:
        X: An array of images.
        y: An array of labels.
        n_classes: Number of label classes.

    Returns:
        A modified array of images, labels.
    """
    for classId in range(n_classes):
        class_indices = np.where(y == classId)
        n_samples = len(class_indices[0])
        if n_samples < min_sample_size:
            X_new = []
            for i in range(min_sample_size - n_samples):
                new_img = X[class_indices[0][i % n_samples]]
                new_img = random_translate(random_rotate(random_scale(random_intensity(new_img))))
                X_new.append(new_img)
            X = np.append(X, X_new, axis=0)
            y_new = []
            [y_new.append(classId) for i in range(min_sample_size - n_samples)]
            y = np.append(y, y_new, axis=0)

    return X, y


def random_translate(image):
    """
    Randomly translate an image.

    Args:
        image: The image to be translated.

    Returns:
        The translated image.
    """
    translation = random.randint(-MAX_IMAGE_TRANSLATION, MAX_IMAGE_TRANSLATION)
    rows, cols, _ = image.shape
    M = np.float32([[1, 0, translation],   # x translation
                    [0, 1, translation]])  # y translation
    img = cv2.warpAffine(image, M, (cols, rows))

    img = img[:, :, np.newaxis]

    return img


def random_rotate(image):
    """
    Randomly rotate an image.

    Args:
        image: The image to be rotated.

    Returns:
        The rotated image.
    """
    angle = random.uniform(-MAX_IMAGE_ROTATION, MAX_IMAGE_ROTATION)
#    return rotate_bound(image, angle)
    return rotate(image, angle)


def rotate(image, angle):
    """
    Rotate an image using OpenCV such that the entire image is preserved.

    Args:
        image: The image to be rotated.
        angle: Rotation angle.

    Returns:
        The rotated image.

    Reference:
        OpenCV documentation
        https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html
    """
    rows, cols, _ = image.shape
    # cols-1 and rows-1 are the coordinate limits.
    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, 1)
    img = cv2.warpAffine(image, M, (cols, rows))

    img = img[:, :, np.newaxis]

    return img


def rotate_bound(image, angle):
    """
    Rotate an image using OpenCV such that the entire image is preserved.

    Warning!
    This method may change the image size.

    Args:
        image: The image to be rotated.
        angle: Rotation angle.

    Returns:
        The rotated image.

    Reference:
        Rotate images (correctly) with OpenCV and Python
        by Adrian Rosebrock on January 2, 2017 in OpenCV 3, Tutorials
        https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    """
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    img = cv2.warpAffine(image, M, (nW, nH))

    img = img[:, :, np.newaxis]

    return img


def random_scale(image):
    """
    Randomly scale an image.

    Args:
        image: The image to be scaled.

    Returns:
        The scaled image.
    """
    rows, cols, _ = image.shape
    sf = random.randint(-MAX_IMAGE_SCALE, MAX_IMAGE_SCALE)  # scale factor

    # src:
    pts1 = np.float32([[sf,        sf], [rows - sf,        sf],
                       [sf, cols - sf], [rows - sf, cols - sf]])
    # dst: four corners of the image
    pts2 = np.float32([[0,    0], [rows,    0],
                       [0, cols], [rows, cols]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(image, M, (rows, cols))

    img= img[:, :, np.newaxis]

    return img


def random_intensity(image):
    """
    Randomly modify the intensity of an image.

    Args:
        image: The image to be modified.

    Returns:
        The modified image.
    """
    intensity_delta = np.uint8(random.randint(-MAX_IMAGE_INTENSITY_MOD,
                                              MAX_IMAGE_INTENSITY_MOD))
    if intensity_delta > 0:
        np.where((255 - image) < intensity_delta, 255, image + intensity_delta)
    else:
        np.where(image < intensity_delta, 0, image + intensity_delta)
    return image


def conv2d(x, W, b, strides=1):
    """
    Add a convolution layer using ReLu as the activation function.

    Args:
        x: input data
        W: weights
        b: bias
        strides: stride size (assume square stride)

    Returns:
        A TensorFlow convolution layer.
    """
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    """
    Add a max pooling layer.

    Args:
        x: input data
        k: filter size

    Returns:
        A TensorFlow max pooling layer.
    """
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')


def LeNet(x, n_classes, keep_prob, log):
    """
    Implement a slightly modified LeNet-5 model.
    The modification is to include a dropout op for each of the two
    fully connected layers.

    Args:
        x: input data
        n_classes: number of output classes
        keep_prob: TensorFlow placeholder for dropout keep probability
        log: log file object

    Returns:
        A TensorFlow LeNet-5 model.
    """
    log.info('Hyperparameters for model weight initialization:')
    log.info('')
    log.info('mu:    {}'.format(MU))
    log.info('sigma: {}'.format(SIGMA))
    log.info('')

    # Store layer weights & biass
    weights = {
        # convolution weights: [filter height, filter width, input depth, output depth]
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 6],    mean=MU, stddev=SIGMA)),
        'wc2': tf.Variable(tf.random_normal([5, 5, 6, 16],   mean=MU, stddev=SIGMA)),
        # fully connected weights: [input length, output length]
        'wd1': tf.Variable(tf.random_normal([400, 120],      mean=MU, stddev=SIGMA)),
        'wd2': tf.Variable(tf.random_normal([120, 84],       mean=MU, stddev=SIGMA)),
        'out': tf.Variable(tf.random_normal([84, n_classes], mean=MU, stddev=SIGMA))}

    biases = {
        # convolution bias: [output depth]
        'bc1': tf.Variable(tf.zeros([6])),
        'bc2': tf.Variable(tf.zeros([16])),
        # fully connected bias: [output length]
        'bd1': tf.Variable(tf.zeros([120])),
        'bd2': tf.Variable(tf.zeros([84])),
        'out': tf.Variable(tf.zeros([n_classes]))}

    log.debug('x: {}'.format(x))

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # Activation.
    layer1 = conv2d(x, weights['wc1'], biases['bc1'])
    log.debug('layer1: {}'.format(layer1))
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    layer1 = maxpool2d(layer1, k=2)  # k=2 -> cut input size in half
    log.debug('layer1: {}'.format(layer1))

    # Layer 2: Convolutional. Output = 10x10x16.
    # Activation.
    layer2 = conv2d(layer1, weights['wc2'], biases['bc2'])
    log.debug('layer2: {}'.format(layer2))
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    layer2 = maxpool2d(layer2, k=2)  # k=2 -> cut input size in half
    log.debug('layer2: {}'.format(layer2))

    # Flatten. Input = 5x5x16. Output = 400.
    layer2 = flatten(layer2)
    log.debug('layer2: {}'.format(layer2))

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    # Activation.
    layer3 = tf.add(tf.matmul(layer2, weights['wd1']), biases['bd1'])
    layer3 = tf.nn.relu(layer3)
    layer3 = tf.nn.dropout(layer3, keep_prob)
    log.debug('layer3: {}'.format(layer3))

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    # Activation.
    layer4 = tf.add(tf.matmul(layer3, weights['wd2']), biases['bd2'])
    layer4 = tf.nn.relu(layer4)
    layer4 = tf.nn.dropout(layer4, keep_prob)
    log.debug('layer4: {}'.format(layer4))

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.add(tf.matmul(layer4, weights['out']), biases['out'], name='logits')
    log.debug('layer4: {}'.format(logits))

    return logits


def evaluate(X_data, y_data, x, y, keep_prob, accuracy_operation):
    """
    Evalutation function used for training.

    Args:
        X_data: input data
        y_data: labels
        x, y: TensorFlow placeholders for accuracy data
        keep_prob: TensorFlow placeholder for dropout keep probability
        accuracy_operation: TensorFlow accuracy operation

    Returns:
        Average accuracy for all batches.
    """
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_data[offset:end], y_data[offset:end]
        accuracy = sess.run(accuracy_operation,
                            feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def load_data(data_dir, log):
    """
    Load data for model training and evaluation.

    Args:
        data_dir: Directory name where pickle files can be found.
        log: Log file writer.

    Returns:
        X_train, y_train: Training data
        X_valid, y_valid: Validation data
        X_test, y_test: Test data
        n_classes: Number of output classes.
    """
    if TEST_ON_CIFAR10:
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_cifar10_data()
    else:
        X_train, y_train = load_pickle_data(data_dir, 'train')
        X_valid, y_valid = load_pickle_data(data_dir, 'valid')
        X_test,  y_test  = load_pickle_data(data_dir, 'test')

    # Sanity checks.
    assert(len(X_train) == len(y_train))
    assert(len(X_valid) == len(y_valid))
    assert(len(X_test)  == len(y_test))

    image_shape = X_train[0].shape  # Shape of a traffic sign image?

    n_train = len(X_train)  # Number of training examples
    n_valid = len(X_valid)  # Number of validation examples
    n_test  = len(X_test)   # Number of testing examples

    # How many unique classes/labels there are in the dataset.
    n_classes = len(np.unique(y_train))

    log.info('')
    log.info('Image Shape: {}'.format(image_shape))
    log.info('')
    log.info('Training Set:   {:5} samples'.format(n_train))
    log.info('Validation Set: {:5} samples'.format(n_valid))
    log.info('Test Set:       {:5} samples'.format(n_test))
    log.info('')
    log.info('Number of classes = {}'.format(n_classes))
    log.info('')

    return X_train, y_train, X_valid, y_valid, X_test, y_test, n_classes


def load_cifar10_data():
    """
    Load the CIFAR-10 data from keras.

    Args:
        None

    Returns:
        X_train, y_train: Training data
        X_valid, y_valid: Validation data
        X_test, y_test: Test data
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # y_train.shape is 2d, (50000, 1). While Keras is smart enough to handle this
    # it's a good idea to flatten the array.
    y_train = y_train.reshape(-1)
    y_test  = y_test.reshape(-1)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42, stratify=y_train)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def load_pickle_data(data_dir, file_name):
    """
    Load data from pickle files in the specified directory.

    Args:
        data_dir: Directory name where the pickle file can be found.
        file_name: Pickle file name.

    Returns:
        X, y: Images and labels.
    """
    pickle_file = os.path.join(data_dir, file_name + '.p')
    with open(pickle_file, mode='rb') as f:
        data = pickle.load(f)

    X, y = data['features'], data['labels']

    return X, y


def load_test_data(test_data_dir, log):
    """
    Load test data from the test images directory.

    Args:
        test_data_dir: Directory name where test image files and labels can be found.
        log: Log file writer.

    Returns:
        X_test, y_test: Test data
    """
    log.info('')
    log.info('Load test data ...')

    # Load additional test images and labels.
    X_test = []
    y_test = []
    image_labels_file = os.path.join(test_data_dir, 'image_labels.csv')
    image_labels = pd.read_csv(image_labels_file, index_col=['ImageId'])
    for index, row in image_labels.iterrows():
        image_file = os.path.join(test_data_dir, 'image' + str(index).zfill(2) + '.jpg')
        log.info('Test image: {}'.format(image_file))
        image = mpimg.imread(image_file)
        image = cv2.resize(image, (32, 32))  # Scale to 32x32x3
        X_test.append(image)
        y_test.append(row['ClassId'])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    n_test = len(X_test)   # Number of testing examples

    log.info('')
    log.info('Test Set: {:5} images'.format(n_test))
    log.info('')

    return X_test, y_test


def visualize_data(data_dir, X, y, n_classes, output_dir, log):
    """
    Data visualization.

    Args:
        data_dir: Look for the sign names file here.
        X: Data values.
        y: Data labels.
        n_classes: Number of output classes.
        output_dir: Save visualization plots to this location.
        log: Log file writer.

    Returns:
        None
    """

    # Load the sign name lookup table.
    sign_names_file = os.path.join(data_dir, 'signnames.csv')
    sign_names = pd.read_csv(sign_names_file, index_col=['ClassId'])
    pd.set_option('display.max_colwidth', -1)
    log.info('Signs:\n{}'.format(sign_names.head(50)))
    # log.info('sign name[4]=''{}'''.format(sign_names.loc[4][0]))

    label_dict = {v: k for v, k in enumerate(y)}
    fig = plt.figure(figsize=(4, 5))
    columns = 3
    rows = 2
    for index, image in enumerate(X):
        class_id = label_dict[index]
        image = image.squeeze()
        ax = fig.add_subplot(rows, columns, index+1)
        ax.set_title('ClassId={}'.format(class_id))
        plt.imshow(image, cmap='gray')
    fig.subplots_adjust(hspace=.1, wspace=.2)
    plt.show()

    # Display a sample of each sign type.
    label_dict = {k: v for v, k in enumerate(y)}
    fig = plt.figure(figsize=(16, 20))
    columns = 10
    rows = int((n_classes / columns) + 1)
    for k in label_dict:
        index = label_dict[k]
        image = X[index].squeeze()
        ax = fig.add_subplot(rows, columns, k+1)
        ax.set_title('ClassId={}'.format(k))
        plt.imshow(image, cmap='gray')
    fig.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'signs.png'))
    plt.show()

    # Convert the training label data to a DataFrame to make it easier to
    # display a histogram using sign names rather than ids.
    signs = pd.DataFrame(y, columns=['ClassId'])
    signs = signs.merge(sign_names, how='inner', on='ClassId')

    # Display a histogram of sign names.
    fig, ax = plt.subplots()
    # n, bins, patches =
    ax.hist(signs['ClassId'].tolist(), bins=n_classes)
    plt.setp(ax.xaxis.get_ticklabels(), rotation=70)
    ax.set_xticks(sign_names.index)
    ax.set_xticklabels(sign_names['SignName'])
    ax.set_xlabel('Sign')
    ax.set_ylabel('Frequency')
    ax.set_title('Traffic Sign Frequency in Training Dataset')
    fig.set_size_inches(18, 10)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'signs_hist.png'))
    plt.show()


def visualize_test_data(data_dir, X, y, log):
    """
    Data visualization for test data. Instead of displaying a sample of each sign type
    we will display all test images. Also no need for a histogram or for saving plots
    to files.

    Args:
        data_dir: Look for the sign names file here.
        X: Data values.
        y: Data labels.
        log: Log file writer.

    Returns:
        None
    """

    # Load the sign name lookup table.
    sign_names_file = os.path.join(data_dir, 'signnames.csv')
    sign_names = pd.read_csv(sign_names_file, index_col=['ClassId'])

    # Display all test images.
    label_dict = {v: k for v, k in enumerate(y)}
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = int((len(y) / columns) + 1)
    for index, image in enumerate(X):
        class_id = label_dict[index]
        image = image.squeeze()
        ax = fig.add_subplot(rows, columns, index+1)
        # ax.set_title('ClassId={}'.format(class_id))
        ax.set_title('Sign={}'.format(sign_names.loc[class_id][0]))
        plt.imshow(image, cmap='gray')
    # fig.subplots_adjust(hspace=.1, wspace=.2)
    fig.tight_layout()
    plt.show()


def top_k_softmax(k, sess, X, X_normalized, x, keep_prob, log):
    """
    Display the top 'k' softmax probabilities for the test images.

    Args:
        k: Number of probabilties to display.
        sess: Current TensorFlow session.
        X: Input data.
        X_normalized: Normalized input data.
        x: Input tensor.
        keep_prob: Keep probability for dropout.
        log: Log file writer.

    Return:
        None
    """
    # Load validation images and labels.
    X_valid, y_valid = load_pickle_data(DATA_DIR, 'valid')

    # Get the logits from the current graph.
    graph = tf.get_default_graph()
    logits = graph.get_tensor_by_name('logits:0')

    # Top 'k' softmax probabilities for additional test data.
    log.info('Top {} softmax probabilities for test images.'.format(k))
    softmax_logits = tf.nn.softmax(logits)
    top_k_op = tf.nn.top_k(softmax_logits, k=k)
    top_k = sess.run(top_k_op,
                    feed_dict={x: X_normalized, keep_prob: 1.0})

    fig, ax = plt.subplots(len(X), 1+k, figsize=(12, 8))
    fig.subplots_adjust(hspace=.4, wspace=.6)

    # top_k[0] == softmax probabilities
    # top_k[1] == index of the best guess
    # top_k[:, i, j] == probability/best guess for image i, rank j
    for i, image in enumerate(X):
        log.info('Top {} softmax probabilities for image{}:'.format(k, str(i).zfill(2)))
        ax[i][0].axis('off')
        ax[i][0].imshow(image)
        ax[i][0].set_title('Input')
        for j in range(5):
            # Use X_valid, y_valid assuming that this dataset contains
            # at least one instance of each traffic sign class.
            prob  = top_k[0][i][j]*100
            guess = top_k[1][i][j]
            index = np.argwhere(y_valid == guess)[0]
            log.info('rank={}: guess: {} ({:.0f}%)'.format(j+1, str(guess).zfill(2), prob))
            ax[i][j+1].axis('off')
            ax[i][j+1].imshow(X_valid[index].squeeze(), cmap='gray')
            ax[i][j+1].set_title('guess: {} ({:.0f}%)'.format(guess, prob))

    if DATA_VISUALIZATION:
        plt.show()


def preprocess_data(X_train, y_train, X_valid, y_valid, X_test, y_test, n_classes, log):
    """
    Preprocess the data: normalization and augmentation.

    Args:
        X_train, y_train: Training data
        X_valid, y_valid: Validation data
        X_test, y_test: Test data
        n_classes: Number of output classes.
        log: Log file writer.

    Returns:
        X_train, y_train: Training data
        X_valid, y_valid: Validation data
        X_test, y_test: Test data
    """
    log.info('Grayscale+normalization for training data ...')
    X_train = grayscale_and_normalize(X_train)
    log.info('Grayscale+normalization for validation data ...')
    X_valid = grayscale_and_normalize(X_valid)
    log.info('Grayscale+normalization for test data ...')
    X_test  = grayscale_and_normalize(X_test)
    log.info('')

    log.info('Hyperparameters for data prep:')
    log.info('')
    log.info('Max. scale factor:         {}'.format(MAX_IMAGE_SCALE))
    log.info('Max. translation:      +/- {} pixels'.format(MAX_IMAGE_TRANSLATION))
    log.info('Max. rotation:         +/- {} degrees'.format(MAX_IMAGE_ROTATION))
    log.info('Max. intensity change: +/- {}'.format(MAX_IMAGE_INTENSITY_MOD))
    log.info('')

    log.info('Data augmentation for training data ...')
    X_train, y_train = add_modified_images(X_train, y_train, n_classes, 700)
    log.info('Data augmentation for validation data ...')
    X_valid, y_valid = add_modified_images(X_valid, y_valid, n_classes,  50)
    # Do *not* add modified images to the test dataset.
    log.info('')

    # Shuffle
    X_train, y_train = shuffle(X_train, y_train)
    X_valid, y_valid = shuffle(X_valid, y_valid)
    X_test,  y_test  = shuffle(X_test,  y_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def train_and_save_model(X_train, y_train, X_valid, y_valid, X_test, y_test, n_classes, model_dir, model_name, log):
    """
    Train the model and save it to a file.

    Args:
        X_train, y_train: Training data
        X_valid, y_valid: Validation data
        X_test, y_test: Test data
        n_classes: Number of output classes.
        model_dir: Save the model to this directory.
        model_name: Save the model using this name.
        log: Log file writer.

    Returns:
        None
    """
    log.info('Hyperparameters for training:')
    log.info('')
    log.info('Dropout rate:  {}'.format(1.0 - DROPOUT_KEEP_PROB))
    log.info('Learning rate: {}'.format(LEARNING_RATE))
    log.info('Epochs:        {}'.format(EPOCHS))
    log.info('Batch size:    {}'.format(BATCH_SIZE))
    log.info('')

    # !Warning!
    # Don't reset the graph if loading a saved model.
    # !Warning!
    tf.reset_default_graph()

    # Train the model on the training data.
    x = tf.placeholder(tf.float32, (None, 32, 32, 1), name='x')
    y = tf.placeholder(tf.int32, (None), name='y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')    # Keep probability for dropout.
    one_hot_y = tf.stop_gradient(tf.one_hot(y, n_classes))

    logits             = LeNet(x, n_classes, keep_prob, log)
    # https://stats.stackexchange.com/questions/327348/how-is-softmax-cross-entropy-with-logits-different-from-softmax-cross-entropy-wi
    cross_entropy      = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=logits)
    loss_operation     = tf.reduce_mean(cross_entropy)
    optimizer          = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_operation')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        log.info('Training...')
        log.info('')
        num_examples = len(X_train)
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation,
                         feed_dict={x: batch_x, y: batch_y,
                                    keep_prob: DROPOUT_KEEP_PROB})

            valid_accuracy = evaluate(X_valid, y_valid,
                                      x, y, keep_prob, accuracy_operation)
            log.info('EPOCH {} ...'.format(i+1))
            log.info('Validation Accuracy = {:.3f}'.format(valid_accuracy))
            log.info('')

        # Save the trained model.
        save_path = saver.save(sess, os.path.join(model_dir, model_name))
        log.info('Model saved to "{}"'.format(save_path))

    # Calculate and display accuracy for the model on the test data.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        test_accuracy = evaluate(X_test, y_test,
                                 x, y, keep_prob, accuracy_operation)
        log.info('Test Accuracy = {:.3f}'.format(test_accuracy))


def main(name=None):

    print('Name: {}'.format(name))
    _, tail = os.path.split(name)
    log_file_base, _ = os.path.splitext(tail)

    # Init
    log = init_logging(log_file_base, logging.INFO)

    # Disable TensorFlow warnings.
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # {'0', '1', '2', '3'}

    #
    # Train the model.
    #
    if not TEST_MODEL:
        # Load data
        X_train, y_train, X_valid, y_valid, X_test, y_test, n_classes = load_data(DATA_DIR, log)

        # Data visualization
        if DATA_VISUALIZATION:
            visualize_data(DATA_DIR, X_train, y_train, n_classes, VISUAL_DIR, log)

        # Preprocess data.
        X_train, y_train, X_valid, y_valid, X_test, y_test = preprocess_data(X_train, y_train, X_valid, y_valid, X_test, y_test, n_classes, log)

        # Train the model.
        train_and_save_model(X_train, y_train, X_valid, y_valid, X_test, y_test, n_classes, MODEL_DIR, MODEL_NAME, log)

    #
    # Test the model on sample images.
    #
    else:
        # Load test images and labels.
        X_test, y_test = load_test_data(TEST_IMAGE_DIR, log)

        # Display test data.
        if DATA_VISUALIZATION:
            visualize_test_data(DATA_DIR, X_test, y_test, log)

        # Normalize the new test data.
        log.info('Grayscale+normalization for training data ...')
        X_test_normalized = grayscale_and_normalize(X_test)
        log.info('')

        # Measure accuracy.
        log.info('Model accuracy:')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.import_meta_graph(os.path.join(MODEL_DIR, MODEL_NAME + '.meta'))
            saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

            graph = tf.get_default_graph()
            x                  = graph.get_tensor_by_name('x:0')
            y                  = graph.get_tensor_by_name('y:0')
            keep_prob          = graph.get_tensor_by_name('keep_prob:0')
            accuracy_operation = graph.get_tensor_by_name('accuracy_operation:0')

            # Accuracy for each image. The accuracy will be either 0.0 or 1.0.
            for offset in range(len(X_test_normalized)):
                end = offset + 1    # Batch size is 1 - test each image individually.
                batch_x, batch_y = X_test_normalized[offset:end], y_test[offset:end]
                accuracy = sess.run(accuracy_operation,
                                    feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                log.info('Test Accuracy (image{}) = {:.3f}'.format(str(offset).zfill(2), accuracy))
            log.info('')

            # Get overall accuracy for the test images.
            test_accuracy = evaluate(X_test_normalized, y_test,
                                     x, y, keep_prob, accuracy_operation)
            log.info('Test Accuracy (overall) = {:.3f}'.format(test_accuracy))

            if TOP_K_SOFTMAX:
                top_k_softmax(5, sess, X_test, X_test_normalized, x, keep_prob, log)

    # Blank line at end of log file.
    log.info('')

    # Close the log file.
    for handler in log.handlers[:]:
        handler.close()
        log.removeHandler(handler)


if __name__ == '__main__':
    main(*sys.argv)
