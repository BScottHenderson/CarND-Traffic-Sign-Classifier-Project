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

from sklearn.utils import shuffle


#
# Hyperparameters
#

# Data prep parameters
MAX_IMAGE_SCALE         = 2.0   # Max. image scale factor.
MAX_IMAGE_TRANSLATION   = 2     # Max. +/- image x/y translation in pixels.
MAX_IMAGE_ROTATION      = 10.0  # Max. +/- image rotation in degrees.
MAX_IMAGE_INTENSITY_MOD = 50.0  # Max +/- image intensity modification.

# Model parameters: arguments used for tf.truncated_normal in LeNet(),
# randomly defines variables for the weights and biases for each layer
MU    = 0
SIGMA = 0.1

# Training parameters
DROPOUT_KEEP_PROB = 0.5
LEARNING_RATE     = 0.001
EPOCHS            = 25
BATCH_SIZE        = 128


def init(log_file_base='LogFile', logging_level=logging.INFO):
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
    string_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = '{}_{}.log'.format(log_file_base, string_date)
    log = setup_logger(log_dir, log_file, log_level=logging_level)

    return log


# Setup Logging
def setup_logger(log_dir=None,
                 log_file=None,
                 log_format=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                              datefmt="%Y-%m-%d %H:%M:%S"),
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
    # grab the dimensions of the image and then determine the
    # center
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
        'bc1': tf.Variable(tf.random_normal([6],         mean=MU, stddev=SIGMA)),
        'bc2': tf.Variable(tf.random_normal([16],        mean=MU, stddev=SIGMA)),
        # fully connected bias: [output length]
        'bd1': tf.Variable(tf.random_normal([120],       mean=MU, stddev=SIGMA)),
        'bd2': tf.Variable(tf.random_normal([84],        mean=MU, stddev=SIGMA)),
        'out': tf.Variable(tf.random_normal([n_classes], mean=MU, stddev=SIGMA))}

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
    logits = tf.add(tf.matmul(layer4, weights['out']), biases['out'])
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


def main(name):

    print('Name: {}'.format(name))
    head, tail = os.path.split(name)
    log_file_base, ext = os.path.splitext(tail)

    # Init
    log = init(log_file_base, logging.INFO)

    #
    # Load data
    #

    training_file   = './data/train.p'
    validation_file = './data/valid.p'
    testing_file    = './data/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test,  y_test  = test['features'],  test['labels']

    assert(len(X_train) == len(y_train))
    assert(len(X_valid) == len(y_valid))
    assert(len(X_test) == len(y_test))

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

    #
    # Visualize the data
    #

    # Load the sign name lookup table.
    sign_names_file = './data/signnames.csv'
    sign_names = pd.read_csv(sign_names_file, index_col=['ClassId'])
#    log.info('Signs:\n{}'.format(sign_names.head(50)))
#    log.info('sign name[4]=''{}'''.format(sign_names.loc[4][0]))

#    # Display a sample of each sign type.
#    label_dict = {k: v for v, k in enumerate(y_train)}
#
#    fig = plt.figure(figsize=(6, 22))
#    columns = 4
#    rows = int((n_classes / columns) + 1)
#    for k in label_dict:
#        index = label_dict[k]
#        image = X_train[index].squeeze()
#        ax = fig.add_subplot(rows, columns, k+1)
#        ax.set_title('ClassId={}'.format(k))
#        plt.imshow(image, cmap='gray')
#    fig.tight_layout()
#    plt.show()
#
#    # Convert the training label data to a DataFrame to make it easier to
#    # display a histogram using sign names rather than ids.
#    signs = pd.DataFrame(y_train, columns=['ClassId'])
#    signs = signs.merge(sign_names, how='inner', on='ClassId')
#
#    # Display a histogram of sign names.
#    fig, ax = plt.subplots()
#    n, bins, patches = ax.hist(signs['ClassId'].tolist(), bins=n_classes)
#    plt.setp(ax.xaxis.get_ticklabels(), rotation=70)
#    ax.set_xticks(sign_names.index)
#    ax.set_xticklabels(sign_names['SignName'])
#    ax.set_xlabel('Sign')
#    ax.set_ylabel('Frequency')
#    ax.set_title('Traffic Sign Frequency in Training Dataset')
#    fig.set_size_inches(16, 8)

#    #
#    # Preprocess data
#    #
#
#    log.info('Grayscale+normalization for training data ...')
#    X_train = grayscale_and_normalize(X_train)
#    log.info('Grayscale+normalization for validation data ...')
#    X_valid = grayscale_and_normalize(X_valid)
#    log.info('Grayscale+normalization for test data ...')
#    X_test  = grayscale_and_normalize(X_test)
#
#    log.info('Data augmentation for training data ...')
#    X_train, y_train = add_modified_images(X_train, y_train, n_classes, 700)
#    log.info('Data augmentation for validation data ...')
#    X_valid, y_valid = add_modified_images(X_valid, y_valid, n_classes,  50)
#    # Do *not* add modified images to the test dataset.
#
#    # Shuffle
#    X_train, y_train = shuffle(X_train, y_train)
#    X_valid, y_valid = shuffle(X_valid, y_valid)
#    X_test,  y_test  = shuffle(X_test,  y_test)
#
#    #
#    # Train
#    #
#
    tf.reset_default_graph()

    # Train the model on the training data.
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    keep_prob = tf.placeholder(tf.float32)  # Keep probability for dropout.
    one_hot_y = tf.one_hot(y, n_classes)

    logits             = LeNet(x, n_classes, keep_prob, log)
    cross_entropy      = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation     = tf.reduce_mean(cross_entropy)
    optimizer          = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#    saver = tf.train.Saver()
#
#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#
#        log.info('Training...')
#        log.info('')
#        num_examples = len(X_train)
#        for i in range(EPOCHS):
#            X_train, y_train = shuffle(X_train, y_train)
#            for offset in range(0, num_examples, BATCH_SIZE):
#                end = offset + BATCH_SIZE
#                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
#                sess.run(training_operation,
#                         feed_dict={x: batch_x, y: batch_y,
#                                    keep_prob: DROPOUT_KEEP_PROB})
#
#            valid_accuracy = evaluate(X_valid, y_valid,
#                                      x, y, keep_prob, accuracy_operation)
#            log.info('EPOCH {} ...'.format(i+1))
#            log.info('Validation Accuracy = {:.3f}'.format(valid_accuracy))
#            log.info('')
#
#        saver.save(sess, './lenet')
#        log.info('Model saved')

#    # Calculate and display accuracy for the model on the test data.
#    saver = tf.train.Saver()
#    with tf.Session() as sess:
#        saver.restore(sess, tf.train.latest_checkpoint('.'))
#
#        test_accuracy = evaluate(X_test, y_test,
#                                 x, y, keep_prob, accuracy_operation)
#        log.info("Test Accuracy = {:.3f}".format(test_accuracy))
#
    #
    # Additional test data.
    #

    # Load additional test images and labels.
    X_test_2 = []
    y_test_2 = []
    image_labels_file = './test_images/image_labels.csv'
    image_labels = pd.read_csv(image_labels_file, index_col=['ImageId'])
    for index, row in image_labels.iterrows():
        image_file = './test_images/image' + str(index).zfill(2) + '.jpg'
        log.info('image: {}'.format(image_file))
        image = mpimg.imread(image_file)
        image = cv2.resize(image, (32, 32))  # Scale to 32x32x3
        X_test_2.append(image)
        y_test_2.append(row['ClassId'])
    X_test_2 = np.array(X_test_2)
    y_test_2 = np.array(y_test_2)

    # Display
    label_2_dict = {v: k for v, k in enumerate(y_test_2)}

    fig = plt.figure(figsize=(4, 5))
    columns = 3
    rows = 2
    for index, image in enumerate(X_test_2):
        class_id = label_2_dict[index]
        image = image.squeeze()
        ax = fig.add_subplot(rows, columns, index+1)
        ax.set_title('ClassId={}'.format(class_id))
        plt.imshow(image, cmap='gray')
    fig.subplots_adjust(hspace=.1, wspace=.2)
    plt.show()

    # Normalize the new test data.
    X_test_2_normalized = grayscale_and_normalize(X_test_2)

    # Accuracy for each image. The accuracy will be either 0.0 or 1.0.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        for offset in range(len(X_test_2_normalized)):
            end = offset + 1
            batch_x, batch_y = X_test_2_normalized[offset:end], y_test_2[offset:end]
            accuracy = sess.run(accuracy_operation,
                                feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            log.info("Test Accuracy (image{}) = {:.3f}".format(str(offset).zfill(2), accuracy))

    # Test the model on new test data as a set to get overall accuracy.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        test_accuracy = evaluate(X_test_2_normalized, y_test_2,
                                 x, y, keep_prob, accuracy_operation)
        log.info("Test Accuracy (2) = {:.3f}".format(test_accuracy))

    # Top five softmax probabilities for additional test data.
    softmax_logits = tf.nn.softmax(logits)
    k = 5
    top_k_op = tf.nn.top_k(softmax_logits, k=k)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        top_k = sess.run(top_k_op,
                         feed_dict={x: X_test_2_normalized, keep_prob: 1.0})

        fig, ax = plt.subplots(len(X_test_2), 1+k, figsize=(12, 8))
        fig.subplots_adjust(hspace=.4, wspace=.6)

        # top_k[0] == softmax probabilities
        # top_k[1] == index of the best guess
        # top_k[:, i, k] == probability/best guess for image i, rank k
        for i, image in enumerate(X_test_2):
            ax[i][0].axis('off')
            ax[i][0].imshow(image)
            ax[i][0].set_title('Input')
            for k in range(5):
                # Use X_valid, y_valid assuming that this dataset contains
                # at least one instance of each traffic sign class.
                prob  = 100*top_k[0][i][k]
                guess = top_k[1][i][k]
                index = np.argwhere(y_valid == guess)[0]
                ax[i][k+1].axis('off')
                ax[i][k+1].imshow(X_valid[index].squeeze(), cmap='gray')
                ax[i][k+1].set_title('guess: {} ({:.0f}%)'.format(guess, prob))

    # Close the log file.
    for handler in log.handlers[:]:
        handler.close()
        log.removeHandler(handler)


if __name__ == '__main__':
    main(*sys.argv)
