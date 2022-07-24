import tensorflow as tf
from tensorflow.keras import layers, initializers, models
from utils import DataGenerator, read_annotation_lines
from models import Yolov4
from config import yolo_config

import os
from glob import glob
import numpy as np

def conv(x, filters, kernel_size, downsampling=False, activation='leaky', batch_norm=True):
    def mish(x):
        return x * tf.math.tanh(tf.math.softplus(x))

    if downsampling:
        x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(x)  # top & left padding
        padding = 'valid'
        strides = 2
    else:
        padding = 'same'
        strides = 1
    x = layers.Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=not batch_norm,
                      # kernel_regularizer=regularizers.l2(0.0005),
                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                      # bias_initializer=initializers.Zeros()
                      )(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    if activation == 'mish':
        x = mish(x)
    elif activation == 'leaky':
        x = layers.LeakyReLU(alpha=0.1)(x)
    return x


def residual_block(x, filters1, filters2, activation='leaky'):
    """
    :param x: input tensor
    :param filters1: num of filter for 1x1 conv
    :param filters2: num of filter for 3x3 conv
    :param activation: default activation function: leaky relu
    :return:
    """
    y = conv(x, filters1, kernel_size=1, activation=activation)
    y = conv(y, filters2, kernel_size=3, activation=activation)
    return layers.Add()([x, y])

def csp_block(x, residual_out, repeat, residual_bottleneck=False):
    """
    Cross Stage Partial Network (CSPNet)
    transition_bottleneck_dims: 1x1 bottleneck
    output_dims: 3x3
    :param x:
    :param residual_out:
    :param repeat:
    :param residual_bottleneck:
    :return:
    """
    route = x
    route = conv(route, residual_out, 1, activation="mish")
    x = conv(x, residual_out, 1, activation="mish")
    for i in range(repeat):
        x = residual_block(x,
                           residual_out // 2 if residual_bottleneck else residual_out,
                           residual_out,
                           activation="mish")
    x = conv(x, residual_out, 1, activation="mish")

    x = layers.Concatenate()([x, route])
    return x


def cbl(x, dims):
    x = conv(x, dims[0], 1)
    x = conv(x, dims[1], 3)
    x = conv(x, dims[2], 1)
    return x

def spp(x, pool_sizes):
    x = layers.Concatenate()([layers.MaxPooling2D(pool_size=pool_sizes[0], strides=1, padding='same')(x),
                              layers.MaxPooling2D(pool_size=pool_sizes[1], strides=1, padding='same')(x),
                              layers.MaxPooling2D(pool_size=pool_sizes[2], strides=1, padding='same')(x),
                              x
                              ])
    return x


def cspdarknet53(input):
    x = conv(input, 32, 3)
    x = conv(x, 64, 3, downsampling=True)

    x = csp_block(x, residual_out=64, repeat=1, residual_bottleneck=True)
    x = conv(x, 64, 1, activation='mish')
    x = conv(x, 128, 3, activation='mish', downsampling=True)

    x = csp_block(x, residual_out=64, repeat=2)
    x = conv(x, 128, 1, activation='mish')
    x = conv(x, 256, 3, activation='mish', downsampling=True)

    x = csp_block(x, residual_out=128, repeat=8)
    x = conv(x, 256, 1, activation='mish')
    route0 = x
    x = conv(x, 512, 3, activation='mish', downsampling=True)

    x = csp_block(x, residual_out=256, repeat=8)
    x = conv(x, 512, 1, activation='mish')
    route1 = x
    x = conv(x, 1024, 3, activation='mish', downsampling=True)

    x = csp_block(x, residual_out=512, repeat=4)

    x = conv(x, 1024, 1, activation="mish")

    route2 = x

    return [route0, route1, route2]

    # x = conv(x, 512, 1)
    # x = conv(x, 1024, 3)
    # x = conv(x, 512, 1)
    x = cbl(x, dims=[512, 1024, 512])

    # x = layers.Concatenate()([layers.MaxPooling2D(pool_size=13, strides=1, padding='same')(x),
    #                           layers.MaxPooling2D(pool_size=9, strides=1, padding='same')(x),
    #                           layers.MaxPooling2D(pool_size=5, strides=1, padding='same')(x),
    #                           x
    #                           ])
    x = spp(x, pool_sizes=[13, 9, 5])
    # x = conv(x, 512, 1)
    # x = conv(x, 1024, 3)
    # route2 = conv(x, 512, 1)
    route2 = cbl(x, dims=[512, 1024, 512])

    return models.Model(input, [route0, route1, route2])

def yolov4_neck(x, num_classes):
    backbone_model = cspdarknet53(x)
    route0, route1, route2 = backbone_model.output

    route_input = route2
    x = conv(route2, 256, 1)
    x = layers.UpSampling2D()(x)
    route1 = conv(route1, 256, 1)
    x = layers.Concatenate()([route1, x])

    x = conv(x, 256, 1)
    x = conv(x, 512, 3)
    x = conv(x, 256, 1)
    x = conv(x, 512, 3)
    x = conv(x, 256, 1)

    route1 = x
    x = conv(x, 128, 1)
    x = layers.UpSampling2D()(x)
    route0 = conv(route0, 128, 1)
    x = layers.Concatenate()([route0, x])

    x = conv(x, 128, 1)
    x = conv(x, 256, 3)
    x = conv(x, 128, 1)
    x = conv(x, 256, 3)
    x = conv(x, 128, 1)

    route0 = x
    x = conv(x, 256, 3)
    conv_sbbox = conv(x, 3 * (num_classes + 5), 1, activation=None, batch_norm=False)

    x = conv(route0, 256, 3, downsampling=True)
    x = layers.Concatenate()([x, route1])

    x = conv(x, 256, 1)
    x = conv(x, 512, 3)
    x = conv(x, 256, 1)
    x = conv(x, 512, 3)
    x = conv(x, 256, 1)

    route1 = x
    x = conv(x, 512, 3)
    conv_mbbox = conv(x, 3 * (num_classes + 5), 1, activation=None, batch_norm=False)

    x = conv(route1, 512, 3, downsampling=True)
    x = layers.Concatenate()([x, route_input])

    x = conv(x, 512, 1)
    x = conv(x, 1024, 3)
    x = conv(x, 512, 1)
    x = conv(x, 1024, 3)
    x = conv(x, 512, 1)

    x = conv(x, 1024, 3)
    conv_lbbox = conv(x, 3 * (num_classes + 5), 1, activation=None, batch_norm=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def build_v4neck(inputs, num_classes):
    [route0, route1, route2] = inputs
    route2 = cbl(route2, dims=[512, 1024, 512])
    route2 = spp(route2, pool_sizes=[13, 9, 5])
    route2 = cbl(route2, dims=[512, 1024, 512])

    route_input = route2
    x = conv(route2, 256, 1)
    x = layers.UpSampling2D()(x)
    route1 = conv(route1, 256, 1)
    x = layers.Concatenate()([route1, x])

    x = conv(x, 256, 1)
    x = conv(x, 512, 3)
    x = conv(x, 256, 1)
    x = conv(x, 512, 3)
    x = conv(x, 256, 1)

    route1 = x
    x = conv(x, 128, 1)
    x = layers.UpSampling2D()(x)
    route0 = conv(route0, 128, 1)
    x = layers.Concatenate()([route0, x])

    x = conv(x, 128, 1)
    x = conv(x, 256, 3)
    x = conv(x, 128, 1)
    x = conv(x, 256, 3)
    x = conv(x, 128, 1)

    route0 = x
    x = conv(x, 256, 3)
    conv_sbbox = build_head(x, num_classes)

    x = conv(route0, 256, 3, downsampling=True)
    x = layers.Concatenate()([x, route1])

    x = conv(x, 256, 1)
    x = conv(x, 512, 3)
    x = conv(x, 256, 1)
    x = conv(x, 512, 3)
    x = conv(x, 256, 1)

    route1 = x
    x = conv(x, 512, 3)
    conv_mbbox = build_head(x, num_classes)

    x = conv(route1, 512, 3, downsampling=True)
    x = layers.Concatenate()([x, route_input])

    x = conv(x, 512, 1)
    x = conv(x, 1024, 3)
    x = conv(x, 512, 1)
    x = conv(x, 1024, 3)
    x = conv(x, 512, 1)

    x = conv(x, 1024, 3)
    conv_lbbox = build_head(x, num_classes)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def build_head(x, num_classes):
    return conv(x, 3*(num_classes+5), 1, activation=None, batch_norm=False)

def build_model():
    inputs = layers.Input((416, 416, 3))
    [route0, route1, route2] = cspdarknet53(inputs)
    [conv_sbbox, conv_mbbox, conv_lbbox] = build_v4neck([route0, route1, route2], num_classes=80)

    model = tf.keras.models.Model(inputs, [conv_sbbox, conv_mbbox, conv_lbbox])
    return model


if __name__ == '__main__':
    print('GG')
    FOLDER_PATH = './yolo-v4-tf.keras'
    train_lines, val_lines = read_annotation_lines('./dataset/train_txt/bccd_annotation.txt',
                                                   test_size=0.2)
    IMG_FOLDER_PATH = './dataset/train_img'
    NUM_CLASSES = 3
    # bccd classes names
    class_name_path = os.path.join(FOLDER_PATH, 'class_names/bccd_classes.txt')

    data_gen_train = DataGenerator(train_lines, NUM_CLASSES, IMG_FOLDER_PATH)
    data_gen_val = DataGenerator(val_lines, NUM_CLASSES, IMG_FOLDER_PATH)

    # inputs = tf.random.normal((1, 416, 416, 3))
    # model = build_model()
    # [conv_sbbox, conv_mbbox, conv_lbbox] = model(inputs)
    #
    # print(conv_sbbox.shape)
    # print(conv_mbbox.shape)
    # print(conv_lbbox.shape)





