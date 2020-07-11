import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras import activations, layers, regularizers, initializers, models
import tensorflow.keras.backend as K
from utils import load_weights, get_detection_data, draw_bbox


def conv(x, filters, kernel_size, downsampling=False, activation='leaky', batch_norm=True):
    def mish(x):
        return x * activations.tanh(K.softplus(x))

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
                      kernel_regularizer=regularizers.l2(0.0005),
                      kernel_initializer=initializers.RandomNormal(stddev=0.01),
                      bias_initializer=initializers.Zeros())(x)
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


def darknet53(x):
    x = conv(x, 32, 3)
    x = conv(x, 64, 3, downsampling=True)

    for i in range(1):
        x = residual_block(x, 32, 64)
    x = conv(x, 128, 3, downsampling=True)

    for i in range(2):
        x = residual_block(x, 64, 128)
    x = conv(x, 256, 3, downsampling=True)

    for i in range(8):
        x = residual_block(x, 128, 256)
    route_1 = x
    x = conv(x, 512, 3, downsampling=True)

    for i in range(8):
        x = residual_block(x, 256, 512)
    route_2 = x
    x = conv(x, 1024, 3, downsampling=True)

    for i in range(4):
        x = residual_block(x, 512, 1024)

    return route_1, route_2, x


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
    x = conv(x, 512, 1)
    x = conv(x, 1024, 3)
    x = conv(x, 512, 1)

    x = layers.Concatenate()([
        layers.MaxPooling2D(pool_size=13, strides=1, padding='same')(x),
        layers.MaxPooling2D(pool_size=9, strides=1, padding='same')(x),
        layers.MaxPooling2D(pool_size=5, strides=1, padding='same')(x),
        x
    ])
    x = conv(x, 512, 1)
    x = conv(x, 1024, 3)
    route2 = conv(x, 512, 1)
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

def yolov4_head(yolo_neck_outputs, anchors, xyscale):
    bbox0, object_probability0, class_probabilities0, pred_box0 = get_boxes(yolo_neck_outputs[0],
                                                                            anchors=anchors[0, :, :], classes=80,
                                                                            grid_size=52, strides=8,
                                                                            xyscale=xyscale[0])
    bbox1, object_probability1, class_probabilities1, pred_box1 = get_boxes(yolo_neck_outputs[1],
                                                                            anchors=anchors[1, :, :], classes=80,
                                                                            grid_size=26, strides=16,
                                                                            xyscale=xyscale[1])
    bbox2, object_probability2, class_probabilities2, pred_box2 = get_boxes(yolo_neck_outputs[2],
                                                                            anchors=anchors[2, :, :], classes=80,
                                                                            grid_size=13, strides=32,
                                                                            xyscale=xyscale[2])
    x = [bbox0, object_probability0, class_probabilities0, pred_box0,
         bbox1, object_probability1, class_probabilities1, pred_box1,
         bbox2, object_probability2, class_probabilities2, pred_box2]

    return x

class Yolov4(object):
    def __init__(self,
                 weight_path=None,
                 img_size=(416, 416, 3),
                 anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                 strides=[8, 16, 32],
                 output_sizes=[52, 26, 13],
                 xyscale=[1.2, 1.1, 1.05],
                 class_name_path='coco_classes.txt',
                 ):
        self.class_names = [line.strip() for line in open(class_name_path).readlines()]
        self.img_size = img_size
        self.num_classes = len(self.class_names)
        self.weight_path = weight_path
        self.anchors = np.array(anchors).reshape((3, 3, 2))
        self.xyscale = xyscale
        self.strides = strides
        self.output_sizes = output_sizes
        self.class_color = {name: list(np.random.random(size=3)*255) for name in self.class_names}
        assert self.num_classes > 0

    def build_model(self, load_pretrained=True):
        input_layer = layers.Input(self.img_size)
        yolov4_output = yolov4_neck(input_layer, self.num_classes)
        self.yolo_model = models.Model(input_layer, yolov4_output)

        if load_pretrained and self.weight_path and self.weight_path.endswith('.weights'):
            load_weights(self.yolo_model, self.weight_path)

        yolov4_output = yolov4_head(yolov4_output, self.anchors, self.xyscale)
        self.inference_model = models.Model(input_layer, nms(yolov4_output))  # [boxes, scores, classes, valid_detections]

    def preprocess_img(self, img):
        img = cv2.resize(img, self.img_size[:2])
        img = img / 255.
        return img

    def predict(self, img_path):
        raw_img = cv2.imread(img_path)
        print('img shape: ', raw_img.shape)
        img = self.preprocess_img(raw_img)
        imgs = np.expand_dims(img, axis=0)
        pred_output = self.inference_model.predict(imgs)
        detections = get_detection_data(img=raw_img,
                                        model_outputs=pred_output,
                                        class_names=self.class_names)
        draw_bbox(raw_img, detections, cmap=self.class_color)
        return detections





def get_boxes(pred, anchors, classes, grid_size, strides, xyscale):
    #     grid_size = tf.shape(pred)[1]
    ##
    pred = tf.reshape(pred,
                      (tf.shape(pred)[0],
                       grid_size,
                       grid_size,
                       3,
                       5 + classes))  # (batch_size, grid_size, grid_size, 3, 5+classes)
    ##
    box_xy, box_wh, object_probability, class_probabilities = tf.split(
        pred, (2, 2, 1, classes), axis=-1
    )  # (?, 52, 52, 3, 2) (?, 52, 52, 3, 2) (?, 52, 52, 3, 1) (?, 52, 52, 3, 80)

    box_xy = tf.sigmoid(box_xy)  # (?, 52, 52, 3, 2)
    object_probability = tf.sigmoid(object_probability)  # (?, 52, 52, 3, 1)
    class_probabilities = tf.sigmoid(class_probabilities)  # (?, 52, 52, 3, 80)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # (?, 52, 52, 3, 4)

    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))  # (52, 52) (52, 52)
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # (52, 52, 1, 2)
    grid = tf.cast(grid, dtype=tf.float32)

    box_xy = ((box_xy * xyscale) - 0.5 * (xyscale - 1) + grid) * strides

    box_wh = tf.exp(box_wh) * anchors
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
    return bbox, object_probability, class_probabilities, pred_box


def pre_nms(outputs):
    bs = tf.shape(outputs[0])[0]
    boxes = tf.zeros((bs, 0, 4))
    confidence = tf.zeros((bs, 0, 1))
    class_probabilities = tf.zeros((bs, 0, 80))

    for output_idx in range(0, len(outputs), 4):
        output_xy = outputs[output_idx]
        output_conf = outputs[output_idx + 1]
        output_classes = outputs[output_idx + 2]
        boxes = tf.concat([boxes, tf.reshape(output_xy, (bs, -1, 4))], axis=1)
        confidence = tf.concat([confidence, tf.reshape(output_conf, (bs, -1, 1))], axis=1)
        class_probabilities = tf.concat([class_probabilities, tf.reshape(output_classes, (bs, -1, 80))], axis=1)

    scores = confidence * class_probabilities
    boxes = tf.expand_dims(boxes, axis=-2)
    boxes = boxes / 416
    return boxes, scores

def nms(model_ouputs):
    """
    Apply Non-Maximum suppression
    ref: https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression
    :param model_ouputs: yolo model model_ouputs
    :return: nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections
    """
    output_boxes, output_scores = pre_nms(model_ouputs)

    (nmsed_boxes,      # [bs, max_detections, 4]
     nmsed_scores,     # [bs, max_detections]
     nmsed_classes,    # [bs, max_detections]
     valid_detections  # [batch_size]
     ) = tf.image.combined_non_max_suppression(
        boxes=output_boxes,  # y1x1, y2x2 [0~1]
        scores=output_scores,
        max_output_size_per_class=100,
        max_total_size=100,  # max_boxes: Maximum nmsed_boxes in a single img.
        iou_threshold=0.413,  # iou_threshold: Minimum overlap that counts as a valid detection.
        score_threshold=0.5,  # # Minimum confidence that counts as a valid detection.
    )
    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections
