#!/usr/bin/env python
# coding: utf-8



import cv2
import numpy as np
from utils import DataGenerator, preprocess_true_boxes
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import tensorflow as tf

from models import Yolov4, yolov4_head, get_boxes
from config import yolo_config

print(tf.__version__)
# In[3]:


with open('../dataset/train_txt2/anno.txt') as f:
    lines = f.readlines()


# In[4]:


FOLDER_PATH = '..'
BS = 1
anchors = np.array([12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]).reshape((-1, 2))


# In[6]:


data_gen = DataGenerator(lines[:], BS, (416, 416), num_classes=80, folder_path=FOLDER_PATH, anchors=anchors)



model = Yolov4(
                weight_path=None,
                # class_name_path='bccd_classes.txt'
#               weight_path='yolov4.weights',

#                img_size=(416, 416, 3),
            
              )
model.build_model(load_pretrained=False)

print('num class : ', model.num_classes)



# In[29]:


# from tflite yolov4
def bbox_giou(bboxes1, bboxes2):
    """
    Generalized IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat([bboxes1[..., :2] - bboxes1[..., 2:] * 0.5, bboxes1[..., :2] + bboxes1[..., 2:] * 0.5], axis=-1, )
    bboxes2_coor = tf.concat([bboxes2[..., :2] - bboxes2[..., 2:] * 0.5, bboxes2[..., :2] + bboxes2[..., 2:] * 0.5], axis=-1, )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area
    iou = tf.math.divide_no_nan(inter_area, union_area)
    # iou = tf.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)  #tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)
    # print(giou)
    return giou

# def bbox_iou(bboxes1, bboxes2):
#     """
#     @param bboxes1: (a, b, ..., 4)
#     @param bboxes2: (A, B, ..., 4)
#         x:X is 1:n or n:n or n:1
#     @return (max(a,A), max(b,B), ...)
#     ex) (4,):(3,4) -> (3,)
#         (2,1,4):(2,3,4) -> (2,3)
#     """
#     bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
#     bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]
#
#     bboxes1_coor = tf.concat(
#         [
#             bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
#             bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
#         ],
#         axis=-1,
#     )
#     bboxes2_coor = tf.concat(
#         [
#             bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
#             bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
#         ],
#         axis=-1,
#     )
#
#     left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
#     right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])
#
#     inter_section = tf.maximum(right_down - left_up, 0.0)
#     inter_area = inter_section[..., 0] * inter_section[..., 1]
#
#     union_area = bboxes1_area + bboxes2_area - inter_area
#
#     iou = tf.math.divide_no_nan(inter_area, union_area)
#
#     return iou
def bbox_iou(boxes1, boxes2):

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1_coor = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_coor = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up    = tf.maximum(boxes1_coor[..., :2], boxes2_coor[..., :2])
    right_down = tf.minimum(boxes1_coor[..., 2:], boxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = tf.math.divide_no_nan(inter_area, union_area)
    # iou       = tf.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return iou

def yolo_loss_wrapper(input_shape, STRIDES, NUM_CLASS, ANCHORS, XYSCALES, IOU_LOSS_THRESH):
    input_shape = input_shape[0]
    def yolo_loss(y_true, y_pred):
        bboxes_tensor = decode_train2(y_pred, input_shape // STRIDES, NUM_CLASS, STRIDES, ANCHORS, XYSCALES)

        label = y_true
        bboxes = bboxes_tensor
        grid_size = tf.shape(label)[1]
        output_size = grid_size
        input_size = STRIDES * output_size

        conv_raw_conf = bboxes[..., 4:5]
        conv_raw_prob = bboxes[..., 5:]

        pred_xywh = bboxes[:, :, :, :, 0:4]  # abs xy wh
        pred_conf = bboxes[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        # iou = utils.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        #
        # iou = bbox_iou(pred_xywh, bboxes)
        iou = bbox_iou(pred_xywh, label_xywh)

        # max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
        max_iou = tf.reduce_max(iou, axis=-1)[..., tf.newaxis, tf.newaxis]
        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < IOU_LOSS_THRESH, tf.float32)

        conf_focal = tf.pow(respond_bbox - pred_conf, 2)
        conf_focal = 1.

        conf_loss = conf_focal * (
                respond_bbox * tf.keras.losses.binary_crossentropy(respond_bbox, pred_conf)[..., tf.newaxis]  # tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.keras.losses.binary_crossentropy(respond_bbox, pred_conf)[..., tf.newaxis]  #  tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.keras.losses.categorical_crossentropy(label_prob, tf.nn.softmax(conv_raw_prob))[..., tf.newaxis]  #  tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return [giou_loss, conf_loss, prob_loss]
    return yolo_loss

def decode_train2(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, XYSCALE):
    conv_output = tf.reshape(conv_output,
                             (tf.shape(conv_output)[0], output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS),
                                                                          axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE) - 0.5 * (XYSCALE - 1) + xy_grid) * STRIDES
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS)
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

# x_batch, y_batch = X, y = data_gen.__getitem__(0)


losses = [yolo_loss_wrapper(input_shape=(416, 416), 
                  STRIDES=[8, 16, 32][i], 
                  NUM_CLASS=80,
                  ANCHORS=anchors.reshape(3, 3, 2)[i], 
                  XYSCALES=[1., 1., 1.][i], 
                  IOU_LOSS_THRESH=0.5) for i in range(3)]


# # In[31]:
INIT_LR = 1e-6
FINAL_LR = 1e-8
opt = tf.keras.optimizers.Adam(lr=INIT_LR, clipvalue=0.5)
steps_per_epoch = len(lines) // BS
warmup_epochs = 20
warmup_steps = warmup_epochs * steps_per_epoch
global_steps = 0
first_stage_epoch = 200
second_stage_epoch = 300
total_steps = (first_stage_epoch + second_stage_epoch) * steps_per_epoch



def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        predict = model.yolo_model(x_batch)
        total_giou_loss = 0
        total_conf_loss = 0
        total_prob_loss = 0
        # giou_loss + conf_loss + prob_loss
        for i in range(3):
            loss_func = losses[i]
            giou_loss, conf_loss, prob_loss = loss_func(y_batch[i], predict[i])
            total_giou_loss += giou_loss
            total_conf_loss += conf_loss
            total_prob_loss += prob_loss
            print(i, total_giou_loss, total_conf_loss, total_prob_loss)
        total_loss = total_giou_loss + total_conf_loss + total_prob_loss
        gradients = tape.gradient(total_loss, model.yolo_model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.yolo_model.trainable_variables))

for epoch in range(first_stage_epoch+second_stage_epoch):
    if epoch < 20:
        for name in ['conv2d_93', 'conv2d_101', 'conv2d_109']:
            layer = model.yolo_model.get_layer(name)
            layer.trainable = False
    elif epoch >= 20:
        for name in ['conv2d_93', 'conv2d_101', 'conv2d_109']:
            layer = model.yolo_model.get_layer(name)
            layer.trainable = True
    for x_batch, y_batch in data_gen:
        train_step(x_batch, y_batch)
    global_steps += 1
    if global_steps < warmup_steps:
        lr = global_steps / warmup_steps * INIT_LR
    else:
        lr = FINAL_LR + 0.5 * (INIT_LR - FINAL_LR) * (
            (1 + np.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
        )
    opt.lr.assign(lr)
    print(f'epoch: {epoch} lr: ', lr)

# for step in range(10000):
#     with tf.GradientTape() as tape:
#         predict = model.yolo_model(x_batch)
#         total_loss = 0
#         total_giou_loss = 0
#         total_conf_loss = 0
#         total_prob_loss = 0
#         # giou_loss + conf_loss + prob_loss
#         for i in range(3):
#             loss_func = losses[i]
#             giou_loss, conf_loss, prob_loss = loss_func(y_batch[i], predict[i])
#             total_giou_loss += giou_loss
#             total_conf_loss += conf_loss
#             total_prob_loss += prob_loss
#             print(i, total_giou_loss, total_conf_loss, total_prob_loss)
#         total_loss = total_giou_loss + total_conf_loss + total_prob_loss
#         gradients = tape.gradient(total_loss, model.yolo_model.trainable_variables)
#         opt.apply_gradients(zip(gradients, model.yolo_model.trainable_variables))


# model.yolo_model.compile(tf.keras.optimizers.Adam(),
#                          # loss= lo
#                          {'conv2d_93': losses[0],  # [93, 101, 109]
#                                 'conv2d_101': losses[1],
#                                 'conv2d_109': losses[2]})
#
#
#
# model.yolo_model.fit(data_gen, epochs=10000, steps_per_epoch=1)

# In[ ]:

# print(model.yolo_model.evaluate(x_batch, y_batch))
#
# while(1):
#     continue




