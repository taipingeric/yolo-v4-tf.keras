import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import Sequence


def load_weights(model, weights_file_path):
    conv_layer_size = 110
    conv_output_idxs = [93, 101, 109]
    with open(weights_file_path, 'rb') as file:
        major, minor, revision, seen, _ = np.fromfile(file, dtype=np.int32, count=5)

        bn_idx = 0
        for conv_idx in range(conv_layer_size):
            conv_layer_name = f'conv2d_{conv_idx}' if conv_idx > 0 else 'conv2d'
            bn_layer_name = f'batch_normalization_{bn_idx}' if bn_idx > 0 else 'batch_normalization'

            conv_layer = model.get_layer(conv_layer_name)
            filters = conv_layer.filters
            kernel_size = conv_layer.kernel_size[0]
            input_dims = conv_layer.input_shape[-1]

            if conv_idx not in conv_output_idxs:
                # darknet bn layer weights: [beta, gamma, mean, variance]
                bn_weights = np.fromfile(file, dtype=np.float32, count=4 * filters)
                # tf bn layer weights: [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = model.get_layer(bn_layer_name)
                bn_idx += 1
            else:
                conv_bias = np.fromfile(file, dtype=np.float32, count=filters)

            # darknet shape: (out_dim, input_dims, height, width)
            # tf shape: (height, width, input_dims, out_dim)
            conv_shape = (filters, input_dims, kernel_size, kernel_size)
            conv_weights = np.fromfile(file, dtype=np.float32, count=np.product(conv_shape))
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if conv_idx not in conv_output_idxs:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        if len(file.read()) == 0:
            print('all weights read')
        else:
            print(f'failed to read  all weights, # of unread weights: {len(file.read())}')


def get_detection_data(img, model_outputs, class_names):
    """

    :param img: target raw image
    :param model_outputs: outputs from inference_model
    :param class_names: list of object class names
    :return:
    """

    num_bboxes = model_outputs[-1][0]
    boxes, scores, classes = [output[0][:num_bboxes] for output in model_outputs[:-1]]

    h, w = img.shape[:2]
    df = pd.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2'])
    df[['x1', 'x2']] = (df[['x1', 'x2']] * w).astype('int64')
    df[['y1', 'y2']] = (df[['y1', 'y2']] * h).astype('int64')
    df['class_name'] = np.array(class_names)[classes.astype('int64')]
    df['score'] = scores
    df['w'] = df['x2'] - df['x1']
    df['h'] = df['y2'] - df['y1']

    print(f'# of bboxes: {num_bboxes}')
    return df


def draw_bbox(img, detections, cmap, random_color=True, plot_img=True):
    """
    Draw bounding boxes on the img.
    :param img: BGR img.
    :param detections: pandas DataFrame containing detections
    :param random_color: assign random color for each objects
    :param cmap: object colormap
    :param plot_img: if plot img with bboxes
    :return: None
    """
    img = img[:, :, ::-1].copy()  # BGR -> RGB for plot img
    for _, row in detections.iterrows():
        x1, y1, x2, y2, cls, score, w, h = row.values
        color = list(np.random.random(size=3) * 255) if random_color else cmap[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f'{cls} {score:.2f}'
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.4
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
        cv2.rectangle(img, (x1 - 1, y1 - text_height - 5), (x1 + text_width, y1 - 2), color, cv2.FILLED)
        cv2.putText(img, text, (x1, y1 - text_height // 2), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    if plot_img:
        plt.figure(figsize=(20, 20))
        plt.imshow(img)
        plt.show()
    return img


class DataGenerator(Sequence):
    """
    Generates data for Keras
    ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self,
                 annotation_lines,
                 batch_size,
                 img_size,
                 folder_path,
                 max_boxes=100,
                 shuffle=True):
        self.annotation_lines = annotation_lines
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_img_size = img_size
        self.indexes = np.arange(len(self.annotation_lines))
        self.folder_path = folder_path
        self.max_boxes = max_boxes
        self.on_epoch_end()

    def __len__(self):
        'number of batches per epoch'
        return int(np.ceil(len(self.annotation_lines) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        idxs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        lines = [self.annotation_lines[i] for i in idxs]

        # Generate data
        X, y = self.__data_generation(lines)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, annotation_lines):
        """
        Generates data containing batch_size samples
        :param annotation_lines:
        :return:
        """

        X = np.empty((len(annotation_lines), *self.target_img_size, 3), dtype=np.float32)
        y = np.empty((len(annotation_lines), self.max_boxes, 5))

        for i, line in enumerate(annotation_lines):
            img_data, box_data = self.get_data(line)
            X[i] = img_data
            y[i] = box_data

        return X, y

    def get_data(self, annotation_line):
        line = annotation_line.split()
        img_path = line[0]
        img = cv2.imread(os.path.join(self.folder_path, img_path))[:, :, ::-1]
        ih, iw = img.shape[:2]
        h, w = self.target_img_size
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        scale_w, scale_h = w / iw, h / ih
        img = cv2.resize(img, (w, h))
        image_data = np.array(img) / 255.

        # correct boxes coordinates
        box_data = np.zeros((self.max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box = box[:self.max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale_w  # + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale_h  # + dy
            box_data[:len(box)] = box

        return image_data, box_data


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(bs, max boxes per img, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), (9, wh)
    num_classes: int

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''

    num_stages = 3  # default setting for yolo, tiny yolo will be 2
    anchor_mask = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    bbox_per_grid = 3

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    true_boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  # (100, 2)
    true_boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]  # (100, 2)
    # Normalize x,y,w, h, relative to img size -> (0~1)
    true_boxes[..., 0:2] = true_boxes_xy/input_shape[::-1]  # xy
    true_boxes[..., 2:4] = true_boxes_wh/input_shape[::-1]  # wh

    bs = true_boxes.shape[0]
    grid_sizes = [input_shape//{0:8, 1:16, 2:32}[stage] for stage in range(num_stages)]
    y_true = [np.zeros((bs,
                        grid_sizes[s][0],
                        grid_sizes[s][1],
                        bbox_per_grid,
                        5+num_classes), dtype='float32')
              for s in range(num_stages)] # [(?, 52, 52, 3, 5+num_classes) (?, 26, 26, 3, 5+num_classes)  (?, 13, 13, 3, 5+num_classes) ]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)  # (1, 9 , 2)
    anchor_maxes = anchors / 2.  # (1, 9 , 2)
    anchor_mins = -anchor_maxes  # (1, 9 , 2)
    valid_mask = true_boxes_wh[..., 0] > 0  # (1, 100)

    for batch_idx in range(bs):
        # Discard zero rows.
        wh = true_boxes_wh[batch_idx, valid_mask[batch_idx]]  # (# of bbox, 2)
        num_boxes = len(wh)
        if num_boxes == 0: continue
        wh = np.expand_dims(wh, -2)  # (# of bbox, 1, 2)
        box_maxes = wh / 2.  # (# of bbox, 1, 2)
        box_mins = -box_maxes  # (# of bbox, 1, 2)

        # Compute IoU between each anchors and true boxes for responsibility assignment
        intersect_mins = np.maximum(box_mins, anchor_mins)  # (# of bbox, 9, 2)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = np.prod(intersect_wh, axis=-1)  # (9,)
        box_area = wh[..., 0] * wh[..., 1]  # (# of bbox, 1)
        anchor_area = anchors[..., 0] * anchors[..., 1]  # (1, 9)
        iou = intersect_area / (box_area + anchor_area - intersect_area)  # (# of bbox, 9)

        # Find best anchor for each true box
        best_anchors = np.argmax(iou, axis=-1)  # (# of bbox,)
        for box_idx in range(num_boxes):
            best_anchor = best_anchors[box_idx]
            for stage in range(num_stages):
                if best_anchor in anchor_mask[stage]:
                    # Grid Index
                    grid_col = np.floor(true_boxes[batch_idx, box_idx, 0]*grid_sizes[stage][1]).astype('int32')
                    grid_row = np.floor(true_boxes[batch_idx, box_idx, 1]*grid_sizes[stage][0]).astype('int32')
                    anchor_idx = anchor_mask[stage].index(best_anchor)
                    class_idx = true_boxes[batch_idx, box_idx, 4].astype('int32')
                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 0:4] = true_boxes[batch_idx,box_idx, 0:4]  # bbox
                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 4] = 1  # confidence
                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 5+class_idx] = 1  # one-hot encoding

    return y_true