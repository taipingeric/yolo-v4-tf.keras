import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt


def load_weights(model, weights_file, model_name='yolov4', is_tiny=False):
    layer_size = 110
    output_pos = [93, 101, 109]
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):
        #         print(i, j)
        conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' % j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    if len(wf.read()) == 0:
        print('all weights read')
    else:
        print(f'failed to read  all weights, # of unread weights: {len(wf.read())}')
    wf.close()


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
    df = df[['class_name', 'x1', 'y1', 'x2', 'y2', 'score']]  # reorder columns

    print(f'# of bboxes: ', num_bboxes)
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
        cls, x1, y1, x2, y2, score = row.values
        color = list(np.random.random(size=3) * 255) if random_color else cmap[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        cv2.putText(img,
                    f'{cls} {round(score, 2)}',
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    (255, 255, 255),
                    2)
    if plot_img:
        plt.figure(figsize=(20, 20))
        plt.imshow(img)
        plt.show()
    return img
