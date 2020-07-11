import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt


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
    df = df[['class_name', 'x1', 'y1', 'x2', 'y2', 'score']]  # reorder columns

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
        cls, x1, y1, x2, y2, score = row.values
        color = list(np.random.random(size=3) * 255) if random_color else cmap[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        cv2.putText(img,
                    f'{cls} {score:.2f}',
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA)
    if plot_img:
        plt.figure(figsize=(15, 15))
        plt.imshow(img)
        plt.show()
    return img
