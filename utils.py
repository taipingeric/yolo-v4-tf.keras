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
    print(len(wf.read()))
    if len(wf.read()) == 0:
        print('all read')
    else:
        print(f'failed to read  all data left : {len(wf.read())}')
    wf.close()

def get_detection_data(image, image_name, outputs, class_names):
    """
    Organize predictions of a single image into a pandas DataFrame.
    Args:
        image: Image as a numpy array.
        image_name: str, name to write in the image column.
        outputs: Outputs from inference_model.predict()
        class_names: A list of object class names.

    Returns:
        data: pandas DataFrame with the detections.
    """
    nums = outputs[-1]
    boxes, scores, classes = 3 * [None]
    if isinstance(outputs[0], np.ndarray):
        boxes, scores, classes = [
            item[0][: int(nums)] for item in outputs[:-1]
        ]
    if not isinstance(outputs[0], np.ndarray):
        boxes, scores, classes = [
            item[0][: int(nums)].numpy() for item in outputs[:-1]
        ]
    w, h = np.flip(image.shape[0:2])
    data = pd.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2'])
    data[['x1', 'x2']] = (data[['x1', 'x2']] * w).astype('int64')
    data[['y1', 'y2']] = (data[['y1', 'y2']] * h).astype('int64')
    data['object_name'] = np.array(class_names)[classes.astype('int64')]
    data['image'] = image_name
    data['score'] = scores
    data['image_width'] = w
    data['image_height'] = h
    data = data[
        [
            'image',
            'object_name',
            'x1',
            'y1',
            'x2',
            'y2',
            'score',
            'image_width',
            'image_height',
        ]
    ]
    return data

def draw_on_image(adjusted, detections):
    """
    Draw bounding boxes over the image.
    Args:
        adjusted: BGR image.
        detections: pandas DataFrame containing detections

    Returns:
        None
    """
    adjusted = adjusted.copy()
    for index, row in detections.iterrows():
        img, obj, x1, y1, x2, y2, score, *_ = row.values
        color = {
            class_name: color
            for class_name, color in zip(
                [str(i) for i in range(80)],
                [
                    list(np.random.random(size=3) * 256)
                    for _ in range(80)
                ],
            )
        }[obj]
        # self.box_colors.get(obj)
#         x1 = int(x1 * 416)
#         x2 = int(x2 * 416)
#         y1 = int(y1 * 416)
#         y2 = int(y2 * 416)
        cv2.rectangle(adjusted, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            adjusted,
            f'{obj}-{round(score, 2)}',
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            0.6,
            color,
            1,
        )
    plt.figure(figsize=(20, 20))
    plt.imshow(adjusted)
    plt.show()