import numpy as np
import cv2
import os
import json
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, initializers, models
from utils import load_weights, get_detection_data, draw_bbox, voc_ap, draw_plot_func, read_txt_to_list
from config import  yolo_config


def conv(x, filters, kernel_size, downsampling=False, activation='leaky', batch_norm=True):
    def mish(x):
        return x * tf.math.tanh(tf.math.softplus(x))
        # return layers.Lambda(lambda x: x * activations.tanh(K.softplus(x)))(x)

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

def yolov4_head(yolo_neck_outputs, classes, anchors, xyscale):
    bbox0, object_probability0, class_probabilities0, pred_box0 = get_boxes(yolo_neck_outputs[0],
                                                                            anchors=anchors[0, :, :], classes=classes,
                                                                            grid_size=52, strides=8,
                                                                            xyscale=xyscale[0])
    bbox1, object_probability1, class_probabilities1, pred_box1 = get_boxes(yolo_neck_outputs[1],
                                                                            anchors=anchors[1, :, :], classes=classes,
                                                                            grid_size=26, strides=16,
                                                                            xyscale=xyscale[1])
    bbox2, object_probability2, class_probabilities2, pred_box2 = get_boxes(yolo_neck_outputs[2],
                                                                            anchors=anchors[2, :, :], classes=classes,
                                                                            grid_size=13, strides=32,
                                                                            xyscale=xyscale[2])
    x = [bbox0, object_probability0, class_probabilities0, pred_box0,
         bbox1, object_probability1, class_probabilities1, pred_box1,
         bbox2, object_probability2, class_probabilities2, pred_box2]

    return x

class Yolov4(object):
    def __init__(self,
                 weight_path=None,
                 class_name_path='coco_classes.txt',
                 config=yolo_config,
                 ):
        assert config['img_size'][0] == config['img_size'][1], 'not support yet'
        assert config['img_size'][0] % config['strides'][-1] == 0, 'must be a multiple of last stride'
        self.class_names = [line.strip() for line in open(class_name_path).readlines()]
        self.img_size = yolo_config['img_size']
        self.num_classes = len(self.class_names)
        self.weight_path = weight_path
        self.anchors = np.array(yolo_config['anchors']).reshape((3, 3, 2))
        self.xyscale = yolo_config['xyscale']
        self.strides = yolo_config['strides']
        self.output_sizes = [self.img_size[0] // s for s in self.strides]
        self.class_color = {name: list(np.random.random(size=3)*255) for name in self.class_names}
        assert self.num_classes > 0

    def build_model(self, load_pretrained=True):
        input_layer = layers.Input(self.img_size)
        yolov4_output = yolov4_neck(input_layer, self.num_classes)
        self.yolo_model = models.Model(input_layer, yolov4_output)
        if load_pretrained and self.weight_path and self.weight_path.endswith('.weights'):
            load_weights(self.yolo_model, self.weight_path)

        yolov4_output = yolov4_head(yolov4_output, self.num_classes, self.anchors, self.xyscale)
        self.inference_model = models.Model(input_layer, nms(yolov4_output, self.img_size, self.num_classes))  # [boxes, scores, classes, valid_detections]

    def load_model(self, path):
        self.yolo_model = models.load_model(path, compile=False)
        yolov4_output = yolov4_head(self.yolo_model.output, self.num_classes, self.anchors, self.xyscale)
        self.inference_model = models.Model(self.yolo_model.input,
                                            nms(yolov4_output, self.img_size, self.num_classes))  # [boxes, scores, classes, valid_detections]

    def save_model(self, path):
        self.yolo_model.save(path)

    def preprocess_img(self, img):
        img = cv2.resize(img, self.img_size[:2])
        img = img / 255.
        return img

    def predict(self, img_path, random_color=True):
        raw_img = cv2.imread(img_path)
        print('img shape: ', raw_img.shape)
        img = self.preprocess_img(raw_img)
        imgs = np.expand_dims(img, axis=0)
        pred_output = self.inference_model.predict(imgs)
        detections = get_detection_data(img=raw_img,
                                        model_outputs=pred_output,
                                        class_names=self.class_names)
        draw_bbox(raw_img, detections, cmap=self.class_color, random_color=random_color)
        return detections

    def export_gt(self, annotation_path, gt_folder_path):
        with open(annotation_path) as file:
            for line in file:
                line = line.split(' ')
                filename = line[0].split(os.sep)[-1].split('.')[0]
                objs = line[1:]
                # export txt file
                with open(os.path.join(gt_folder_path, filename + '.txt'), 'w') as output_file:
                    for obj in objs:
                        x_min, y_min, x_max, y_max, class_id = [float(o) for o in obj.strip().split(',')]
                        output_file.write(f'{self.class_names[int(class_id)]} {x_min} {y_min} {x_max} {y_max}\n')

    def export_prediction(self, annotation_path, pred_folder_path, img_folder_path, bs=2):
        with open(annotation_path) as file:
            img_paths = [os.path.join(img_folder_path, line.split(' ')[0].split(os.sep)[-1]) for line in file]
            # print(img_paths[:20])
            for batch_idx in tqdm(range(0, len(img_paths), bs)):
                # print(len(img_paths), batch_idx, batch_idx*bs, (batch_idx+1)*bs)
                paths = img_paths[batch_idx:batch_idx+bs]
                # print(paths)
                # read and process img
                imgs = np.zeros((len(paths), *self.img_size))
                raw_img_shapes = []
                for j, path in enumerate(paths):
                    img = cv2.imread(path)
                    raw_img_shapes.append(img.shape)
                    img = self.preprocess_img(img)
                    imgs[j] = img

                # process batch output
                b_boxes, b_scores, b_classes, b_valid_detections = self.inference_model.predict(imgs)
                for k in range(len(paths)):
                    num_boxes = b_valid_detections[k]
                    raw_img_shape = raw_img_shapes[k]
                    boxes = b_boxes[k, :num_boxes]
                    classes = b_classes[k, :num_boxes]
                    scores = b_scores[k, :num_boxes]
                    # print(raw_img_shape)
                    boxes[:, [0, 2]] = (boxes[:, [0, 2]] * raw_img_shape[1])  # w
                    boxes[:, [1, 3]] = (boxes[:, [1, 3]] * raw_img_shape[0])  # h
                    cls_names = [self.class_names[int(c)] for c in classes]
                    # print(raw_img_shape, boxes.astype(int), cls_names, scores)

                    img_path = paths[k]
                    filename = img_path.split(os.sep)[-1].split('.')[0]
                    # print(filename)
                    output_path = os.path.join(pred_folder_path, filename+'.txt')
                    with open(output_path, 'w') as pred_file:
                        for box_idx in range(num_boxes):
                            b = boxes[box_idx]
                            pred_file.write(f'{cls_names[box_idx]} {scores[box_idx]} {b[0]} {b[1]} {b[2]} {b[3]}\n')


    def eval_map(self, gt_folder_path, pred_folder_path, temp_json_folder_path, output_files_path):
        """Process Gt"""
        ground_truth_files_list = glob(gt_folder_path + '/*.txt')
        assert len(ground_truth_files_list) > 0, 'no ground truth file'
        ground_truth_files_list.sort()
        # dictionary with counter per class
        gt_counter_per_class = {}
        counter_images_per_class = {}

        gt_files = []
        for txt_file in ground_truth_files_list:
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            # check if there is a correspondent detection-results file
            temp_path = os.path.join(pred_folder_path, (file_id + ".txt"))
            assert os.path.exists(temp_path), "Error. File not found: {}\n".format(temp_path)
            lines_list = read_txt_to_list(txt_file)
            # create ground-truth dictionary
            bounding_boxes = []
            is_difficult = False
            already_seen_classes = []
            for line in lines_list:
                class_name, left, top, right, bottom = line.split()
                # check if class is in the ignore list, if yes skip
                bbox = left + " " + top + " " + right + " " + bottom
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

            # dump bounding_boxes into a ".json" file
            new_temp_file = os.path.join(temp_json_folder_path, file_id+"_ground_truth.json") #TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            gt_files.append(new_temp_file)
            with open(new_temp_file, 'w') as outfile:
                json.dump(bounding_boxes, outfile)

        gt_classes = list(gt_counter_per_class.keys())
        # let's sort the classes alphabetically
        gt_classes = sorted(gt_classes)
        n_classes = len(gt_classes)
        print(gt_classes, gt_counter_per_class)

        """Process prediction"""

        dr_files_list = sorted(glob(os.path.join(pred_folder_path, '*.txt')))

        for class_index, class_name in enumerate(gt_classes):
            bounding_boxes = []
            for txt_file in dr_files_list:
                # the first time it checks if all the corresponding ground-truth files exist
                file_id = txt_file.split(".txt", 1)[0]
                file_id = os.path.basename(os.path.normpath(file_id))
                temp_path = os.path.join(gt_folder_path, (file_id + ".txt"))
                if class_index == 0:
                    if not os.path.exists(temp_path):
                        error_msg = "Error. File not found: {}\n".format(temp_path)
                        error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                        print(error_msg)
                lines = read_txt_to_list(txt_file)
                for line in lines:
                    try:
                        tmp_class_name, confidence, left, top, right, bottom = line.split()
                    except ValueError:
                        error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                        error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                        error_msg += " Received: " + line
                        print(error_msg)
                    if tmp_class_name == class_name:
                        # print("match")
                        bbox = left + " " + top + " " + right + " " + bottom
                        bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
            # sort detection-results by decreasing confidence
            bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
            with open(temp_json_folder_path + "/" + class_name + "_dr.json", 'w') as outfile:
                json.dump(bounding_boxes, outfile)

        """
         Calculate the AP for each class
        """
        sum_AP = 0.0
        ap_dictionary = {}
        # open file to store the output
        with open(output_files_path + "/output.txt", 'w') as output_file:
            output_file.write("# AP and precision/recall per class\n")
            count_true_positives = {}
            for class_index, class_name in enumerate(gt_classes):
                count_true_positives[class_name] = 0
                """
                 Load detection-results of that class
                """
                dr_file = temp_json_folder_path + "/" + class_name + "_dr.json"
                dr_data = json.load(open(dr_file))

                """
                 Assign detection-results to ground-truth objects
                """
                nd = len(dr_data)
                tp = [0] * nd  # creates an array of zeros of size nd
                fp = [0] * nd
                for idx, detection in enumerate(dr_data):
                    file_id = detection["file_id"]
                    gt_file = temp_json_folder_path + "/" + file_id + "_ground_truth.json"
                    ground_truth_data = json.load(open(gt_file))
                    ovmax = -1
                    gt_match = -1
                    # load detected object bounding-box
                    bb = [float(x) for x in detection["bbox"].split()]
                    for obj in ground_truth_data:
                        # look for a class_name match
                        if obj["class_name"] == class_name:
                            bbgt = [float(x) for x in obj["bbox"].split()]
                            bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                            iw = bi[2] - bi[0] + 1
                            ih = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:
                                # compute overlap (IoU) = area of intersection / area of union
                                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + \
                                     (bbgt[2] - bbgt[0]+ 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                                ov = iw * ih / ua
                                if ov > ovmax:
                                    ovmax = ov
                                    gt_match = obj

                    min_overlap = 0.5
                    if ovmax >= min_overlap:
                        # if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            # true positive
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            # update the ".json" file
                            with open(gt_file, 'w') as f:
                                f.write(json.dumps(ground_truth_data))
                        else:
                            # false positive (multiple detection)
                            fp[idx] = 1
                    else:
                        fp[idx] = 1


                # compute precision/recall
                cumsum = 0
                for idx, val in enumerate(fp):
                    fp[idx] += cumsum
                    cumsum += val
                print('fp ', cumsum)
                cumsum = 0
                for idx, val in enumerate(tp):
                    tp[idx] += cumsum
                    cumsum += val
                print('tp ', cumsum)
                rec = tp[:]
                for idx, val in enumerate(tp):
                    rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
                print('recall ', cumsum)
                prec = tp[:]
                for idx, val in enumerate(tp):
                    prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
                print('prec ', cumsum)

                ap, mrec, mprec = voc_ap(rec[:], prec[:])
                sum_AP += ap
                text = "{0:.2f}%".format(
                    ap * 100) + " = " + class_name + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)

                print(text)
                ap_dictionary[class_name] = ap

                n_images = counter_images_per_class[class_name]
                # lamr, mr, fppi = log_average_miss_rate(np.array(prec), np.array(rec), n_images)
                # lamr_dictionary[class_name] = lamr

                """
                 Draw plot
                """
                if True:
                    plt.plot(rec, prec, '-o')
                    # add a new penultimate point to the list (mrec[-2], 0.0)
                    # since the last line segment (and respective area) do not affect the AP value
                    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                    area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                    plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                    # set window title
                    fig = plt.gcf()  # gcf - get current figure
                    fig.canvas.set_window_title('AP ' + class_name)
                    # set plot title
                    plt.title('class: ' + text)
                    # plt.suptitle('This is a somewhat long figure title', fontsize=16)
                    # set axis titles
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    # optional - set axes
                    axes = plt.gca()  # gca - get current axes
                    axes.set_xlim([0.0, 1.0])
                    axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
                    # Alternative option -> wait for button to be pressed
                    # while not plt.waitforbuttonpress(): pass # wait for key display
                    # Alternative option -> normal display
                    plt.show()
                    # save the plot
                    # fig.savefig(output_files_path + "/classes/" + class_name + ".png")
                    # plt.cla()  # clear axes for next plot

            # if show_animation:
            #     cv2.destroyAllWindows()

            output_file.write("\n# mAP of all classes\n")
            mAP = sum_AP / n_classes
            text = "mAP = {0:.2f}%".format(mAP * 100)
            output_file.write(text + "\n")
            print(text)

        """
         Count total of detection-results
        """
        # iterate through all the files
        det_counter_per_class = {}
        for txt_file in dr_files_list:
            # get lines to list
            lines_list = read_txt_to_list(txt_file)
            for line in lines_list:
                class_name = line.split()[0]
                # check if class is in the ignore list, if yes skip
                # if class_name in args.ignore:
                #     continue
                # count that object
                if class_name in det_counter_per_class:
                    det_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    det_counter_per_class[class_name] = 1
        # print(det_counter_per_class)
        dr_classes = list(det_counter_per_class.keys())

        """
         Plot the total number of occurences of each class in the ground-truth
        """
        if True:
            window_title = "ground-truth-info"
            plot_title = "ground-truth\n"
            plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
            x_label = "Number of objects per class"
            output_path = output_files_path + "/ground-truth-info.png"
            to_show = False
            plot_color = 'forestgreen'
            draw_plot_func(
                gt_counter_per_class,
                n_classes,
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                '',
            )

        """
         Finish counting true positives
        """
        for class_name in dr_classes:
            # if class exists in detection-result but not in ground-truth then there are no true positives in that class
            if class_name not in gt_classes:
                count_true_positives[class_name] = 0
        # print(count_true_positives)

        """
         Plot the total number of occurences of each class in the "detection-results" folder
        """
        if True:
            window_title = "detection-results-info"
            # Plot title
            plot_title = "detection-results\n"
            plot_title += "(" + str(len(dr_files_list)) + " files and "
            count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
            plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
            # end Plot title
            x_label = "Number of objects per class"
            output_path = output_files_path + "/detection-results-info.png"
            to_show = False
            plot_color = 'forestgreen'
            true_p_bar = count_true_positives
            draw_plot_func(
                det_counter_per_class,
                len(det_counter_per_class),
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                true_p_bar
            )

        """
         Draw mAP plot (Show AP's of all classes in decreasing order)
        """
        if True:
            window_title = "mAP"
            plot_title = "mAP = {0:.2f}%".format(mAP * 100)
            x_label = "Average Precision"
            output_path = output_files_path + "/mAP.png"
            to_show = True
            plot_color = 'royalblue'
            draw_plot_func(
                ap_dictionary,
                n_classes,
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                ""
            )

    def predict_raw(self, img_path):
        raw_img = cv2.imread(img_path)
        print('img shape: ', raw_img.shape)
        img = self.preprocess_img(raw_img)
        imgs = np.expand_dims(img, axis=0)
        return self.yolo_model.predict(imgs)

    def predict_nonms(self, img_path, iou_threshold=0.413, score_threshold=0.1):
        raw_img = cv2.imread(img_path)
        print('img shape: ', raw_img.shape)
        img = self.preprocess_img(raw_img)
        imgs = np.expand_dims(img, axis=0)
        yolov4_output = self.yolo_model.predict(imgs)
        output = yolov4_head(yolov4_output, self.num_classes, self.anchors, self.xyscale)
        pred_output = nms(output, self.img_size, self.num_classes, iou_threshold, score_threshold)
        pred_output = [p.numpy() for p in pred_output]
        detections = get_detection_data(img=raw_img,
                                        model_outputs=pred_output,
                                        class_names=self.class_names)
        draw_bbox(raw_img, detections, cmap=self.class_color, random_color=True)
        return detections





def get_boxes(pred, anchors, classes, grid_size, strides, xyscale):
    """

    :param pred:
    :param anchors:
    :param classes:
    :param grid_size:
    :param strides:
    :param xyscale:
    :return:
    """
    pred = tf.reshape(pred,
                      (tf.shape(pred)[0],
                       grid_size,
                       grid_size,
                       3,
                       5 + classes))  # (batch_size, grid_size, grid_size, 3, 5+classes)
    box_xy, box_wh, obj_prob, class_prob = tf.split(
        pred, (2, 2, 1, classes), axis=-1
    )  # (?, 52, 52, 3, 2) (?, 52, 52, 3, 2) (?, 52, 52, 3, 1) (?, 52, 52, 3, 80)

    box_xy = tf.sigmoid(box_xy)  # (?, 52, 52, 3, 2)
    obj_prob = tf.sigmoid(obj_prob)  # (?, 52, 52, 3, 1)
    class_prob = tf.sigmoid(class_prob)  # (?, 52, 52, 3, 80)
    pred_box_xywh = tf.concat((box_xy, box_wh), axis=-1)  # (?, 52, 52, 3, 4)

    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))  # (52, 52) (52, 52)
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # (52, 52, 1, 2)
    grid = tf.cast(grid, dtype=tf.float32)

    box_xy = ((box_xy * xyscale) - 0.5 * (xyscale - 1) + grid) * strides  # (?, 52, 52, 1, 4)

    box_wh = tf.exp(box_wh) * anchors  # (?, 52, 52, 3, 2)
    box_x1y1 = box_xy - box_wh / 2  # (?, 52, 52, 3, 2)
    box_x2y2 = box_xy + box_wh / 2  # (?, 52, 52, 3, 2)
    pred_box_x1y1x2y2 = tf.concat([box_x1y1, box_x2y2], axis=-1)  # (?, 52, 52, 3, 4)
    return pred_box_x1y1x2y2, obj_prob, class_prob, pred_box_xywh
    # pred_box_x1y1x2y2: absolute xy value


def nms(model_ouputs, input_shape, num_class, iou_threshold=0.413, score_threshold=0.3):
    """
    Apply Non-Maximum suppression
    ref: https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression
    :param model_ouputs: yolo model model_ouputs
    :param input_shape: size of input image
    :return: nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections
    """
    bs = tf.shape(model_ouputs[0])[0]
    boxes = tf.zeros((bs, 0, 4))
    confidence = tf.zeros((bs, 0, 1))
    class_probabilities = tf.zeros((bs, 0, num_class))

    for output_idx in range(0, len(model_ouputs), 4):
        output_xy = model_ouputs[output_idx]
        output_conf = model_ouputs[output_idx + 1]
        output_classes = model_ouputs[output_idx + 2]
        boxes = tf.concat([boxes, tf.reshape(output_xy, (bs, -1, 4))], axis=1)
        confidence = tf.concat([confidence, tf.reshape(output_conf, (bs, -1, 1))], axis=1)
        class_probabilities = tf.concat([class_probabilities, tf.reshape(output_classes, (bs, -1, num_class))], axis=1)

    scores = confidence * class_probabilities
    boxes = tf.expand_dims(boxes, axis=-2)
    boxes = boxes / input_shape[0]  # box normalization: relative img size
    print(f'nms iou: {iou_threshold} score: {score_threshold}')
    (nmsed_boxes,      # [bs, max_detections, 4]
     nmsed_scores,     # [bs, max_detections]
     nmsed_classes,    # [bs, max_detections]
     valid_detections  # [batch_size]
     ) = tf.image.combined_non_max_suppression(
        boxes=boxes,  # y1x1, y2x2 [0~1]
        scores=scores,
        max_output_size_per_class=100,
        max_total_size=100,  # max_boxes: Maximum nmsed_boxes in a single img.
        iou_threshold=iou_threshold,  # iou_threshold: Minimum overlap that counts as a valid detection.
        score_threshold=score_threshold,  # # Minimum confidence that counts as a valid detection.
    )
    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections

