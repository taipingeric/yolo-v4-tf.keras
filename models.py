import numpy as np
import cv2
import os
import json
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from custom_layers import yolov4_neck, yolov4_head, nms
from utils import load_weights, get_detection_data, draw_bbox, voc_ap, draw_plot_func, read_txt_to_list
from config import yolo_config
from loss import yolo_loss


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
        # Training
        self.max_boxes = yolo_config['max_boxes']
        self.iou_loss_thresh = yolo_config['iou_loss_thresh']
        self.config = yolo_config
        assert self.num_classes > 0, 'no classes detected!'

        tf.keras.backend.clear_session()
        if yolo_config['num_gpu'] > 1:
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                self.build_model(load_pretrained=True if self.weight_path else False)
        else:
            self.build_model(load_pretrained=True if self.weight_path else False)

    def build_model(self, load_pretrained=True):
        # core yolo model
        input_layer = layers.Input(self.img_size)
        yolov4_output = yolov4_neck(input_layer, self.num_classes)
        self.yolo_model = models.Model(input_layer, yolov4_output)

        # Build training model
        y_true = [
            layers.Input(name='input_2', shape=(52, 52, 3, (self.num_classes + 5))),  # label small boxes
            layers.Input(name='input_3', shape=(26, 26, 3, (self.num_classes + 5))),  # label medium boxes
            layers.Input(name='input_4', shape=(13, 13, 3, (self.num_classes + 5))),  # label large boxes
            layers.Input(name='input_5', shape=(self.max_boxes, 4)),  # true bboxes
        ]
        loss_list = tf.keras.layers.Lambda(yolo_loss, name='yolo_loss',
                                           arguments={'num_classes': self.num_classes,
                                                      'iou_loss_thresh': self.iou_loss_thresh,
                                                      'anchors': self.anchors})([*self.yolo_model.output, *y_true])
        self.training_model = models.Model([self.yolo_model.input, *y_true], loss_list)

        # Build inference model
        yolov4_output = yolov4_head(yolov4_output, self.num_classes, self.anchors, self.xyscale)
        # output: [boxes, scores, classes, valid_detections]
        self.inference_model = models.Model(input_layer,
                                            nms(yolov4_output, self.img_size, self.num_classes,
                                                iou_threshold=self.config['iou_threshold'],
                                                score_threshold=self.config['score_threshold']))

        if load_pretrained and self.weight_path and self.weight_path.endswith('.weights'):
            if self.weight_path.endswith('.weights'):
                load_weights(self.yolo_model, self.weight_path)
                print(f'load from {self.weight_path}')
            elif self.weight_path.endswith('.h5'):
                self.training_model.load_weights(self.weight_path)
                print(f'load from {self.weight_path}')

        self.training_model.compile(optimizer=optimizers.Adam(lr=1e-3),
                                    loss={'yolo_loss': lambda y_true, y_pred: y_pred})

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

    def fit(self, train_data_gen, epochs, val_data_gen=None, initial_epoch=0, callbacks=None):
        self.training_model.fit(train_data_gen,
                                steps_per_epoch=len(train_data_gen),
                                validation_data=val_data_gen,
                                validation_steps=len(val_data_gen),
                                epochs=epochs,
                                callbacks=callbacks,
                                initial_epoch=initial_epoch)
    # raw_img: RGB
    def predict_img(self, raw_img, random_color=True, plot_img=True, figsize=(10, 10), show_text=True, return_output=False):
        print('img shape: ', raw_img.shape)
        img = self.preprocess_img(raw_img)
        imgs = np.expand_dims(img, axis=0)
        pred_output = self.inference_model.predict(imgs)
        detections = get_detection_data(img=raw_img,
                                        model_outputs=pred_output,
                                        class_names=self.class_names)

        output_img = draw_bbox(raw_img, detections, cmap=self.class_color, random_color=random_color, figsize=figsize,
                  show_text=show_text, show_img=plot_img)
        if return_output:
            return output_img, detections
        else:
            return detections

    def predict(self, img_path, random_color=True, plot_img=True, figsize=(10, 10), show_text=True):
        raw_img = cv2.imread(img_path)[:, :, ::-1]
        return self.predict_img(raw_img, random_color, plot_img, figsize, show_text)

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
                        error_msg = f"Error. File not found: {temp_path}\n"
                        print(error_msg)
                lines = read_txt_to_list(txt_file)
                for line in lines:
                    try:
                        tmp_class_name, confidence, left, top, right, bottom = line.split()
                    except ValueError:
                        error_msg = f"""Error: File {txt_file} in the wrong format.\n 
                                        Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n 
                                        Received: {line} \n"""
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

