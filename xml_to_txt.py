import xml.etree.ElementTree as ET
import os
from os import getcwd
from glob import glob

FOLDER_PATH = '/Users/SheepLi/Google 雲端硬碟/yolo-v4-tf.keras'
CLASSES_PATH = '/Users/SheepLi/Google 雲端硬碟/yolo-v4-tf.keras/coco_classes.txt'
TXT_PATH = '/Users/SheepLi/Google 雲端硬碟/yolo-v4-tf.keras/dataset/train_txt/anno.txt'

'''loads the classes'''
def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

classes = get_classes(CLASSES_PATH)

# output file
list_file = open(TXT_PATH, 'w')

for path in glob(os.path.join(FOLDER_PATH, 'dataset/train_xml/*.xml')):
    file_id = ''.join(path.split('/')[-1].split('.')[:-1])
    print(file_id)
    in_file = open(path)

    # Parse .xml file
    tree=ET.parse(in_file)
    root = tree.getroot()
    # Write object information to .txt file
    list_file.write(f'dataset/train_img2/{file_id}.jpg') # path
    for obj in root.iter('object'):
        cls = obj.find('name').text 
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
    list_file.write('\n')
list_file.close()