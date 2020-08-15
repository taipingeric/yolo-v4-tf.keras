import xml.etree.ElementTree as ET
import os
from glob import glob

XML_PATH = './dataset/xml'
CLASSES_PATH = './class_names/classes.txt'
TXT_PATH = './dataset/txt/anno.txt'


'''loads the classes'''
def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


classes = get_classes(CLASSES_PATH)
assert len(classes) > 0, 'no class names detected!'
print(f'num classes: {len(classes)}')

# output file
list_file = open(TXT_PATH, 'w')

for path in glob(os.path.join(XML_PATH, '*.xml')):
    in_file = open(path)

    # Parse .xml file
    tree = ET.parse(in_file)
    root = tree.getroot()
    # Write object information to .txt file
    file_name = root.find('filename').text
    print(file_name)
    list_file.write(file_name)
    for obj in root.iter('object'):
        cls = obj.find('name').text 
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
    list_file.write('\n')
list_file.close()
