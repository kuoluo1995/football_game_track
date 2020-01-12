import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

from utils import yaml_utils

train_scale = 0.99
classes = {'person': 0}
dataset_names = ['person_trainval']
data_dir = '/home/yf/dataset/VOC2012'  # E:/Dataset/VOC2012
output_dir = Path('../../dataset/yolo3/voc2012')

dataset = list()
for dataset_name in dataset_names:
    image_ids = open((data_dir + '/ImageSets/Main/{}.txt').format(dataset_name)).read().strip().split('\n')
    for image_id in image_ids:
        if '-1' in image_id:
            continue
        image_id = image_id.split()[0]
        data = dict()  # as one data in dataset
        # read annotation
        annotation_file = open((data_dir + '/Annotations/{}.xml').format(image_id))
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        image_path = (data_dir + '/JPEGImages/{}.jpg').format(image_id)
        data['image_path'] = image_path
        img = cv2.imread(image_path)
        size = root.find('size')
        data['size'] = [int(size.find('height').text), int(size.find('width').text), int(size.find('depth').text)]
        data['objects'] = list()  # save all objects in data
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            name = obj.find('name').text
            if name not in classes.keys() or int(difficult) == 1:
                continue
            item = dict()  # save object params
            item['class_id'] = classes[name]
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            item['bndbox'] = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
            data['objects'].append(item)
        if len(data['objects']) < 1:
            continue
        # cv2.imshow('image', img)  # 展示图片
        # cv2.waitKey(1)
        dataset.append(data)
output_dir.mkdir(parents=True, exist_ok=True)
np.random.shuffle(dataset)
train_steps = int(len(dataset) * train_scale)
yaml_utils.write(output_dir / 'train_dataset.yaml', dataset[:train_steps])
yaml_utils.write(output_dir / 'eval_dataset.yaml', dataset[train_steps:])
yaml_utils.write(output_dir / 'classes.yaml', classes)
