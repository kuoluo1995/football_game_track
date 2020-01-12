import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

from utils import yaml_utils

train_scale = 0.99
classes = {'ball': 0, 'messi': 1}
data_dir = '/home/yf/dataset/football/'# E:/Dataset/football/
bounding_boxes = yaml_utils.read(data_dir + 'dataset.yaml')
output_dir = Path('../../dataset/yolo3/football/')

dataset = list()
for item in bounding_boxes:
    image_path = data_dir + 'images/' + item['name'] + '.jpg'
    image = cv2.imread(image_path)
    image_shape = item['shape']
    objects = list()
    for _obj in item['bounding_box']:
        bndbox = {'xmin': _obj['shape'][0], 'ymin': _obj['shape'][1], 'xmax': _obj['shape'][2],
                  'ymax': _obj['shape'][3]}
        cv2.rectangle(image, (bndbox['xmin'], bndbox['ymin']), (bndbox['xmax'], bndbox['ymax']), (0, 0, 255),
                      thickness=2)
        objects.append({'class_id': classes[_obj['name']], 'bndbox': bndbox})
    dataset.append({'image_path': image_path, 'size': image_shape, 'objects': objects})
    # cv2.imshow('image', image)  # 展示图片
    # cv2.waitKey(1)
output_dir.mkdir(parents=True, exist_ok=True)
np.random.shuffle(dataset)
train_steps = int(len(dataset) * train_scale)
yaml_utils.write(output_dir / 'train_dataset.yaml', dataset[:train_steps])
yaml_utils.write(output_dir / 'eval_dataset.yaml', dataset[train_steps:])
yaml_utils.write(output_dir / 'classes.yaml', classes)
