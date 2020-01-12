import os
import numpy as np

from utils import yaml_utils
from yolo3.model import YOLO3

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # dataset
    batch_size = 8
    data_shape = np.array((416, 416))  # multiple of 32, hw
    dataset_name = 'football'
    anchors = np.array(yaml_utils.read('configs/yolo3/anchors.yaml'))
    classes = yaml_utils.read('dataset/yolo3/{}/classes.yaml'.format(dataset_name))
    train_dataset = yaml_utils.read('dataset/yolo3/{}/train_dataset.yaml'.format(dataset_name))
    eval_dataset = yaml_utils.read('dataset/yolo3/{}/eval_dataset.yaml'.format(dataset_name))
    num_layers = len(anchors) // 3  # Different detection scales   y1,y2,y3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # model
    yolo3 = YOLO3(data_shape, classes, num_layers, anchor_mask, anchors, './_checkpoints/yolov3/')
    yolo3.build_model()
    yolo3.train_model(dataset_name, batch_size, train_dataset, eval_dataset)
