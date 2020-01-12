import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from utils import yaml_utils
from yolo3.model import YOLO3

if __name__ == '__main__':
    # dataset
    image_path = './dataset/yolo3/football/img2.jpg'
    data_shape = np.array((416, 416))  # multiple of 32, hw
    dataset_name = 'coco2017'
    anchors = np.array(yaml_utils.read('configs/yolo3/anchors.yaml'))
    classes = yaml_utils.read('dataset/yolo3/{}/classes.yaml'.format(dataset_name))
    num_layers = len(anchors) // 3  # Different detection scales   y1,y2,y3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # model
    yolo3 = YOLO3(data_shape, classes, num_layers, anchor_mask, anchors, './_checkpoints/yolov3/')
    yolo3.build_eval_model(dataset_name, score_threshold=0.3, iou_threshold=0.45)
    image = cv2.imread(image_path)
    image = Image.fromarray(image)
    objects, _ = yolo3.detect_image(image)
    image = np.array(image)
    for obj in objects:
        bounding_box = obj['bounding_box']
        cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0, 0, 255),
                      thickness=2)
    output_dir = Path('./_results') / dataset_name
    output_dir.mkdir(exist_ok=True, parents=True)
    cv2.imshow('source', cv2.imread(image_path))
    cv2.imshow('result_image', image)  # 展示图片
    cv2.waitKey()
    cv2.imwrite(str(output_dir / 'result.png'), image)
