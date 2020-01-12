import numpy as np
from PIL import Image

from utils import yaml_utils
from utils.image_utils import resize_pil_image, distort_image
from yolo3.tools import rand


def boxes2labels(boxes, input_shape, num_classes, num_layers, anchors, anchor_mask, grid_shapes):
    labels = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes), dtype='float32')
              for l in range(num_layers)]  # 5 is x, y, w, h, confidence
    if len(boxes) > 0:
        boxes_xy = (boxes[..., 0:2] + boxes[..., 2:3]) // 2
        boxes_wh = (boxes[..., 2:4] - boxes[..., 0:2])
        boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        # Expand dim to apply broadcasting to numpy.maximum
        anchors = np.expand_dims(anchors, 0)
        anchors_max = anchors / 2.
        anchors_min = - anchors_max
        anchors_area = anchors[..., 0] * anchors[..., 1]

        boxes_wh = np.expand_dims(boxes_wh, -2)
        boxes_max = boxes_wh / 2.
        boxes_min = - boxes_max
        boxes_area = boxes_wh[..., 0] * boxes_wh[..., 1]

        intersects_min = np.maximum(boxes_min, anchors_min)
        intersects_max = np.minimum(boxes_max, anchors_max)
        intersects_wh = np.maximum(intersects_max - intersects_min, 0.)
        intersects_area = intersects_wh[..., 0] * intersects_wh[..., 1]

        iou = intersects_area / (boxes_area + anchors_area - intersects_area)

        # Find best anchor for each true box
        best_anchor_index = np.argmax(iou, axis=-1)
        for object_type, anchor_index in enumerate(best_anchor_index):
            # object_type is the kinds of object in that image
            for l in range(num_layers):
                if anchor_index in anchor_mask[l]:
                    i = np.floor(grid_shapes[l][1] * boxes[object_type, 0]).astype('int32')
                    j = np.floor(grid_shapes[l][0] * boxes[object_type, 1]).astype('int32')
                    k = anchor_mask[l].index(anchor_index)  # Small scale at each large scale
                    class_id = boxes[object_type, 4].astype('int32')
                    labels[l][j, i, k, 0:4] = boxes[object_type, 0:4]  # x,y,w,h
                    labels[l][j, i, k, 4] = 1  # confidence score
                    labels[l][j, i, k, 5 + class_id] = 1
    return labels


def preprocess_data(data, input_shape, num_classes, num_layers, anchors, anchor_mask, grid_shapes, is_augmented):
    image = Image.open(data['image_path'])

    image_width, image_height = image.size
    objects = data['objects']
    boxes = list()  # xmin, ymin, xmax, ymax, class_id
    if not is_augmented:
        # resize image
        scale = min(input_shape[1] / image_width, input_shape[0] / image_height)
        new_height = int(image_height * scale)
        new_width = int(image_width * scale)
        offset_y = (input_shape[0] - new_height) // 2
        offset_x = (input_shape[1] - new_width) // 2
        new_image = resize_pil_image(image, (input_shape[1], input_shape[0]), (new_width, new_height),
                                     (offset_x, offset_y))
        image = np.array(new_image) / 255.

        # correct boxes
        if len(objects) > 0:
            for obj in objects:
                xmin = obj['bndbox']['xmin'] * scale + offset_x
                ymin = obj['bndbox']['ymin'] * scale + offset_y
                xmax = obj['bndbox']['xmax'] * scale + offset_x
                ymax = obj['bndbox']['ymax'] * scale + offset_y
                class_id = obj['class_id']
                boxes.append([xmin, ymin, xmax, ymax, class_id])
        return image, boxes2labels(np.array(boxes), input_shape, num_classes, num_layers, anchors, anchor_mask,
                                   grid_shapes)

    # resize image
    new_aspect_ratio = input_shape[1] / input_shape[0] * rand(1 - 0.3, 1 + 0.3) / rand(1 - 0.3, 1 + 0.3)
    scale = rand(.25, 2)
    if new_aspect_ratio < 1:
        new_height = int(scale * input_shape[0])
        new_width = int(new_height * new_aspect_ratio)
    else:
        new_width = int(scale * input_shape[1])
        new_height = int(new_width / new_aspect_ratio)
    offset_x = int(rand(0, input_shape[1] - new_width))
    offset_y = int(rand(0, input_shape[0] - new_height))
    flip = rand() < .5
    new_image = resize_pil_image(image, (input_shape[1], input_shape[0]), (new_width, new_height), (offset_x, offset_y),
                                 flip)
    image = np.array(new_image) / 255.

    # distort image
    hue = rand(-0.1, 0.1)
    sat = rand(0.67, 1.5)
    val = rand(0.67, 1.5)
    image = distort_image(image, hue, sat, val)

    # correct boxes
    if len(objects) > 0:
        np.random.shuffle(objects)
        for i, obj in enumerate(objects):
            xmin = obj['bndbox']['xmin'] * new_width / image_width + offset_x
            xmax = obj['bndbox']['xmax'] * new_width / image_width + offset_x
            if flip:
                xmin, xmax = input_shape[1] - xmax, input_shape[1] - xmin
            ymin = obj['bndbox']['ymin'] * new_height / image_height + offset_y
            ymax = obj['bndbox']['ymax'] * new_height / image_height + offset_y
            xmin, ymin = max(xmin, 0), max(ymin, 0)
            xmax, ymax = min(xmax, input_shape[1]), min(ymax, input_shape[0])
            if (xmax - xmin) < 1 or (ymax - ymin) < 1:
                continue
            class_id = obj['class_id']
            boxes.append([xmin, ymin, xmax, ymax, class_id])
    return image, boxes2labels(np.array(boxes), input_shape, num_classes, num_layers, anchors, anchor_mask, grid_shapes)


def data_generator(data_list, batch_size, input_shape, num_classes, num_layers, anchors, anchor_mask, is_augmented):
    '''data generator for fit_generator'''
    data_size = len(data_list)
    grid_shapes = [input_shape // (4 * 2 ** (num_layers - l)) for l in range(num_layers)]
    while True:
        batch_images = []
        batch_labels = [list() for _ in range(num_layers)]
        if is_augmented:
            np.random.shuffle(data_list)
        for i in range(data_size):
            image, labels = preprocess_data(data_list[i], input_shape, num_classes, num_layers, anchors, anchor_mask,
                                            grid_shapes, is_augmented)
            batch_images.append(image)
            for l in range(num_layers):
                batch_labels[l].append(labels[l])
            if len(batch_images) == batch_size:
                batch_labels = [np.array(batch_labels[l]) for l in range(num_layers)]
                yield [np.array(batch_images), *batch_labels], np.zeros(batch_size)
                batch_images = []
                batch_labels = [list() for _ in range(num_layers)]


if __name__ == '__main__':
    dataset_name = 'voc2012'
    anchors = np.array(yaml_utils.read('../configs/yolo3/anchors.yaml'))
    classes = yaml_utils.read('../dataset/yolo3/{}/classes.yaml'.format(dataset_name))
    train_dataset = yaml_utils.read('../dataset/yolo3/{}/train_dataset.yaml'.format(dataset_name))
    eval_dataset = yaml_utils.read('../dataset/yolo3/{}/eval_dataset.yaml'.format(dataset_name))
    data_shape = np.array((416, 416))  # multiple of 32, hw
    batch_size = 3

    # dataset
    train_generator = data_generator(train_dataset, batch_size, data_shape, len(classes), anchors, True)
    train_value = next(train_generator)
    eval_generator = data_generator(eval_dataset, batch_size, data_shape, len(classes), anchors, False)
    eval_value = next(eval_generator)
    print(11)
