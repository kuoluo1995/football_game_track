import numpy as np
import keras.backend as K


def box_iou(box1, box2):
    # box1 = [grid_i, grid_j, num_grid_anchors, 5 + num_classes]) 5 is w,y,w,h,c
    # box2 = [, 4] 4 is x,y,w,h
    box1 = K.expand_dims(box1, -2)
    box1_xy = box1[..., :2]
    box1_wh = box1[..., 2:4]
    box1_wh_half = box1_wh / 2.
    box1_min = box1_xy - box1_wh_half
    box1_max = box1_xy + box1_wh_half

    # Expand dim to apply broadcasting.
    box2 = K.expand_dims(box2, 0)
    box2_xy = box2[..., :2]
    box2_wh = box2[..., 2:4]
    box2_wh_half = box2_wh / 2.
    box2_min = box2_xy - box2_wh_half
    box2_max = box2_xy + box2_wh_half

    intersect_min = K.maximum(box1_min, box2_min)
    intersect_max = K.minimum(box1_max, box2_max)
    intersect_wh = K.maximum(intersect_max - intersect_min, 0.)

    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box1_area = box1_wh[..., 0] * box1_wh[..., 1]
    box2_area = box2_wh[..., 0] * box2_wh[..., 1]

    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou


def get_values_by_logits(logits, input_shape, num_classes, grid_anchors):
    grid_shape = K.shape(logits)[1:3]  # height, width
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(logits))

    num_grid_anchors = len(grid_anchors)
    logits = K.reshape(logits, [-1, grid_shape[0], grid_shape[1], num_grid_anchors, 5 + num_classes])

    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(grid_anchors), [1, 1, 1, num_grid_anchors, 2])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(logits[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(logits))
    box_wh = K.exp(logits[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(logits))
    box_confidence = K.sigmoid(logits[..., 4:5])
    box_class_probs = K.sigmoid(logits[..., 5:])
    return grid, logits, box_xy, box_wh, box_confidence, box_class_probs


def correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a