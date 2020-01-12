from pathlib import Path

import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Input, UpSampling2D, Concatenate, Lambda
from keras.models import Model, load_model
from keras.optimizers import Adam

from utils.image_utils import resize_pil_image
from yolo3.data_loader import data_generator
from yolo3.networks import darknet_body, last_layers, conv2d_bn_leaky
from yolo3.tools import get_values_by_logits, box_iou, correct_boxes


class YOLO3:
    def __init__(self, data_shape, classes, num_layers, anchor_mask, anchors, checkpoint_dir):
        self.data_shape = data_shape
        self.num_classes = len(classes)
        self.num_anchors = len(anchors)
        self.num_layers = num_layers
        self.anchor_mask = anchor_mask
        self.anchors = anchors
        self.classes = classes
        self.checkpoint_dir = Path(checkpoint_dir)
        self.sess = K.get_session()

    def build_model(self):
        self.inputs = Input(shape=(None, None, 3))
        labels = [Input(shape=(self.data_shape[0] // (4 * 2 ** (self.num_layers - l)),  # 4 * 2 ** (num_layers - l)
                               self.data_shape[1] // (4 * 2 ** (self.num_layers - l)),  # = [32,16,8]
                               self.num_anchors // self.num_layers,  # num_grid_anchors
                               5 + self.num_classes)) for l in range(self.num_layers)]  # 5 = x,y,w,h,c

        num_grid_anchors = self.num_anchors // self.num_layers
        darknet = Model(self.inputs, darknet_body(self.inputs))
        x, y1 = last_layers(darknet.output, filters=512, out_filters=num_grid_anchors * (5 + self.num_classes))

        x = conv2d_bn_leaky(x, filters=256, kernel_size=(1, 1))
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, darknet.layers[152].output])
        x, y2 = last_layers(x, filters=256, out_filters=num_grid_anchors * (5 + self.num_classes))

        x = conv2d_bn_leaky(x, filters=128, kernel_size=(1, 1))
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, darknet.layers[92].output])
        x, y3 = last_layers(x, filters=128, out_filters=num_grid_anchors * (5 + self.num_classes))

        model_body = Model(self.inputs, [y1, y2, y3])
        model_loss = Lambda(self.loss, output_shape=(1,), name='yolo3_loss')([*model_body.output, *labels])
        self.model = Model([model_body.input, *labels], model_loss)

    def loss(self, inputs, ignore_thresh=0.5):
        logits = inputs[:self.num_layers]
        labels = inputs[self.num_layers:]
        input_shape = K.cast(K.shape(logits[0])[1:3] * 32, K.dtype(labels[0]))  # h,w
        grid_shapes = [K.cast(K.shape(logits[l])[1:3], K.dtype(labels[0])) for l in range(self.num_layers)]

        batch_size = K.shape(logits[0])[0]
        batch_size_float = K.cast(batch_size, K.dtype(logits[0]))

        loss = 0
        for l in range(self.num_layers):
            label_confidence = labels[l][..., 4:5]
            label_classes = labels[l][..., 5:]

            grid, raw_pred, pred_xy, pred_wh, _, _ = get_values_by_logits(logits[l], input_shape, self.num_classes,
                                                                          self.anchors[self.anchor_mask[l]])
            pred_box = K.concatenate([pred_xy, pred_wh])

            # Darknet raw box to calculate loss.
            label_xy = labels[l][..., :2] * grid_shapes[l][::-1] - grid
            label_wh = K.log(labels[l][..., 2:4] / self.anchors[self.anchor_mask[l]] * input_shape[::-1])
            label_wh = K.switch(label_confidence, label_wh, K.zeros_like(label_wh))  # avoid log(0)=-inf

            # Find ignore mask, iterate over each of batch.
            ignore_mask = tf.TensorArray(K.dtype(labels[0]), size=1, dynamic_size=True)
            label_confidence_bool = K.cast(label_confidence, 'bool')

            def loop_body(batch_id, ignore_mask):
                label_box = tf.boolean_mask(labels[l][batch_id, ..., 0:4], label_confidence_bool[batch_id, ..., 0])
                iou = box_iou(pred_box[batch_id], label_box)  # todo label_box?
                best_iou = K.max(iou, axis=-1)
                ignore_mask = ignore_mask.write(batch_id, K.cast(best_iou < ignore_thresh, K.dtype(label_box)))
                return batch_id + 1, ignore_mask

            _, ignore_mask = K.control_flow_ops.while_loop(lambda batch_id, *args: batch_id < batch_size, loop_body,
                                                           [0, ignore_mask])
            ignore_mask = ignore_mask.stack()
            ignore_mask = K.expand_dims(ignore_mask, -1)

            # Large boxes give small weights, small boxes give large weights,
            # because the xywh of large boxes does not need to be learned so well,
            # while the small boxes are very sensitive to xywh
            # In order to adjust the proportion of loss of prediction boxes of different sizes,
            # the smaller the true value box is, the larger the box_loss_scale is.
            # The smaller the box, the larger the loss ratio is.
            box_loss_weights = 2 - labels[l][..., 2:3] * labels[l][..., 3:4]

            # K.binary_crossentropy is helpful to avoid exp overflow.
            xy_loss = label_confidence * box_loss_weights * K.binary_crossentropy(label_xy, raw_pred[..., 0:2],
                                                                                  from_logits=True)
            wh_loss = label_confidence * box_loss_weights * 0.5 * K.square(label_wh - raw_pred[..., 2:4])
            confidence_loss = label_confidence * \
                              K.binary_crossentropy(label_confidence, raw_pred[..., 4:5], from_logits=True) + \
                              (1 - label_confidence) * \
                              K.binary_crossentropy(label_confidence, raw_pred[..., 4:5],
                                                    from_logits=True) * ignore_mask
            class_loss = label_confidence * K.binary_crossentropy(label_classes, raw_pred[..., 5:], from_logits=True)

            xy_loss = K.sum(xy_loss) / batch_size_float
            wh_loss = K.sum(wh_loss) / batch_size_float
            confidence_loss = K.sum(confidence_loss) / batch_size_float
            class_loss = K.sum(class_loss) / batch_size_float
            loss += xy_loss + wh_loss + confidence_loss + class_loss
        return loss

    def train_model(self, dataset_name, batch_size, train_dataset, eval_dataset):
        # dataset
        train_generator = data_generator(train_dataset, batch_size, self.data_shape, self.num_classes, self.num_layers,
                                         self.anchors,
                                         self.anchor_mask, True)
        eval_generator = data_generator(eval_dataset, batch_size, self.data_shape, self.num_classes, self.num_layers,
                                        self.anchors,
                                        self.anchor_mask, False)

        # K.clear_session()  # get a new session
        self.model.compile(optimizer=Adam(lr=1e-3), loss={'yolo3_loss': lambda y_true, y_pred: y_pred})

        # hook
        log_dir = '../tensorboard_logs/yolo3/' + dataset_name + '/'
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        logging = TensorBoard(log_dir=log_dir)
        checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss',
                                     save_weights_only=True, save_best_only=True, period=3)

        # Train with frozen layers first, to get a stable loss.
        self.model.fit_generator(train_generator, steps_per_epoch=max(1, len(train_dataset) // batch_size),
                                 validation_data=eval_generator,
                                 validation_steps=max(1, len(eval_dataset) // batch_size),
                                 epochs=100, initial_epoch=0, callbacks=[logging, checkpoint])
        checkpoint_path = self.checkpoint_dir / dataset_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(self.checkpoint_dir / dataset_name / 'yolov3.weights'))
        self.model.save(str(self.checkpoint_dir / dataset_name / 'yolov3.h5'))

    def build_eval_model(self, dataset_name, score_threshold=0.6, iou_threshold=0.5):
        try:
            self.model = load_model(str(self.checkpoint_dir / dataset_name / 'yolov3.h5'), compile=False)
        except:
            self.build_model()
            self.model.load_weights(str(self.checkpoint_dir / dataset_name / 'yolov3.weights'))

        self.input_image_shape = K.placeholder(shape=(2,))
        logits = self.model.output
        input_shape = K.shape(logits[0])[1:3] * 32
        boxes_list = []
        box_scores_list = []
        for l in range(self.num_layers):
            _, _, pred_xy, pred_wh, pred_confidence, pred_class_probs = \
                get_values_by_logits(logits[l], input_shape, self.num_classes, self.anchors[self.anchor_mask[l]])
            _boxes = correct_boxes(pred_xy, pred_wh, input_shape, self.input_image_shape)
            _boxes = K.reshape(_boxes, [-1, 4])
            _box_scores = pred_confidence * pred_class_probs
            _box_scores = K.reshape(_box_scores, [-1, self.num_classes])
            boxes_list.append(_boxes)
            box_scores_list.append(_box_scores)
        boxes = K.concatenate(boxes_list, axis=0)
        box_scores = K.concatenate(box_scores_list, axis=0)

        mask = box_scores >= score_threshold
        max_boxes_tensor = K.constant(self.num_classes, dtype='int32')

        out_boxes = list()
        out_scores = list()
        out_classes = list()
        for class_id in range(self.num_classes):
            class_boxes = tf.boolean_mask(boxes, mask[:, class_id])
            class_box_scores = tf.boolean_mask(box_scores[:, class_id], mask[:, class_id])
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,
                                                     iou_threshold=iou_threshold)
            class_boxes = K.gather(class_boxes, nms_index)
            class_box_scores = K.gather(class_box_scores, nms_index)
            classes = K.ones_like(class_box_scores, 'int32') * class_id
            out_boxes.append(class_boxes)
            out_scores.append(class_box_scores)
            out_classes.append(classes)
        self.out_boxes = K.concatenate(out_boxes, axis=0)
        self.out_scores = K.concatenate(out_scores, axis=0)
        self.out_classes = K.concatenate(out_classes, axis=0)

    def detect_image(self, image):
        # dataset
        image_width, image_height = image.size
        input_shape = (image.height - (image.height % 32), image.width - (image.width % 32))
        scale = min(input_shape[1] / image_width, input_shape[0] / image_height)
        new_height, new_width = int(image_height * scale), int(image_width * scale)
        offset_y, offset_x = (input_shape[0] - new_height) // 2, (input_shape[1] - new_width) // 2
        new_image = resize_pil_image(image, (input_shape[1], input_shape[0]), (new_width, new_height),
                                     (offset_x, offset_y))
        image = np.array(new_image) / 255.
        batch_images = np.expand_dims(image, 0)  # Add batch dimension.

        # test
        out_boxes, out_scores, out_classes = self.sess.run([self.out_boxes, self.out_scores, self.out_classes],
                                                           feed_dict={self.model.input: batch_images,
                                                                      self.input_image_shape: [image_height,
                                                                                               image_width],
                                                                      K.learning_phase(): 0})
        classes = dict()
        for key, item in self.classes.items():
            classes[item] = key
        objects = list()
        boxes = list()
        for i, class_id in reversed(list(enumerate(out_classes))):
            predicted_class = classes[class_id]
            if predicted_class != 'person' and predicted_class != 'ball':
                continue
            box = out_boxes[i]  # miny, minx, maxy, maxx
            score = out_scores[i]
            ymin, xmin, ymax, xmax = box
            ymin = max(0, np.floor(ymin + 0.5).astype('int32'))
            xmin = max(0, np.floor(xmin + 0.5).astype('int32'))
            ymax = min(image_height, np.floor(ymax + 0.5).astype('int32'))
            xmax = min(image_width, np.floor(xmax + 0.5).astype('int32'))
            objects.append(
                {'bounding_box': [xmin, ymin, xmax, ymax], 'class_name': predicted_class, 'class_id': class_id,
                 'score': score})
            left, top, w, h = xmin, ymin, xmax - xmin, ymax - ymin
            boxes.append([left, top, w, h])
        return objects, boxes
