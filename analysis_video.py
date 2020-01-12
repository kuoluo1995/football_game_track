import colorsys
import cv2
import numpy as np
from PIL import Image

from deep_sort.detection import Detection
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.preprocessing import non_max_suppression
from deep_sort.tracker import Tracker
from deep_sort.tools import create_box_encoder
from utils import yaml_utils
from yolo3.model import YOLO3

if __name__ == '__main__':
    video_path = './dataset/messi.mp4'
    # yolov3
    # dataset
    data_shape = np.array((416, 416))  # multiple of 32, hw
    dataset_name = 'coco2017'
    anchors = np.array(yaml_utils.read('configs/yolo3/anchors.yaml'))
    classes = yaml_utils.read('dataset/yolo3/{}/classes.yaml'.format(dataset_name))
    num_layers = len(anchors) // 3  # Different detection scales   y1,y2,y3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    yolo3 = YOLO3(data_shape, classes, num_layers, anchor_mask, anchors, './_checkpoints/yolov3/')
    yolo3.build_eval_model(dataset_name, score_threshold=0.3, iou_threshold=0.4)

    # deep sort
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    deep_sort_model = './_checkpoints/deep_sort/mars-small128.pb'
    encoder = create_box_encoder(deep_sort_model, batch_size=1)
    metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # video
    video_capture = cv2.VideoCapture(video_path)
    # Define the codec and create VideoWriter object
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('_results/output.avi', fourcc, 15, (width, height))

    # # Generate colors for drawing bounding boxes.
    num_classes = len(classes) + 1  # 1 is text color
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if not ret:
            break
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        objects, boxs = yolo3.detect_image(image)

        features = encoder(frame, boxs)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature, obj) for bbox, feature, obj in zip(boxs, features, objects)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            yolo3_box = track.obj['bounding_box']
            class_name = track.obj['class_name']
            score = track.obj['score']
            class_id = track.obj['class_id']
            cv2.rectangle(frame, (yolo3_box[0], yolo3_box[1]), (yolo3_box[2], yolo3_box[3]), colors[class_id], 2)
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colors[class_id], 2)
            scale = np.sqrt((yolo3_box[3] - yolo3_box[1]) * (yolo3_box[2] - yolo3_box[0]) / (width * height))
            font_scale = max(0.1, min(0.8, scale * 4))
            cv2.putText(frame, '{}:{} '.format(class_name, track.track_id),
                        (yolo3_box[0] + min(10, int(14 * scale)), yolo3_box[1] + min(20, int(50 * scale))), 0,
                        font_scale, colors[-1], 1)
        cv2.imshow('football_game_video_analysis', frame)
        # save a frame
        out.write(frame)
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
