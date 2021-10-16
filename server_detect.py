import os
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import config as cfg
import pickle
import communication as com
import struct
import time

HOST, PORT = ('localhost', 6790)


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


configs = (
    config_util
    .get_configs_from_pipeline_file(cfg.files['PIPELINE_CONFIG'])
)

detection_model = model_builder.build(
    model_config=configs['model'],
    is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(cfg.paths['CHECKPOINT_PATH'],
             'ckpt-6')).expect_partial()

category_index = label_map_util.create_category_index_from_labelmap(
    cfg.files['LABELMAP'])

# Video Detection
connect = com.Communication()

print(f'{"[SERVER]":10}Communication starting up')
connect.Host(HOST, PORT)

data = b''
headSize = struct.calcsize('I')

ST = 0
PT = 0

while True:
    data = connect.RecvFirstChunk(headSize, data)

    shutdown = connect.HandleHeaderParams()
    if shutdown:
        cv2.destroyAllWindows()
        break

    msg, data = connect.RecvSecondChunk(headSize, data)
    frame = pickle.loads(msg)

    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0),
                                        dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = (
        detections['detection_classes']
        .astype(np.int64)
    )

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.7,
        agnostic_mode=False)

    classes = detections['detection_classes'] + 1
    boxes = detections['detection_boxes']
    scores = detections['detection_scores']
    detected = []

    for i in range(min(5, boxes.shape[0])):
        if scores is None or scores[i] > 0.7:
            if classes[i] in category_index.keys():
                detected.append(category_index[classes[i]]['name'])

    msg = pickle.dumps(detected)
    header = struct.pack('!I', len(msg))
    connect.Send(header, msg)

    ST = time.time()
    FPS = 1 / (ST-PT)
    PT = ST
    FPS = str(round(FPS))

    cv2.putText(image_np_with_detections, FPS, (7, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0),
                3, cv2.LINE_AA)
    cv2.imshow('object detection', image_np_with_detections)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        msg = b''
        header = struct.pack('!I', 0xffffffff)
        connect.Send(header, msg)
        cv2.destroyAllWindows()
        break

connect.Close()
