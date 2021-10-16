import os
#import tensorflow as tf
import cv2
import numpy as np
import tkinter
from matplotlib import pyplot as plt
#from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as viz_utils
#from object_detection.builders import model_builder
#from object_detection.utils import config_util
import config as cfg
import multiprocessing
from gtts import gTTS
from playsound import playsound
import time
import argparse
from PIL import Image

MAX_BOXES = 5
MIN_SCORE_THRESH = 0.6


def speak():
    while True:
        with open('Tensorflow-lite/detections.txt', 'r') as file:
            file.seek(0, os.SEEK_END)
            if file.tell():
                file.seek(0)
                speak = ""
                for line in file:
                    speak += line
                tts = gTTS(speak, lang="pl")
                tts.save("speech.mp3")
                playsound('./speech.mp3')
                time.sleep(2)


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


if __name__ == '__main__':
    configs = (
        config_util
        .get_configs_from_pipeline_file(cfg.files['PIPELINE_CONFIG'])
    )

   detection_model = model_builder.build(
       model_config=configs['model'],
       is_training=False)

   ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
   ckpt.restore(os.path.join(cfg.paths['CHECKPOINT_PATH'],
               'ckpt-4')).expect_partial()

   category_index = label_map_util.create_category_index_from_labelmap(
       cfg.files['LABELMAP'])

   IMAGE_PATH = os.path.join('Networking','Image.jpeg')

   while True:
       img = cv2.imread(IMAGE_PATH)
       image_np = np.array(img)

       input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
       detections = detect_fn(input_tensor)

       num_detections = int(detections.pop('num_detections'))
       detections = {key: value[0, :num_detections].numpy()
                     for key, value in detections.items()}
       detections['num_detections'] = num_detections

       # detection_classes should be ints.
       detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

       label_id_offset = 1
       image_np_with_detections = image_np.copy()

       viz_utils.visualize_boxes_and_labels_on_image_array(
                   image_np_with_detections,
                   detections['detection_boxes'],
                   detections['detection_classes']+label_id_offset,
                   detections['detection_scores'],
                   category_index,
                   use_normalized_coordinates=True,
                   max_boxes_to_draw=5,
                   min_score_thresh=.8,
                   agnostic_mode=False)

       plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
       plt.savefig('demo.pdf')
#### Video Detection
#    proc = multiprocessing.Process(target=speak)
#    proc.start()
#    cap = cv2.VideoCapture(0)
#    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#    while cap.isOpened():
#        ret, frame = cap.read()
#        image_np = np.array(frame)
#
#        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0),
#                                            dtype=tf.float32)
#        detections = detect_fn(input_tensor)
#
#        num_detections = int(detections.pop('num_detections'))
#        detections = {key: value[0, :num_detections].numpy()
#                    for key, value in detections.items()}
#        detections['num_detections'] = num_detections
#
#        # detection_classes should be ints.
#        detections['detection_classes'] = (
#            detections['detection_classes']
#            .astype(np.int64)
#        )
#
#        label_id_offset = 1
#        image_np_with_detections = image_np.copy()
#
#        classes = detections['detection_classes'] + 1
#        boxes = detections['detection_boxes']
#        scores = detections['detection_scores']
#        detected = []
#
#        with open('Tensorflow-lite/detections.txt', 'w') as file:
#                    for i in range(min(MAX_BOXES, boxes.shape[0])):
#                        if scores is None or scores[i] > MIN_SCORE_THRESH:
#                            if classes[i] in category_index.keys():
#                                file.write(category_index[classes[i]]['name']+" ")
#
#        viz_utils.visualize_boxes_and_labels_on_image_array(
#            image_np_with_detections,
#            detections['detection_boxes'],
#            detections['detection_classes'] + label_id_offset,
#            detections['detection_scores'],
#            category_index,
#            use_normalized_coordinates=True,
#            max_boxes_to_draw=MAX_BOXES,
#            min_score_thresh=MIN_SCORE_THRESH,
#            agnostic_mode=False)
#
#        cv2.imshow('object detection', cv2.resize(image_np_with_detections,
#                (800, 600)))
#
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            cap.release()
#            cv2.destroyAllWindows()
#            break
#### Video Detection
    proc.terminate()
