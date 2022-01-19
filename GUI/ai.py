import os
import tensorflow as tf
import cv2
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import config as cfg

class AI:
    def __init__(self):
        configs = (config_util.get_configs_from_pipeline_file(cfg.files['PIPELINE_CONFIG']))
        self.detection_model = model_builder.build(model_config=configs['model'], is_training=False)
        ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        ckpt.restore(os.path.join(cfg.paths['CHECKPOINT_PATH'],
                'ckpt-6')).expect_partial()
        self.category_index = label_map_util.create_category_index_from_labelmap(
                cfg.files['LABELMAP'])
    
    def detect(self, image):
        image = np.array(image)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
        detections = self.detect_image(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                     for key, value in detections.items()}
        detections['num_detections'] = num_detections

        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                   image_np_with_detections,
                   detections['detection_boxes'],
                   detections['detection_classes']+label_id_offset,
                   detections['detection_scores'],
                   self.category_index,
                   use_normalized_coordinates=True,
                   max_boxes_to_draw=5,
                   min_score_thresh=.3,
                   agnostic_mode=False)

        return image_np_with_detections

    @tf.function
    def detect_image(self, image):
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections