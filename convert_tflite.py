import os
import config as cfg

FREEZE_SCRIPT = os.path.join(
    cfg.paths['APIMODEL_PATH'],
    'research',
    'object_detection',
    'exporter_main_v2.py ')

print("Freezing graph...")
os.system(
    f"python {FREEZE_SCRIPT} "
    f"--input_type=image_tensor "
    f"--pipeline_config_path={cfg.files['PIPELINE_CONFIG']} "
    f"--trained_checkpoint_dir={cfg.paths['CHECKPOINT_PATH']} "
    f"--output_directory={cfg.paths['OUTPUT_PATH']}")


TFLITE_SCRIPT = os.path.join(
    cfg.paths['APIMODEL_PATH'],
    'research',
    'object_detection',
    'export_tflite_graph_tf2.py')

print("Exporting graph...")
os.system(
    f"python {TFLITE_SCRIPT} "
    f"--pipeline_config_path={cfg.files['PIPELINE_CONFIG']} "
    f"--trained_checkpoint_dir={cfg.paths['CHECKPOINT_PATH']} "
    f"--output_directory={cfg.paths['TFLITE_PATH']}")

FROZEN_TFLITE_PATH = os.path.join(cfg.paths['TFLITE_PATH'], 'saved_model')
TFLITE_MODEL = os.path.join(cfg.paths['TFLITE_PATH'], 'saved_model',
                            'detect.tflite')

print("Converting model...")
os.system(
    f"tflite_convert "
    f"--saved_model_dir={FROZEN_TFLITE_PATH} "
    f"--output_file={TFLITE_MODEL} "
    f"--input_shapes=1,300,300,3 "
    f"--input_arrays=normalized_input_image_tensor "
    f"--output_arrays='TFLite_Detection_PostProcess',"
    f"'TFLite_Detection_PostProcess:1',"
    f"'TFLite_Detection_PostProcess:2',"
    f"'TFLite_Detection_PostProcess:3' "
    f"--inference_type=FLOAT "
    f"--allow_custom_ops")
