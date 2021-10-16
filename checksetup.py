import os
import config as cfg

VERIFICATION_SCRIPT = os.path.join(
    cfg.paths['APIMODEL_PATH'],
    'research',
    'object_detection',
    'builders',
    'model_builder_tf2_test.py')

os.system(f"python {VERIFICATION_SCRIPT}")
