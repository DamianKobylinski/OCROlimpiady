import os
import config as cfg

for path in cfg.paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            os.system(f"mkdir -p {path}")
        if os.name == 'nt':
            os.system(f"mkdir {path}")

if os.name == 'nt':
    os.system("pip install wget")
    import wget

if not os.path.exists(os.path.join(cfg.paths['APIMODEL_PATH'],
                      'research', 'object_detection')):
    os.system(f"git clone https://github.com/tensorflow/models "
              f"{cfg.paths['APIMODEL_PATH']}")

if os.name == 'posix':
    os.system("apt-get install protobuf-compiler")
    os.system(
        "cd Tensorflow/models/research "
        "&& protoc object_detection/protos/*.proto --python_out=. "
        "&& cp object_detection/packages/tf2/setup.py . "
        "&& python -m pip install .")

if os.name == 'nt':
    url = (
        "https://github.com/protocolbuffers/protobuf/releases/"
        "download/v3.15.6/protoc-3.15.6-win64.zip"
    )
    wget.download(url)
    os.system(f"move protoc-3.15.6-win64.zip {cfg.paths['PROTOC_PATH']}")
    os.system(
        f"cd {cfg.paths['PROTOC_PATH']} "
        f"&& tar -xf protoc-3.15.6-win64.zip")
    os.system(
        "cd Tensorflow/models/research "
        "&& ..\\..\\protoc\\bin\\protoc object_detection\\protos\\*.proto "
        "--python_out=. "
        "&& copy object_detection\\packages\\tf2\\setup.py setup.py "
        "&& python setup.py build "
        "&& python setup.py install")
    os.system(
        "cd Tensorflow/models/research/slim "
        "&& pip install -e . ")

os.system(f"wget {cfg.PRETRAINED_MODEL_URL}")
if os.name == 'posix':
    os.system(f"wget {cfg.PRETRAINED_MODEL_URL}")
    os.system(
        f"mv {cfg.PRETRAINED_MODEL_NAME+'.tar.gz'} "
        f"{cfg.paths['PRETRAINED_MODEL_PATH']}")
    os.system(
        f"cd {cfg.paths['PRETRAINED_MODEL_PATH']} "
        f"&& tar -zxvf {cfg.PRETRAINED_MODEL_NAME+'.tar.gz'}")
if os.name == 'nt':
    wget.download(cfg.PRETRAINED_MODEL_URL)
    os.system(
        f"move {cfg.PRETRAINED_MODEL_NAME+'.tar.gz'} "
        f"{cfg.paths['PRETRAINED_MODEL_PATH']}")
    os.system(
        f"cd {cfg.paths['PRETRAINED_MODEL_PATH']} "
        f"&& tar -zxvf {cfg.PRETRAINED_MODEL_NAME+'.tar.gz'}")


labels = [{'name': 'thumbsup', 'id': 1}, {'name': 'thumbsdown', 'id': 2}]

# with open(cfg.files['LABELMAP'], 'w') as f:
#     for label in labels:
#         f.write('item { \n')
#         f.write('\tname:\'{}\'\n'.format(label['name']))
#         f.write('\tid:{}\n'.format(label['id']))
#         f.write('}\n')

"""OPTIONAL IF RUNNING ON COLAB
ARCHIVE_FILES = os.path.join(cfg.paths['IMAGE_PATH'], 'archive.tar.gz')
if os.path.exists(ARCHIVE_FILES):
    os.system(f"tar -zxvf {ARCHIVE_FILES}")
"""

if os.name == 'posix':
    param = os.path.join(cfg.paths['PRETRAINED_MODEL_PATH'],
                         cfg.PRETRAINED_MODEL_NAME,
                         'pipeline.config')
    os.system(
        f"cp {param} "
        f"{os.path.join(cfg.paths['CHECKPOINT_PATH'])}")
if os.name == 'nt':
    param = os.path.join(cfg.paths['PRETRAINED_MODEL_PATH'],
                         cfg.PRETRAINED_MODEL_NAME,
                         'pipeline.config')
    os.system(
        f"copy {param} "
        f"{os.path.join(cfg.paths['CHECKPOINT_PATH'])}")

# os.system("pip install --update tensorflow")
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

config = config_util.get_configs_from_pipeline_file(
    cfg.files['PIPELINE_CONFIG'])

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(cfg.files['PIPELINE_CONFIG'], "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(
    cfg.paths['PRETRAINED_MODEL_PATH'], cfg.PRETRAINED_MODEL_NAME,
    'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path = cfg.files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
    os.path.join(cfg.paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = cfg.files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
    os.path.join(cfg.paths['ANNOTATION_PATH'], 'test.record')]

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(cfg.files['PIPELINE_CONFIG'], "wb") as f:
    f.write(config_text)

print(config)