import cv2
import uuid
import os
import time
import config as cfg

labels = ['thumbsup', 'thumbsdown']

number_imgs = 3

IMAGES_PATH = os.path.join('Tensorflow', 'workspace',
                           'images', 'collectedimages')

if not os.path.exists(IMAGES_PATH):
    if os.name == 'posix':
        os.system(f"mkdir -p {IMAGES_PATH}")
    if os.name == 'nt':
        os.system(f"mkdir {IMAGES_PATH}")
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.system(f"mkdir {path}")

cap = cv2.VideoCapture(0)
for label in labels:
    print('Collecting images for {}'.format(label))
    i = 0
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            print('Collecting image {}'.format(imgnum))
            imgname = os.path.join(IMAGES_PATH, label, label+'.'+'{}.jpg'
                                   .format(str(uuid.uuid1())))
            cv2.imwrite(imgname, frame)
            i += 1
            if i == number_imgs:
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

LABELIMG_PATH = os.path.join('Tensorflow', 'labelimg')

if not os.path.exists(LABELIMG_PATH):
    os.system(f"mkdir {LABELIMG_PATH}")
    os.system(f"git clone https://github.com/tzutalin/labelImg "
              f"{LABELIMG_PATH}")

if os.name == 'posix':
    os.system("make qt5py3")
if os.name == 'nt':
    os.system(f"cd {LABELIMG_PATH} && pyrcc5 -o "
              "libs/resources.py resources.qrc")

os.system(f"cd {LABELIMG_PATH} && python labelImg.py")

""" Compress for Google Colab.
TRAIN_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'train')
TEST_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'test')
ARCHIVE_PATH = os.path.join('Tensorflow', 'workspace', 'images',
                            'archive.tar.gz')

os.system(f"tar -czf {ARCHIVE_PATH} {TRAIN_PATH} {TEST_PATH}")
"""
