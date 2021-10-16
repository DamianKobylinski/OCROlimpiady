import keras_ocr
import matplotlib.pyplot as plt
import matplotlib.image as mtl
import multiprocessing
from gtts import gTTS
from playsound import playsound
import time
import os


def speak():
    while True:
        with open('text.txt', 'r') as file:
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


proc = multiprocessing.Process(target=speak)
proc.start()
pipeline = keras_ocr.pipeline.Pipeline()
images = [keras_ocr.tools.read(img) for img in ['./Networking/test2.png']]

len(images)

#plt.figure(figsize = (10,20))
#plt.imshow(images[0])

#plt.figure(figsize=(10,20))
#plt.imshow(images[1])

prediction_groups = pipeline.recognize(images)
# fig, axs = plt.subplots(nrows=len(images),figsize=(100,200))
# for ax, image, predictions in zip(axs,images,prediction_groups):
#     keras_ocr.tools.drawAnnotations(image=image, predictions=predictions,ax=ax)

x_max = 0
temp_str = ""
txt_file = open("text.txt", "w")
for i in prediction_groups[0]:

    x_max_local = i[1][:, 0].max()
    if x_max_local > x_max:
        x_max = x_max_local
        temp_str = temp_str + " " + i[0].ljust(15)
    else:
        x_max = 0
        temp_str = temp_str + "\n"
        txt_file.write(temp_str)
        print(temp_str)
        temp_str = ""
txt_file.close()
print(prediction_groups[0])
# print(prediction_groups[0][0])
# print(prediction_groups[0][0][0])
for i in range(0,len(prediction_groups[0])):
    textPre = prediction_groups[0][i][0]
    print(textPre)
# plt.show()
proc.terminate()