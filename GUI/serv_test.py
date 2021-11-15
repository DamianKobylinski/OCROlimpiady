import communication as com
import pickle
import threading
from PIL import Image
import struct
import cv2

RPI_HOST = '127.0.0.1'
RPI_PORT = 6789

def send_img(connect):
    cap = cv2.VideoCapture(0)
    _, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    msg = pickle.dumps(image)
    header = struct.pack("!II", com.MODE_CAMSEND, len(msg))
    connect.Send(header, msg)

def recv_thrd(connect):
    data = b''
    headSize = struct.calcsize('II')

    while True:
        try:
            data = connect.RecvFirstChunk(headSize, data)
            msg, data = connect.RecvSecondChunk(headSize, data)

            # Protocol handling
            protocol_fields = connect.HandleHeaderParams()

            if protocol_fields['mode'] == com.MODE_CAMSHOT:
                send_img(connect)
                print("Sending")
            elif protocol_fields['mode'] == com.MODE_EXIT:
                break
            # ---
        except Exception as e:
            print(e)
            break


conn = com.Communication()
conn.Host(RPI_HOST, RPI_PORT)
thread = threading.Thread(target=recv_thrd, args=(conn,))
thread.start()
thread.join()