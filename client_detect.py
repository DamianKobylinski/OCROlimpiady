import struct
import threading
import pickle
import communication as com
import cv2

HOST, PORT = ('localhost', 6790)


def RecieveThread(connect):
    data = b''
    headSize = struct.calcsize('I')

    while True:
        data = connect.RecvFirstChunk(headSize, data)

        # Protocol handling
        shutdown = connect.HandleHeaderParams()

        if shutdown:
            break
        # ---

        msg, data = connect.RecvSecondChunk(headSize, data)
        detections = pickle.loads(msg)
        print(detections)


if __name__ == '__main__':
    connect = com.Communication()

    while True:
        try:
            connect.Connect(HOST, PORT)
            break
        except Exception as e:
            print(f'{"[SERVER]":10}Reconnecting...')

    thread = threading.Thread(target=RecieveThread, args=(connect,))
    thread.start()

    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        msg = pickle.dumps(frame)
        header = struct.pack('!I', len(msg))
        connect.Send(header, msg)

    thread.join()
    connect.Close()
