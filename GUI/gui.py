import tkinter as tk
from PIL import ImageTk, Image
import os
import communication as com
import threading
import struct
import pickle
import ai

RPI_HOST = '127.0.0.1'
RPI_PORT = 6789


class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.ai_detect = ai.AI()
        self.parent.protocol("WM_DELETE_WINDOW", self.on_close)
        # TODO sprawdzic czy da sie uzyc tylko connected
        self.net_running = True
        self.network_thrd = None

        self.connect_hndl = com.Communication()
        self.connected = False
        self.image_path = ""

        self.settings_frame = tk.Frame(parent)
        self.setting1 = tk.Label(self.settings_frame, text="Nazwa")
        self.setting2 = tk.Label(self.settings_frame, text="Jasność")
        self.entry1 = tk.Entry(self.settings_frame)
        self.entry2 = tk.Entry(self.settings_frame)

        self.image_frame = tk.Frame(parent)
        self.image = tk.Label()
        
        self.buttons_frame = tk.Frame(parent)
        self.connect_btn = tk.Button(self.buttons_frame, text="Connect", command=self.connect)
        self.disconnect_btn = tk.Button(self.buttons_frame, text="Disconnect", command=self.disconnect)
        self.picture_shot_btn = tk.Button(self.buttons_frame, text="Take a shot", command=self.download_image)
        self.info_panel = tk.Label(self.buttons_frame, text="Application succesfully loaded")


        self.settings_frame.pack(side="top", fill="y")
        self.setting1.grid(row=0, column=0)
        self.setting2.grid(row=1, column=0)
        self.entry1.grid(row=0, column=1)
        self.entry2.grid(row=1, column=1)

        self.image_frame.pack(side="top", fill="x")
        self.image.pack(side="top", fill="x")

        self.buttons_frame.pack(side="bottom", fill="x")
        self.connect_btn.grid(row=0, column=0)
        self.disconnect_btn.grid(row=0, column=1)
        self.picture_shot_btn.grid(row=0, column=2)
        self.info_panel.grid(row=1)

    def connect(self):
        if not self.connected:
            try:
                self.connect_hndl.Connect(RPI_HOST, RPI_PORT)
                self.info_panel.config(text=f"Connected to {RPI_HOST}:{RPI_PORT}")
                self.connected = True
                self.network_thrd = threading.Thread(target=self.recieve_networking_thrd)
                self.network_thrd.start()
            except Exception as e:
                self.info_panel.config(text=f"Can't connect to {RPI_HOST}:{RPI_PORT}")
                print(e)
    
    def disconnect(self):
        if self.connected:
            self.connect_hndl.DropConnection()
            self.info_panel.config(text="Disconnected")
            self.connected = False
            self.net_running = False

    def recieve_networking_thrd(self):
        data = b''
        headSize = struct.calcsize('II')

        while self.net_running:
            try:
                data = self.connect_hndl.RecvFirstChunk(headSize, data)
                msg, data = self.connect_hndl.RecvSecondChunk(headSize, data)

                protocol_fields = self.connect_hndl.HandleHeaderParams()
                if protocol_fields['mode'] == com.MODE_CAMSEND:
                    self.update_image(msg)
                elif protocol_fields['mode'] == com.MODE_EXIT:
                    break
            except Exception as e:
                self.connected = False
                self.info_panel.config(text="The connection has been broken")
                self.connect_hndl.DropConnection()
                break

    def download_image(self):
        msg = b''
        header = struct.pack("!II", com.MODE_CAMSHOT, len(msg))
        self.connect_hndl.Send(header, msg)
        print("aa")

    def update_image(self, pickled):
        unpickled = pickle.loads(pickled)
        detected_img = self.ai_detect.detect(unpickled)
        img = ImageTk.PhotoImage(Image.fromarray(detected_img))
        self.image.config(image=img)
        self.image.panel = img

    def on_close(self):
        self.net_running = False

        self.connect_hndl.Close()
        self.parent.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x800")
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()