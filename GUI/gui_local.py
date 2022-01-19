import tkinter as tk
from PIL import ImageTk, Image
import os
import threading
import ai
import cv2

class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.ai_detect = ai.AI()
        self.parent.protocol("WM_DELETE_WINDOW", self.on_close)
        self.capture_video = True
        self.video_feed_hndl = None

        self.settings_frame = tk.Frame(parent)
        self.setting1 = tk.Label(self.settings_frame, text="Nazwa")
        self.setting2 = tk.Label(self.settings_frame, text="Jasność")
        self.entry1 = tk.Entry(self.settings_frame)
        self.entry2 = tk.Entry(self.settings_frame)

        self.image_frame = tk.Frame(parent)
        self.image = tk.Label()
        
        self.buttons_frame = tk.Frame(parent)
        self.connect_btn = tk.Button(self.buttons_frame, text="Run video feed", command=self.start_webcam)
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
        self.info_panel.grid(row=1)

    def start_webcam(self):
        self.video_feed_hndl = threading.Thread(target=self.video_feed_loop)
        self.video_feed_hndl.start()

    def video_feed_loop(self):
        cap = cv2.VideoCapture(0)
        while self.capture_video:
            _, image = cap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detected_img = self.ai_detect.detect(image)
            img = ImageTk.PhotoImage(Image.fromarray(detected_img))
            self.image.config(image=img)
            self.image.panel = img

    def on_close(self):
        self.capture_video = False
        self.parent.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x800")
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()