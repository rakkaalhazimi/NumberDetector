'''
=================================================================================
Python Number Detector with Convolutional Neural Network
=================================================================================

This app is powered by tkinter GUI for faster deployment and data science modules.
Python Number Detector provide simple drawable widget with predicting utilities using
deep learning model. Its predicting accuracy is not perfect, so you may get a wrong
answer or even more.

How to use this app:
1. Run mainApp.py and wait until the interface pop out.
2. There are 3 buttons save, detect and clear. You are obliged to use it, unless you
   start drawing something on the canvas.
3. After you done with drawing the digit, you can press detect button to get your
   answer showed by incoming showinfo messagebox.
4. You can use save button to save the image, it was programmed to store 28 x 28 pixel
   size image according to default mnist datasets pixels.
5. Finally use clear button to reset your drawing.

What are the files in this projects used for?
1. partial_func => It store my prototype code, you can ignore it
2. saved_img => Some images that have been saved using save button in main app.
3. saved_model => Store trained model bundled with pickle module.

any suggestion? email me garciagyax@gmail.com
'''

import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageDraw
import numpy as np
import pickle, os, re

CV_HEIGHT = 280
CV_WIDTH = 280

IMG_DIRNAME = "saved_img"
INIT_NAME = "drawing1.png"

# Load pre trained convolutional neural network model
# You can either use base_model.pkl or add_conv_model.pkl
with open('saved_model/add_conv_model.pkl', 'rb') as file:
    convModel = pickle.load(file)


class MyApp(tk.Tk):
    """Widget meta container"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Number Predictor")
        self.resizable(width=False, height=False)
        DrawCanvas(self, background="green").pack()

class DrawCanvas(tk.Frame):
    """Main feature of this App, Drawing Board"""

    def __init__(self, master, **options):
        super().__init__(master, **options)
        self.makewidget()

        # Create PIL ImageDraw Instance
        self.createPil()

    def clear(self):
        '''Reset everything'''
        for button in [self.detBtn, self.saveBtn, self.clrBtn]:
            button['state'] = 'disabled'

        self.cv.delete(tk.ALL)
        self.createPil()

    def detect(self):
        self.saveBtn['state'] = 'normal'
        number = self.resizeAndReshape()
        pred = convModel.predict_classes(number)

        messagebox.showinfo("Prediction", f"This is {pred[0]}")

    def paintCanvas(self, event):
        '''Paint canvas in parallel with PIL'''
        for button in [self.detBtn, self.clrBtn]: # enable buttons
            button['state'] = 'normal'

        x1, y1 = event.x, event.y
        x2, y2 = event.x + 1, event.y + 1
        event.widget.create_oval(x1, y1, x2, y2, fill="black", width=20)
        self.draw.ellipse([x1 - 10, y1 - 10, x2 + 10, y2 + 10], fill="black") # ellipse use bounding box

    def saveImg(self):
        if not os.path.exists(IMG_DIRNAME):
            os.mkdir(IMG_DIRNAME)

        # self.filecheck = FileChecker()
        fileImg = filedialog.asksaveasfilename(initialdir=IMG_DIRNAME, initialfile=".png", defaultextension=".png",
                                               filetypes=[('PNG Files', '*.png'), ('JPG Files', '*.jpg')])
        self.resized.save(fileImg)

    def createPil(self):
        '''New blank image'''
        self.img = Image.new("RGB", (CV_WIDTH, CV_HEIGHT), "white")
        self.draw = ImageDraw.Draw(self.img)

    def makewidget(self):
        # Canvas Widget
        cv = tk.Canvas(self, width=CV_WIDTH, height=CV_HEIGHT)
        cv.bind("<B1-Motion>", self.paintCanvas)
        cv.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        self.cv = cv

        # Buttons Widget
        self.detBtn = tk.Button(self, text="Detect", command=self.detect, state="disabled") # disabled button in startup
        self.detBtn.grid(row=1, column=0, columnspan=2, sticky=tk.EW)

        self.saveBtn = tk.Button(self, text="Save", command=self.saveImg, state="disabled")
        self.saveBtn.grid(row=2, column=0, sticky=tk.EW)

        self.clrBtn = tk.Button(self, text="Clear", activebackground="red", command=self.clear, state="disabled")
        self.clrBtn.grid(row=2, column=1, sticky=tk.EW)

    def resizeAndReshape(self):
        'format the image before feed it into model'
        self.resized = self.img.resize((28, 28), Image.ANTIALIAS).convert("L")

        self.reshaped = np.asarray(self.resized).reshape(1, 28, 28, 1)
        self.reshaped = self.reshaped.astype('float32') / 255
        return self.reshaped


if __name__ == '__main__':
    app = MyApp()
    app.mainloop()




