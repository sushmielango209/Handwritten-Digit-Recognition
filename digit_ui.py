import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

model = load_model('mnist_model.h5')

class DigitRecognizer:
    def __init__(self, root):
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        self.image = Image.new("L", (28, 28), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        tk.Button(root, text="Predict", command=self.predict).pack()

    def paint(self, event):
        x, y = event.x // 10, event.y // 10
        self.draw.rectangle([x, y, x+1, y+1], fill=255)
        self.canvas.create_rectangle(event.x, event.y, event.x+10, event.y+10, fill='black')

    def predict(self):
        img_array = np.array(self.image).reshape(1, 28, 28, 1) / 255.0
        prediction = model.predict(img_array)
        print("Predicted Digit:", np.argmax(prediction))

root = tk.Tk()
app = DigitRecognizer(root)
root.mainloop()

