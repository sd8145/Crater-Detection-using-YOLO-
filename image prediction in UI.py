import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from ttkthemes import ThemedStyle  # Import ThemedStyle from ttkthemes

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("A YOLO-based Model for Moon Crater Detection")
        self.root.geometry("800x800")  # Set window size

        style = ThemedStyle(root)
        style.set_theme("equilux")  # Set the theme for the whole app

        title_label = tk.Label(root, text="A YOLO-based Model for Moon Crater Detection ", font=("Helvetica", 20))
        title_label.pack(pady=20)

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        self.detect_button = tk.Button(root, text="Detect Objects", command=self.detect_objects)
        self.detect_button.pack(pady=10)

        self.image_label = tk.Label(root, bg="LightYellow", padx=20, pady=20)  # Set panel color and padding
        self.image_label.pack(fill=tk.BOTH, expand=True)  # Expand label to fill window

        self.loaded_image = None
        self.loaded_cv_image = None

        # Load YOLO model
        self.model = YOLO("runs/detect/train2/weights/best.pt")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.loaded_image = Image.open(file_path)
            self.loaded_image = self.loaded_image.resize((600, 600))  # Resize image if needed
            self.loaded_cv_image = cv2.cvtColor(np.array(self.loaded_image), cv2.COLOR_RGB2BGR)
            self.display_cv_image(self.loaded_cv_image)

    def detect_objects(self):
        if self.loaded_cv_image is not None:
            if self.model is not None:
                results = self.model(source=self.loaded_cv_image)
                res_plotted = results[0].plot()

                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

                self.display_cv_image(res_plotted_rgb)
            else:
                self.display_message("Model not loaded!")

    def display_cv_image(self, cv_image):
        cv_image = Image.fromarray(cv_image)
        cv_image = cv_image.resize((600, 600))
        photo = ImageTk.PhotoImage(cv_image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def display_message(self, message):
        self.image_label.config(text=message)

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()