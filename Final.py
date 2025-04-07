import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os

# Load your pre-trained model ('CNN.model')
model = load_model('CNN.model')  # Replace with the correct path to your CNN.model
data_dir = "data"
class_names = os.listdir(data_dir)

# Function to classify an image
def classify_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and preprocess the selected image
        img = cv2.imread(file_path)
        img = cv2.medianBlur(img, 1)
        img = cv2.resize(img, (50, 50))  # Adjust the size if needed
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize the image
        img = np.asarray(img)  # Convert to numpy array
        
        # Make predictions
        predictions = model.predict(img)
        class_index = np.argmax(predictions)
        class_label = class_names[class_index]
        
        # Display the result
        result_label.config(text=f'Result: {class_label}')

# Create a tkinter window
root = tk.Tk()
root.title("Classifier")

# Create a label for the title
title_label = tk.Label(root, text="Classification", font=("Helvetica", 20))
title_label.pack(pady=20)

# Create a label to display the result
result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=20)

# Create a button to select an image
classify_button = tk.Button(root, text="Select Image", command=classify_image)
classify_button.pack()

# Create a quit button
quit_button = tk.Button(root, text="Quit", command=root.destroy)
quit_button.pack()

# Start the tkinter main loop
root.mainloop()

