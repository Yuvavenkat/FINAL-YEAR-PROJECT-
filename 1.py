import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import numpy as np
import urllib.request
import pickle
from tensorflow.keras.models import load_model
import os

# Load pre-trained models
cnn_model = load_model('CNN.model')  # Replace with the correct path
knn_model = pickle.load(open('model1.sav', 'rb'))  # Load KNN model

base = "http://192.168.137.129/"  # NodeMCU ESP address
esp32_cam_url = "http://192.168.137.222/capture"  # ESP32-CAM URL

data_dir = "data"
class_names = os.listdir(data_dir)  # Update class labels accordingly

image_health_status = None  # Global variable for image classification result

def preview_esp32cam():
    while True:
        try:
            img_resp = urllib.request.urlopen(esp32_cam_url)
            img_arr = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            img = cv2.imdecode(img_arr, -1)

            if img is None:
                raise Exception("Failed to retrieve image.")

            cv2.imshow("ESP32-CAM Preview", img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                classify_image(img)
                break

        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch image: {e}")
            break

def classify_image(img):
    global image_health_status
    try:
        img = cv2.medianBlur(img, 1)
        img = cv2.resize(img, (50, 50))
        img = np.expand_dims(img, axis=0) / 255.0

        predictions = cnn_model.predict(img)
        class_index = np.argmax(predictions)
        class_label = class_names[class_index]

        result_label.config(text=f'Image Result: {class_label}')
        image_health_status = "healthy" if class_label.lower() == "healthy" else "unhealthy"
        
        choose_input_method()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to classify image: {e}")

def choose_input_method():
    try:
        choice = messagebox.askyesno("Input Method", "Do you want to fetch sensor data automatically? (Yes for auto, No for manual)")
        if choice:
            get_sensor_data()
        else:
            manual_input_mode()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to select input method: {e}")

def get_sensor_data():
    try:
        res = transfer("0")
        response = str(res)
        values = response.split('-')

        if len(values) == 5:
            temperature, humidity, pH, nitrogen, moisture = values
            messagebox.showinfo(
                "Sensor Data",
                f"Auto-detected values:\nTemperature: {temperature}°C\nHumidity: {humidity}%\npH: {pH}\nNitrogen: {nitrogen} ppm\nMoisture: {moisture}%"
            )
        else:
            messagebox.showwarning(
                "Manual Input Mode",
                "Unable to fetch sensor data automatically. Switching to manual input."
            )
            manual_input_mode()

        reports = [[float(temperature), float(humidity), float(pH), float(nitrogen), float(moisture)]]
        predicted_crop = knn_model.predict(reports)
        plant_health_percentage = predicted_crop[0]

        health_label.config(text=f'Plant Health: {plant_health_percentage}%')
        send_to_cloud(temperature, humidity, pH, nitrogen, moisture, plant_health_percentage)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch sensor data: {e}")

def manual_input_mode():
    try:
        temperature = validate_input("Temperature (°C)")
        humidity = validate_input("Humidity (%)")
        pH = validate_input("pH Level")
        nitrogen = validate_input("Nitrogen (ppm)")
        moisture = validate_input("Moisture (%)")

        reports = [[float(temperature), float(humidity), float(pH), float(nitrogen), float(moisture)]]
        predicted_crop = knn_model.predict(reports)
        plant_health_percentage = predicted_crop[0]

        health_label.config(text=f'Plant Health: {plant_health_percentage}%')
        send_to_cloud(temperature, humidity, pH, nitrogen, moisture, plant_health_percentage)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to manually input data: {e}")

def validate_input(field_name):
    while True:
        try:
            value = simpledialog.askfloat("Manual Input", f"Enter a valid float for {field_name}:")
            if value is not None:
                return value
            else:
                messagebox.showwarning("Invalid Input", "Input cannot be empty. Please enter a valid value.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid numeric value.")

def send_to_cloud(temperature, humidity, pH, nitrogen, moisture, plant_health_percentage):
    try:
        conn = urllib.request.urlopen(
            f"https://api.thingspeak.com/update?api_key=GNRSRJK8C2GWOPAC&field1={temperature}&field2={humidity}&field3={pH}&field4={nitrogen}&field5={moisture}&field6={plant_health_percentage}"
        )
        conn.read()
        conn.close()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to send data to the cloud: {e}")

def transfer(my_url):
    try:
        return urllib.request.urlopen(base + my_url).read().decode("utf-8")
    except Exception as e:
        return str(e)

root = tk.Tk()
root.title("Crop Name and Disease Prediction")

title_label = tk.Label(root, text="Crop Name and Disease Prediction", font=("Helvetica", 18))
title_label.pack(pady=20)

result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=20)

health_label = tk.Label(root, text="", font=("Helvetica", 16))
health_label.pack(pady=20)

classify_button = tk.Button(root, text="Start Prediction", command=preview_esp32cam)
classify_button.pack(pady=10)

quit_button = tk.Button(root, text="Exit", command=root.destroy)
quit_button.pack(pady=10)

root.mainloop()
