import tkinter as tk
from tkinter import filedialog
from tkinter import *

from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2 as cv

# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, 'r') as file:
        loaded_model_json = file.read()
        model= model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background="#CDCDCD")

label1 = Label(top, background= '#CDCDCD', font= ('arial', 15, 'bold'))
sign_image = Label(top)

facec = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel('/Users/pradeepmeena/Downloads/Machine learning/ML/NULL CLASS/Emotion Detection/model_a1.json' , '/Users/pradeepmeena/Downloads/Machine learning/ML/NULL CLASS/Emotion Detection/model.weights.weights.h5')

EMOTIONS_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def Detect(file_path):
    global Label_packed

    image = cv.imread(file_path)
    grey_image =  cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(grey_image, 1.3, 5)
    try:
        for(x, y, w, h) in faces:
            fc = grey_image[y:y+h, x:x+w]
            ROI = cv.resize(fc, (48, 48))
            predicted = EMOTIONS_LIST[np.argmax(model.predict(ROI[np.newaxis, :, :, np.newaxis]))]
            print("Predicted Emotion is: "  +  predicted)
            label1.configure(foreground='#011638', text = predicted)
    except:
        label1.configure(foreground='#011638', text = "Unable to Detect")

def show_detect_button(file_path):
    detect_b = Button(top, text = "Detect Emotion" ,command = lambda: Detect(file_path))
    detect_b.configure(background='#364125', foreground='black', font =('arial', 15, 'bold'))
    detect_b.place(relx = 0.79, rely = 0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.3), (top.winfo_height()/2.3)))
        img = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image= img)
        sign_image.image = img
        label1.configure(text = '')
        show_detect_button(file_path)
    except:
        pass

upload = Button(top,text = 'Upload Image', command = upload_image, padx = 10, pady = 10)
upload.configure(background='#364125', foreground='white', font =('arial', 20, 'bold'))
upload.pack(side= 'bottom', pady = 50)
sign_image.pack(side = 'bottom', expand ='True')
label1.pack(side='bottom', expand = 'True')
heading = Label(top, text = "Emotion Detector", pady = 20, font = ('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364125')
heading.pack()
top.mainloop()

