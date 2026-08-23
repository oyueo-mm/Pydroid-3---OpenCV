import tkinter as tk
import cv2
import PySimpleGUI as sg

USE_CAMERA = 0 

layout = [[sg.Image(filename='', key='-IMAGE-')]]
window = sg.Window('Pydroid OpenCV Camera', layout, location=(0, 0), grab_anywhere=True)

cap = cv2.VideoCapture(USE_CAMERA)

while True:
    event, values = window.read(timeout=20)
    
    if event == sg.WINDOW_CLOSED:
        break

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    window['-IMAGE-'].update(data=imgbytes)

cap.release()
window.close()