import tkinter as tk
import cv2 as cv
import PySimpleGUI as sg

USE_CAMERA = 0

root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

layout = [[sg.Image(filename='', key='-IMAGE-')]]

window = sg.Window('Canny', layout, location=(0, 0), size=(screen_width, screen_height), grab_anywhere=True)

cap = cv.VideoCapture(USE_CAMERA)

while True:
    event, values = window.read(timeout=20)
    if event == sg.WINDOW_CLOSED:
        break 
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.Canny(frame, 100, 300)
    frame = cv.resize(frame, (screen_width, screen_height))
   
    imgbytes = cv.imencode('.png', frame)[1].tobytes()
    window['-IMAGE-'].update(data=imgbytes)

cap.release()
window.close()