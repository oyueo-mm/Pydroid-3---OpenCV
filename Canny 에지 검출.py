import tkinter as tk
import cv2
from PIL import Image, ImageTk

USE_CAMERA = 0

root = tk.Tk()
root.title('Canny')

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

root.geometry(f'{screen_width}x{screen_height}')

label = tk.Label(root)
label.pack()

cap = cv2.VideoCapture(USE_CAMERA)

def update():
	ret, frame = cap.read()
	if not ret:
		root.destroy()
		return	
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.Canny(frame, 100, 300)
	frame = cv2.resize(frame, (screen_width, screen_height))
	frame = Image.fromarray(frame)
	frame = ImageTk.PhotoImage(frame)
	
	label.config(image=frame)
	label.image = frame
	
	root.after(10, update)

update()
root.mainloop()
cap.release()