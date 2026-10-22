import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk

CAMERA = 0

modes = [
    "NEON VISION",
    "PSEUDO THERMAL",
    "MOTION DETECTOR",
    "RED OBJECT TRACKER",
    "CARTOON CAMERA",
    "INVISIBLE CLOAK"
]

mode = 0
previous = None
background = None
last_frame = None
running = True

root = tk.Tk()
root.title("Cyber Vision Lab")
root.configure(bg="#080808")

sw = root.winfo_screenwidth()
sh = root.winfo_screenheight()
root.geometry(f"{sw}x{sh}")

title = tk.Label(
    root,
    text=modes[mode],
    bg="#080808",
    fg="#00ffcc",
    font=("Arial", 17, "bold")
)
title.pack(pady=8)

label = tk.Label(root, bg="black")
label.pack(expand=True, fill="both")

status = tk.Label(
    root,
    text="Choose an effect",
    bg="#080808",
    fg="white",
    font=("Arial", 11)
)
status.pack(pady=5)

panel = tk.Frame(root, bg="#080808")
panel.pack(fill="x", pady=12)

cap = cv2.VideoCapture(CAMERA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def change_mode(step):
    global mode, previous
    mode = (mode + step) % len(modes)
    previous = None
    title.config(text=modes[mode])
    status.config(text=f"MODE {mode + 1} / {len(modes)}")

def capture_bg():
    global background
    if last_frame is not None:
        background = last_frame.copy()
        status.config(text="Background captured!")

def switch_camera():
    global cap, CAMERA, previous, background

    next_camera = 1 - CAMERA
    new_cap = cv2.VideoCapture(next_camera)

    if not new_cap.isOpened():
        new_cap.release()
        status.config(text="Camera not available")
        return

    cap.release()
    cap = new_cap
    CAMERA = next_camera
    previous = None
    background = None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def close():
    global running
    running = False
    cap.release()
    root.destroy()

def process(frame):
    global previous

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (7, 7), 0)
    result = frame.copy()

    if mode == 0:
        edges = cv2.Canny(gray, 70, 150)
        glow = cv2.GaussianBlur(edges, (0, 0), 4)
        zeros = np.zeros_like(gray)

        neon = cv2.merge((edges, glow, zeros))
        result = cv2.addWeighted(
            frame, 0.15, neon, 1.0, 0
        )

    elif mode == 1:
        result = cv2.applyColorMap(
            gray, cv2.COLORMAP_JET
        )

    elif mode == 2:
        count = 0

        if previous is not None:
            diff = cv2.absdiff(previous, smooth)
            _, mask = cv2.threshold(
                diff, 25, 255, cv2.THRESH_BINARY
            )
            mask = cv2.dilate(mask, None, iterations=2)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                if cv2.contourArea(contour) < 400:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(
                    result, (x, y), (x + w, y + h),
                    (0, 0, 255), 2
                )
                count += 1

        cv2.putText(
            result, f"MOTION: {count}", (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

    elif mode == 3:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower1 = cv2.inRange(
            hsv, (0, 100, 70), (10, 255, 255)
        )
        lower2 = cv2.inRange(
            hsv, (170, 100, 70), (179, 255, 255)
        )
        mask = cv2.bitwise_or(lower1, lower2)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            target = max(contours, key=cv2.contourArea)

            if cv2.contourArea(target) > 300:
                (x, y), radius = cv2.minEnclosingCircle(
                    target
                )

                center = (int(x), int(y))
                cv2.circle(
                    result, center, int(radius),
                    (0, 255, 0), 3
                )
                cv2.drawMarker(
                    result, center, (0, 255, 255),
                    cv2.MARKER_CROSS, 25, 2
                )
                cv2.putText(
                    result, "TARGET LOCKED", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2
                )

    elif mode == 4:
        blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 5
        )
        colors = cv2.bilateralFilter(
            frame, 7, 75, 75
        )
        result = cv2.bitwise_and(
            colors, colors, mask=edges
        )

    elif mode == 5:
        if background is not None:
            hsv = cv2.cvtColor(
                frame, cv2.COLOR_BGR2HSV
            )

            m1 = cv2.inRange(
                hsv, (0, 100, 70), (10, 255, 255)
            )
            m2 = cv2.inRange(
                hsv, (170, 100, 70), (179, 255, 255)
            )
            mask = cv2.bitwise_or(m1, m2)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, kernel
            )
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

            alpha = mask.astype(np.float32) / 255.0
            alpha = alpha[:, :, None]

            result = (
                frame.astype(np.float32) * (1 - alpha)
                + background.astype(np.float32) * alpha
            ).astype(np.uint8)

        else:
            cv2.putText(
                result, "CAPTURE BG FIRST", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2
            )

    previous = smooth.copy()
    return result

def update():
    global last_frame

    if not running:
        return

    ret, frame = cap.read()

    if ret:
        h, w = frame.shape[:2]
        scale = min(1.0, 480 / max(w, h))

        if scale < 1:
            frame = cv2.resize(
                frame, None, fx=scale, fy=scale
            )

        last_frame = frame.copy()
        result = process(frame)

        h, w = result.shape[:2]
        factor = min(
            (sw - 12) / w,
            max(100, sh - 180) / h
        )

        result = cv2.resize(
            result,
            (max(1, int(w * factor)),
             max(1, int(h * factor)))
        )

        result = cv2.cvtColor(
            result, cv2.COLOR_BGR2RGB
        )

        image = ImageTk.PhotoImage(
            Image.fromarray(result)
        )

        label.config(image=image)
        label.image = image

    else:
        status.config(text="Camera frame unavailable")

    root.after(40, update)

buttons = [
    ("PREV", lambda: change_mode(-1)),
    ("NEXT", lambda: change_mode(1)),
    ("BG", capture_bg),
    ("CAM", switch_camera),
    ("EXIT", close)
]

for name, command in buttons:
    tk.Button(
        panel,
        text=name,
        command=command,
        bg="#16352c",
        fg="white",
        font=("Arial", 11, "bold"),
        relief="flat"
    ).pack(
        side="left",
        expand=True,
        fill="x",
        padx=2
    )

root.protocol("WM_DELETE_WINDOW", close)

if not cap.isOpened():
    status.config(text="Camera unavailable: check permissions")

update()
root.mainloop()