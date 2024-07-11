from tkinter import *
import os

root = Tk()
root.config(bg="blue")
root.geometry("900x1200")

lbl_title = Label(root, text="PU CODE HACKTHON")
lbl_title.config(font=("Calibri", 24, "bold"), bg="blue", fg="white")
lbl_title.pack(pady=50)

img = PhotoImage(file="epic.png")
lbl_image = Label(root, image=img)
lbl_image.pack()

def run_detect():
    os.system("python C:/Users/aloys/yolov5/detect.py --source 1")
btn_detect = Button(root, text="Crash Detection", command=run_detect)
btn_detect.config(font=("Calibri", 24, "bold"))
btn_detect.pack(pady=20)
def run_test():
    result = os.system("python C:/Users/aloys/OneDrive/Desktop/test.py")
    print("Script executed, return code:", result)

btn_test = Button(root, text="Theft Detection", command=run_test)
btn_test.config(font=("Calibri", 24, "bold"))
btn_test.pack(pady=20)
def run_crash():
    os.system("python C:/Users/aloys/yolov5/detect.py")

btn_crash = Button(root, text="Emergency Alert", command=run_crash)
btn_crash.config(font=("Calibri", 24, "bold"))
btn_crash.pack(pady=20)

root.mainloop()
