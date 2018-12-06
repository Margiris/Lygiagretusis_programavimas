from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import copy
import tkinter.filedialog

# inputImagePath = "test1.png"
inputImagePath = "to2.jpg"
outputImagePath = "aaa"

statusBarHeight = 10
buttonsWidth = 500


class ResizingImageDisplay(Canvas):
    def __init__(self, parent, image, **kwargs):
        super().__init__(parent, **kwargs)

        self.width = parent.winfo_width()
        self.height = parent.winfo_height()

        self.image = image
        self.image.thumbnail((self.width, self.height), Image.ANTIALIAS)
        self.imageThumbnail = ImageTk.PhotoImage(self.image)
        self.imageOnCanvas = self.create_image(self.width / 2, self.height / 2, image=self.imageThumbnail)

        self.pack(fill=BOTH, expand=YES)

    def on_resize(self, event, image):
        deltaX = (event.width - self.width) / 2
        deltaY = (event.height - self.height) / 2

        self.width = event.width
        self.height = event.height

        # resize the canvas
        self.config(width=self.width - buttonsWidth, height=self.height - statusBarHeight)

        self.image = image

        self.image.thumbnail((self.width, self.height), Image.ANTIALIAS)
        self.imageThumbnail = ImageTk.PhotoImage(self.image)
        self.itemconfig(self.imageOnCanvas, image=self.imageThumbnail)
        self.move(self.imageOnCanvas, deltaX, deltaY)


def openImage():
    # try:
    return filedialog.askopenfilename(title="Select photo")
    # except AttributeError:            not working ATM
    #     print("trying one more time...")
    #     openImage(imagePath)


def brightness(point):
    return point * 0.5


def main():
    mainWindow = Tk()
    mainWindow.geometry('160x100')

    frameMain = Frame(mainWindow, bg="blue")
    frameMain.pack(side=TOP, fill=BOTH, expand=YES)

    frameStatusBar = Frame(mainWindow, height=statusBarHeight, bg="green")
    frameStatusBar.pack(side=BOTTOM, fill=X, expand=NO)

    frameImage = Frame(frameMain, bg="red")
    frameImage.pack(side=LEFT, fill=BOTH, expand=YES)

    frameButtons = Frame(frameMain, width=buttonsWidth, bg="gray")
    frameButtons.pack(side=LEFT, fill=Y, expand=NO)

    buttonOpen = Button(frameButtons, text="Open file...", command=openImage)
    buttonOpen.pack(side=RIGHT, anchor=NE)

    inputImagePath = openImage()
    image1 = Image.open(inputImagePath, 'r')

    imageDisplay = ResizingImageDisplay(frameImage, copy.copy(image1), width=1366, height=768, highlightthickness=0)
    imageDisplay.bind("<Configure>", lambda event: imageDisplay.on_resize(event, copy.copy(image1)))

    status = Label(frameStatusBar, text="nothing", bd=1, relief=SUNKEN, anchor=W)
    status.pack(side=BOTTOM, fill=X)

    # image1 = image1.point(lambda i: brightness(i))
    mainWindow.mainloop()


if __name__ == "__main__":
    main()

# for r, g, b, a in pixel_values:
#     print(a)
#
#     pixel_values = list(image1.getdata())
#
#     out = image1.point(lambda i: brightness(i))

# print(image.point())


# topFrame = Frame(mainWindow)
# bottomFrame = Frame(mainWindow)
#
# topFrame.pack()
# bottomFrame.pack(side=BOTTOM)
#
# button1 = Button(topFrame, text="Button 1", fg="red")
# button2 = Button(topFrame, text="Button 2", fg="blue")
# button3 = Button(topFrame, text="Button 3", fg="green")
# button4 = Button(bottomFrame, text="Button 4", fg="purple")
#
# button1.pack(side=LEFT)
# button2.pack(side=RIGHT)
# button3.pack(side=BOTTOM)
# button4.pack(side=RIGHT)

# one = Label(mainWindow, text="One", bg="red", fg="white")
# two = Label(mainWindow, text="Two", bg="green", fg="black")
# three = Label(mainWindow, text="Three", bg="blue", fg="white")
#
# one.pack()
# two.pack(fill=X)
# three.pack(side=LEFT)


# label1 = Label(mainWindow, text="Name")
# label2 = Label(mainWindow, text="Password")
#
# entry1 = Entry(mainWindow)
# entry2 = Entry(mainWindow)
#
# label1.grid(row=0, column=0, sticky=W)
# label2.grid(row=1, column=0, sticky=W)
#
# entry1.grid(row=0, column=1)
# entry2.grid(row=1, column=1)
#
# checkbox1 = Checkbutton(mainWindow, text="Stay logged in")
#
# checkbox1.grid(columnSpan=2)

#
# def printName():
#     print("Hi")
#
#
# button1 = Button(mainWindow, text="Print hi", command=printName)
# button1.pack()
#
#
# def printName(event):
#     print("Hi")
#
#
# button1 = Button(mainWindow, text="Print hi")
# button1.bind("<Button-1>", printName)
# button1.pack()
#
#
# def leftClick(event):
#     print("left")
#
#
# def middleClick(event):
#     print("middle")
#
#
# def rightClick(event):
#     print("right")
#
#
# frame = Frame(mainWindow, width=300, height=250)
# frame.bind("<Button-1>", leftClick)
# frame.bind("<Button-2>", middleClick)
# frame.bind("<Button-3>", rightClick)
#
# frame.pack()
