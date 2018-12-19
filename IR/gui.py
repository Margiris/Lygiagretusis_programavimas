import base64
from io import BytesIO
from multiprocessing.pool import Pool
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import copy

import algorithms

    # print(pixelValues)

    # imageDisplay.image = imageDisplay.image.point(lambda point: point + int(value) - int(imageDisplay.brightness))
    # imageDisplay.brightness = value
    # imageDisplay.updateImage()


def setBrightness(pixel):
    # print(pixel)
    return pixel * 2


def printAAA():
    print("AAA")


if __name__ == "__main__":
    statusBarHeight = 19
    buttonsWidth = 15

    # noinspection SpellCheckingInspection
    imageInitialData = '''iVBORw0KGgoAAAANSUhEUgAAAfQAAAH0CAYAAADL1t+KAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAF+mlUWHRYTUw6Y29tLmFkb2
    JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4gPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZT
    puczptZXRhLyIgeDp4bXB0az0iQWRvYmUgWE1QIENvcmUgNS42LWMxNDUgNzkuMTYzNDk5LCAyMDE4LzA4LzEzLTE2OjQwOjIyICAgICAgICAiPiA8cmRmOl
    JERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPiA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIi
    B4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIgeG1sbn
    M6cGhvdG9zaG9wPSJodHRwOi8vbnMuYWRvYmUuY29tL3Bob3Rvc2hvcC8xLjAvIiB4bWxuczp4bXBNTT0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL2
    1tLyIgeG1sbnM6c3RFdnQ9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZUV2ZW50IyIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZS
    BQaG90b3Nob3AgQ0MgMjAxOSAoV2luZG93cykiIHhtcDpDcmVhdGVEYXRlPSIyMDE4LTEyLTA3VDE5OjU5OjM0KzAyOjAwIiB4bXA6TW9kaWZ5RGF0ZT0iMj
    AxOC0xMi0wN1QyMTowMzozMSswMjowMCIgeG1wOk1ldGFkYXRhRGF0ZT0iMjAxOC0xMi0wN1QyMTowMzozMSswMjowMCIgZGM6Zm9ybWF0PSJpbWFnZS9wbm
    ciIHBob3Rvc2hvcDpDb2xvck1vZGU9IjMiIHBob3Rvc2hvcDpJQ0NQcm9maWxlPSJzUkdCIElFQzYxOTY2LTIuMSIgeG1wTU06SW5zdGFuY2VJRD0ieG1wLm
    lpZDo3OGNhNzQzMC01MWRhLWYyNDAtODlmYi0wMjJiMTBhYzc0NDIiIHhtcE1NOkRvY3VtZW50SUQ9ImFkb2JlOmRvY2lkOnBob3Rvc2hvcDo2MzQ0YTc0YS
    1kODNiLTc3NGQtYjczNS1lN2U4YjZiMjMyOTciIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDo5MDM4MjdiZS1lZjBkLTg4NDgtOTgzMy1kZT
    lkNGU4NjhlYTUiPiA8eG1wTU06SGlzdG9yeT4gPHJkZjpTZXE+IDxyZGY6bGkgc3RFdnQ6YWN0aW9uPSJjcmVhdGVkIiBzdEV2dDppbnN0YW5jZUlEPSJ4bX
    AuaWlkOjkwMzgyN2JlLWVmMGQtODg0OC05ODMzLWRlOWQ0ZTg2OGVhNSIgc3RFdnQ6d2hlbj0iMjAxOC0xMi0wN1QxOTo1OTozNCswMjowMCIgc3RFdnQ6c2
    9mdHdhcmVBZ2VudD0iQWRvYmUgUGhvdG9zaG9wIENDIDIwMTkgKFdpbmRvd3MpIi8+IDxyZGY6bGkgc3RFdnQ6YWN0aW9uPSJzYXZlZCIgc3RFdnQ6aW5zdG
    FuY2VJRD0ieG1wLmlpZDo3OGNhNzQzMC01MWRhLWYyNDAtODlmYi0wMjJiMTBhYzc0NDIiIHN0RXZ0OndoZW49IjIwMTgtMTItMDdUMjE6MDM6MzErMDI6MD
    AiIHN0RXZ0OnNvZnR3YXJlQWdlbnQ9IkFkb2JlIFBob3Rvc2hvcCBDQyAyMDE5IChXaW5kb3dzKSIgc3RFdnQ6Y2hhbmdlZD0iLyIvPiA8L3JkZjpTZXE+ID
    wveG1wTU06SGlzdG9yeT4gPC9yZGY6RGVzY3JpcHRpb24+IDwvcmRmOlJERj4gPC94OnhtcG1ldGE+IDw/eHBhY2tldCBlbmQ9InIiPz6ndGoyAAAqRUlEQV
    R4nO3debQsZ12v8afNYR5PQEJuCMNJFpgQpuyACWOCJ4hBIIScgAyKIAlhXuj1AF6uqLhI9IIKCJwYh6Wg3hxBQEU0ASJXLkPOkSxmlERGQSI5EoYLYtL3j6
    rXXbt2dXd1dVW91W89n7V6de+9e1f9uvvt+tZbw1uT6XSKJElabz8QuwBJkrQ6A12SpAQY6JIkJcBAlyQpAQa6JEkJMNAlSUqAgS5JUgIMdEmSEmCgS5KUAA
    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'''


    class ResizeEvent(Event):
        def __init__(self, width, height):
            super().__init__()
            self.width = width
            self.height = height


    class ResizingImageDisplay(Canvas):
        def __init__(self, parent, **kwargs):
            super().__init__(parent, **kwargs)

            self.image = Image.open(BytesIO(base64.b64decode(imageInitialData)))
            self.imagePixelValues = list(self.image.getdata())

            self.brightness = 0

            self.width = self.image.width
            self.height = self.image.height

            self.imageForDisplay = copy.copy(self.image)
            self.imageForDisplay.thumbnail((self.width, self.height), Image.ANTIALIAS)
            self.imageThumbnail = ImageTk.PhotoImage(self.imageForDisplay)
            self.imageOnCanvas = self.create_image(self.width / 2, self.height / 2, image=self.imageThumbnail)

        def onResize(self, event):
            deltaX = (event.width - self.width) / 2
            deltaY = (event.height - self.height) / 2

            self.width = event.width
            self.height = event.height

            # resize the canvas
            self.config(width=self.width, height=self.height)

            self.imageForDisplay = copy.copy(self.image)
            self.imageForDisplay.thumbnail((self.width, self.height), Image.ANTIALIAS)
            self.imageThumbnail = ImageTk.PhotoImage(self.imageForDisplay)
            self.itemconfig(self.imageOnCanvas, image=self.imageThumbnail)
            self.move(self.imageOnCanvas, deltaX, deltaY)

        def updateImage(self):
            if mainWindow.state() == "zoomed":
                event = ResizeEvent(mainWindow.winfo_width() - controlsFrame.winfo_width(),
                                    mainWindow.winfo_height())
            else:
                mainWindow.geometry(str(imageDisplay.image.width + controlsFrame.winfo_width()) + "x" +
                                    str(imageDisplay.image.height))
                event = ResizeEvent(imageDisplay.image.width, imageDisplay.image.height)

            self.onResize(event)


    def brightness(value):
        # print(value)
        temp = copy.copy(imageDisplay.imagePixelValues)

        print(imageDisplay.imagePixelValues)

        with Pool() as pool:
            imageDisplay.imagePixelValues = pool.map(setBrightness, temp)
        print(imageDisplay.imagePixelValues)
        # imageDisplay.image.putdata(imageDisplay.imagePixelValues)
        imageDisplay.updateImage()


    def openImage():
        imagePath = filedialog.askopenfilename(title="Select photo")
        sliderBrightness.set(0)
        imageDisplay.image = Image.open(imagePath)
        imageDisplay.updateImage()


    def saveImage():
        outputImagePath = filedialog.asksaveasfilename(title="Save as")
        imageDisplay.image.save(outputImagePath)

    mainWindow = Tk()

    imageDisplay = ResizingImageDisplay(mainWindow, highlightthickness=0)
    mainWindow.geometry(str(imageDisplay.image.width + (buttonsWidth * 7) + 10) + "x" + str(imageDisplay.image.height))
    imageDisplay.bind("<Configure>", lambda event: imageDisplay.onResize(event))
    imageDisplay.pack(side=LEFT, fill=BOTH, expand=YES)

    controlsFrame = Frame(mainWindow)
    controlsFrame.pack(side=RIGHT, fill=Y, expand=NO)

    buttonOpen = Button(controlsFrame, text="Open...", width=buttonsWidth, command=openImage)
    buttonOpen.pack()

    sliderBrightness = Scale(controlsFrame, from_=-100, to=100, orient=HORIZONTAL, width=buttonsWidth,
                             command=brightness)
    sliderBrightness.pack()

    buttonSave = Button(controlsFrame, text="Save image...", width=buttonsWidth, command=saveImage)
    buttonSave.pack()

    mainWindow.mainloop()

elif __name__ == "__mp_main__":
    a = 0
    # print("will do some threading!")

else:
    print(__name__)
    messagebox.showinfo("Error", "This package cannot be imported")

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
