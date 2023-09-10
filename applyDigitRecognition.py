import numpy as np
from deeplearningnumpy.models import NeuralNetwork
from deeplearningnumpy.layers import ConvolutionalLayer, DenseLayer
from deeplearningnumpy.activations import ActivationLeakyReLU, ActivationSoftmax
from deeplearningnumpy.cost_functions import CategoricalCrossEntropy
from tkinter import *
from PIL import Image, ImageDraw
# Fix blurry widgets on screen
import ctypes

# Fix poor rendering of pen drawing
ctypes.windll.shcore.SetProcessDpiAwareness(1)

currentX, currentY = 0, 0

window = Tk()

outputText = StringVar()

outputLabel = Label(window, text="Output:")

def testImage(network, img):
    # Default if first layer is Dense layer
    pixels = np.reshape(img, (1, 28 * 28)) / 255.0
    if isinstance(network._layers[0], ConvolutionalLayer):
        pixels = np.reshape(img, (1, 28, 28)) / 255.0
    
    network.forward(pixels)
    return network.getOutputs()

def locateXy(event):
    """Get x,y coordinates of the user's mouse on the canvas."""
    global currentX, currentY
    currentX, currentY = event.x, event.y
    #print(currentX, currentY)

def addLine(event, canvas, draw):
    global currentX, currentY

    if (abs(event.x - currentX > 0)) or (abs(event.y - currentY) > 0):
        #Draw line on canvas between cursor positions
        canvas.create_line(currentX, currentY, event.x, event.y, width=20, capstyle=ROUND, smooth=True)

        #Make same change to image stored in memory
        draw.line([currentX, currentY, event.x, event.y], fill=(255), width=20, joint="curve")

        #Update cursor position
        currentX, currentY = event.x, event.y

def submitImage(network, originalImage):
    global outputText
    #Resize the image stored in memory
    resizedImage = originalImage.resize((28, 28), Image.ANTIALIAS)

    #Save file to disk
    resizedImage.save("currentImage.png")

    #Get and output the result of the network
    res = testImage(network, resizedImage)
    digitDetected = np.argmax(res)
    outputText.set(str(digitDetected))

def resetImage(canvas, img):
    img.paste((0,), (0, 0, img.width, img.height))
    canvas.delete("all")

def loadFrontEnd(network):
    #Create empty image in PIL to draw object on in memory
    img = Image.new("L", (255, 255))
    draw = ImageDraw.ImageDraw(img)

    window.title("Digit Classifier")
    window.rowconfigure(0, weight=1)
    window.rowconfigure(1, weight=1)
    window.rowconfigure(2, weight=1)
    window.columnconfigure(0, weight=1)
    window.columnconfigure(1, weight=2)
    window.columnconfigure(2, weight=2)

    #Create image submit button
    imageSubmit = Button(window, text="Submit image", command=lambda: submitImage(network, img))
    imageSubmit.grid(row=0, column=1, sticky='n')

    imageReset = Button(window, text="Reset image", command=lambda: resetImage(canvas, img))
    imageReset.grid(row=1, column=1, sticky='n')

    #Create text labels and output boxes
    outputLabel.grid(row=2, column=1, sticky='n')

    #outputString.set("")
    outputBox = Label(window, textvariable=outputText)
    outputBox.grid(row=2, column=2, sticky='n')

    #Create canvas
    canvas = Canvas(window, bg="white")
    canvas.grid(row=0, rowspan=3, column=0, sticky='nw')
    canvas.configure(width=250, height=250)

    #Binding for when user clicks on canvas
    canvas.bind('<Button-1>', locateXy)
    #Binding for when user drags on canvas
    canvas.bind('<B1-Motion>', lambda event: addLine(event, canvas, draw))

    #Show window
    window.mainloop()




            