import numpy as np
from deeplearningnumpy.cost_functions import MSE, CategoricalCrossEntropy, BinaryCrossEntropy
from deeplearningnumpy.activations import ActivationLeakyReLU, ActivationSoftmax, ActivationLogistic
from deeplearningnumpy.models import NeuralNetwork
from deeplearningnumpy.layers import ConvolutionalLayer, DenseLayer
from applyDigitRecognition import loadFrontEnd

BATCH_SIZE = 8    # The number of training samples in a batch
N_PIXELS = 784      # The number of pixels in a single image
N_EPOCHS = 10       # The number of times to iterate over the training set
LEARNING_RATE = 0.05 # How much the weights should be updated by during training

def loadMnist(fileNamePrefix):
    """Loads images and labels for MNIST."""
    #Load images
    with open("deeplearningnumpy/resources/" + fileNamePrefix + "-images.idx3-ubyte", "rb") as imagesFile:
        images = np.frombuffer(imagesFile.read(), np.uint8, offset=16)
        X = np.reshape(np.array(images) / 255, (-1, 784))

    #Load labels
    with open("deeplearningnumpy/resources/" + fileNamePrefix + "-labels.idx1-ubyte", "rb") as labelsFile:
        labels = np.frombuffer(labelsFile.read(), np.uint8, offset=8)
        Y = np.array(labels)
        #Convert labels to one-hot encoding
        Y = np.identity(np.max(labels)+1)[labels]    #Add one to account for 0 mapping to a vector too

    return X, Y

def loadMnistTraining():
    return loadMnist("train")

def loadMnistTesting():
    return loadMnist("t10k")

def oneHotEncoding(n):
    """Converts a digit from 0 to 9 to its one-hot encoding.

    Returns a 10-element list with a 1 at the nth index
    and zeros elsewhere.
    """
    encoding = [0]*10
    encoding[n] = 1
    return encoding

imageData, labelData = loadMnistTraining()

#Define network layers
'''
layer1 = DenseLayer(N_PIXELS, 1024, ActivationLeakyReLU(0.20))
layer2 = DenseLayer(1024, 256, ActivationLeakyReLU(0.20))
layer3 = DenseLayer(256, 64, ActivationLeakyReLU(0.20))
layer4 = DenseLayer(64, 10, ActivationSoftmax())
layers = [layer1, layer2, layer3, layer4]
'''
layers = [ ConvolutionalLayer(16, 4, 1, ActivationLeakyReLU(0.20), 2)
         , ConvolutionalLayer(32, 3, 16, ActivationLeakyReLU(0.20), 2)
         , DenseLayer(32 * 6**2, 10, ActivationLogistic())
]

# We use categorical cross entropy as we are trying to categorise samples into more than 2 categories
costFunction = MSE()

#Create new network
digitNetwork = NeuralNetwork("testNetwork", layers)

testImages, testLabels = loadMnistTesting()

#Load previous weights, or generate them if they don't exist
try:
    digitNetwork.loadWeights()
except FileNotFoundError:
    digitNetwork.train(imageData.reshape(-1, 1, 28, 28), labelData, costFunction, BATCH_SIZE, N_EPOCHS, LEARNING_RATE, True, testImages.reshape(-1, 1, 28, 28), testLabels)
    digitNetwork.saveWeights()

# Evaluate accuracy on test set
'''
digitNetwork.forward(testImages)
testOutputs = np.argmax(digitNetwork.getOutputs(), -1)
matchingCount = 0
for i in range(testOutputs.size):
    if testOutputs[i] == np.argmax(testLabels[i], -1):
        matchingCount += 1

print("Accuracy on test set: {}%".format((matchingCount / testOutputs.size) * 100))
'''
# Load the GUI application to interact with the network
loadFrontEnd(digitNetwork)