import numpy as np
from deeplearningnumpy.layers.base import Layer
from deeplearningnumpy.layers.dense import DenseLayer

class ConvolutionalLayer(Layer):
    def __init__(self, nOfFilters, filterSize, previousNOfFilters, activationFunction, stride=1):
        super().__init__()
        self.nOfFilters = nOfFilters
        self.activation = activationFunction

        # Randomly initialise filters of layer
        self.weights = 0.025 * np.random.randn(nOfFilters, previousNOfFilters, filterSize, filterSize)
        # Biases are initialised as zeros
        self.biases = np.zeros((nOfFilters), dtype=np.float32)
        self.stride = stride

        #Get the size of the filter
        (_, _, filterSize, _) = self.weights.shape
        self.windowSize = filterSize

    def forward(self, inputs):
        """Performs the convolution and stores the result after applying
        the activation function as the output of the layer.
        """
        self.inputs = inputs
        #Perform a convolution
        self.convolutionOutputs = self.convolution(inputs, self.filters, self.biases)

        #Get the output of the activation function
        self.outputs = self.activation.forward(self.convolutionOutputs)

    def convolution(self, images, filters, stride, biases=None):
        """Performs a convolution of a set of filters over a set of images with a given stride and set of biases."""
        nFilters, filterDepth, filterSize, _ = filters.shape
        nImages, imageDepth, imageSize, _ = images.shape

        # Filter must should not go outside image bounds when moved by stride distance.
        assert ((imageSize - filterSize) / stride) % 1 == 0
        # Filter depth must match image depth for convolution to be defined.
        assert filterDepth == imageDepth

        if not biases:
            biases = np.zeros((nFilters))

        outImageSize = (imageSize - filterSize) // stride + 1
        out = np.zeros((nImages, nFilters, outImageSize, outImageSize))

        for imageIndex in range(nImages):
            for filterIndex in range(nFilters):
                currentFilter = filters[filterIndex]
                for outRow, maskedRow in enumerate(range(0, imageSize - filterSize + 1, stride)):
                    for outCol, maskedColumn in enumerate(range(0, imageSize - filterSize + 1, stride)):
                        maskedImage = images[imageIndex, :, maskedRow:maskedRow+filterSize, maskedColumn:maskedColumn+filterSize]
                        res = np.sum(np.multiply(maskedImage, currentFilter)) + np.tensordot(biases, np.ones(outImageSize, outImageSize))
                        out[imageIndex, filterIndex, outRow, outCol] = res

        return out

    def getHiddenDeltas(self, nextLayer):
        """Calculates delta values for each neuron, assuming that this is a hidden layer
        (i.e. not the output layer).
        """
        activationDerivative = self.activation.getDerivative(self.convolutionOutputs)
        if isinstance(nextLayer, DenseLayer):
            errorSize = np.dot(nextLayer.outputDeltas, nextLayer.weights.T)  
            #Reshape error to match convolutional layer
            errorSize = np.reshape(errorSize, activationDerivative.shape)
            #Get the derivative of the error with respect to the output before activation
            outputDerivative = np.multiply(errorSize, activationDerivative)

        else:
            outputDerivative = np.multiply(nextLayer.outputDeltas, activationDerivative)

        dilatedDerivative = np.zeros((outputDerivative.shape[0], 
        outputDerivative.shape[1], 
        outputDerivative.shape[2] + (outputDerivative.shape[2] - 1) * (self.stride - 1), 
        outputDerivative.shape[3] + (outputDerivative.shape[3] - 1) * (self.stride - 1)), 
        dtype=np.float32)

        dilatedDerivative[:,:,::self.stride,::self.stride] = outputDerivative

        #The error gradient of the filter is the convolution of the dilated output derivative over the inputs
        self.filterErrorGradient = self.convolution(self.inputs, dilatedDerivative, 1)

        #Rotate filters by 180 degrees
        temp = np.rot90(self.weights, axes=(-1,-2))
        rotatedFilters = np.rot90(temp, axes=(-1,-2))

        #Pad the dilated output derivative
        paddedDerivative = np.pad(dilatedDerivative, ((0,0), (0,0), (rotatedFilters.shape[2] - 1, rotatedFilters.shape[2] - 1), (rotatedFilters.shape[2] - 1, rotatedFilters.shape[2] - 1)), 'constant', constant_values=(0))
        self.outputDeltas = self.convolution(paddedDerivative, rotatedFilters, 1)
    
    def getErrorGradients(self):
        """Returns the convolution of the dilated output derivative over the inputs."""
        return self.filterErrorGradient
