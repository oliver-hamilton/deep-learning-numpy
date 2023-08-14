import numpy as np

class Layer:
    """Base class describing shared functionality of Dense, MaxPool and Convolutional layers."""

    def updateWeights(self, gradientSums, learningRate):
        """Updates the weights according to the accumulated gradients and the learning rate."""
        self.weights = np.subtract(self.weights, learningRate * gradientSums)
        # Sum the output deltas for each data item in the batch
        accumulatedDeltas = np.reshape(np.sum(self.outputDeltas, axis=0), (1, -1))
        self.biases = np.subtract(self.biases, learningRate * accumulatedDeltas)

    def getOutputDeltas(self, expectedValues, costFunction):
        """Calculates delta values for each neuron in the output layer."""
        costDerivative = costFunction.getDerivative(self.outputs, expectedValues)
        #Get the rate of change of the output with respect to the weighted sum
        activationDerivative = self.activationFunction.getDerivative(self.weightedSum)
        #Mulitply the terms together
        self.outputDeltas = np.multiply(costDerivative, activationDerivative)

class DenseLayer(Layer):
    def __init__(self, nInputs, nNeurons, activationFunction):
        super().__init__()
        # Weights are initialised using He initialisation
        self.weights = np.random.randn(nInputs, nNeurons) * (2.0 / nInputs)**0.5
        # Biases are initialised as zero vectors
        self.biases = np.zeros((1, nNeurons), dtype=np.float32)
        self.activationFunction = activationFunction
        
    def forward(self, inputs):
        """Calculates the weighted sum of the inputs and stores the result
        after applying the activation function as the output of the layer.
        """
        self.inputs = inputs
        #Calculate weighted sum
        self.weightedSum = np.matmul(inputs, self.weights) + self.biases
        #Apply non-linear activation function
        self.outputs = self.activationFunction.forward(self.weightedSum)

    def getHiddenDeltas(self, nextLayer):
        """Calculates delta values for each neuron, assuming that this is a hidden layer
        (i.e. not the output layer).
        """
        #Get the rate of change of the error with respect to the output
        errorSize = np.matmul(nextLayer.outputDeltas, nextLayer.weights.T)

        #Get the rate of change of the output with respect to the weighted sum
        activationDerivative = self.activationFunction.getDerivative(self.weightedSum)

        #Multiply the terms together
        self.outputDeltas = np.multiply(errorSize, activationDerivative)

    def getErrorGradients(self):
        """Turn delta values into error gradients by taking a weighted sum with respect to the inputs."""
        errorGradients = np.matmul(self.outputDeltas.T, self.inputs).T
        return errorGradients

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
    
class MaxPoolLayer(Layer):
    def __init__(self, windowSize, stride):
        super().__init__()
        # A pooling layer has no weights, so we just assign a zero array
        self.weights = np.zeros((1))
        self.windowSize = windowSize
        self.stride = stride

    def forward(self, inputs):
        """Performs max pooling to generate the layer's output."""
        self.inputs = inputs
        self.outputs = self.maxPool(inputs)

    def maxPool(self, images):
        """Performs max pooling to downsample the matrix."""
        (batchSize, depth, imageSize, _) = images.shape

        #Get size of output tensor
        outputSize = int(np.floor((imageSize - self.windowSize) / self.stride)) + 1

        #Create order 3 tensor to store output of max pooling
        out = np.zeros((batchSize, depth, outputSize, outputSize), dtype=np.float32)

        #Move window vertically across image
        currentRow = 0

        while currentRow < imageSize - self.windowSize:
            #Move window horizontally across image
            currentCol = 0

            while currentCol < imageSize - self.windowSize:
                #Get submatrices
                subImages = images[:, :, currentRow:currentRow + self.windowSize, currentCol:currentCol + self.windowSize]

                #Get maximum within each sub image
                out[:, :, currentRow // self.stride, currentCol // self.stride] = np.max(subImages, axis=(2,3))

                #Move section being considered to right by stride length
                currentCol += self.stride
            
            #Move section being considered down by stride length
            currentRow += self.stride
        
        #Return the output matrix
        return out

    def getHiddenDeltas(self, nextLayer):
        """Calculates delta values, assuming that this is a hidden layer.
        Even though this layer has no weights, we still need to propagate derivatives
        backwards from subsequent layers.
        """
        if isinstance(nextLayer, DenseLayer):
            #Get the rate of change of the error with respect to the output
            errorSize = np.dot(nextLayer.outputDeltas, nextLayer.weights.T)

            #Reshape to rank 4 tensor
            errorSize = np.reshape(errorSize, self.outputs.shape)

        else:
            errorSize = np.multiply(nextLayer.outputDeltas, np.ones(self.outputs.shape))

        (batchSize, depth, imageSize, _) = self.inputs.shape

        #Get size of output tensor
        outputSize = self.inputs.shape[3]

        #Create rank 4 tensor to store output of max pooling
        out = np.zeros((batchSize, depth, outputSize, outputSize), dtype=np.float32)

        #Move window vertically across image
        currentRow = 0

        for batchItem in range(batchSize):

            for d in range(depth):

                while currentRow < imageSize - self.windowSize:
                    #Move window horizontally across image
                    currentCol = 0

                    while currentCol < imageSize - self.windowSize:
                        #Get submatrix
                        subMatrix = self.inputs[batchItem, d, currentRow:currentRow + self.windowSize, currentCol:currentCol + self.windowSize]

                        #Get indices of maximum values within each sub image
                        #print(subMatrix.reshape(self.windowSize*self.windowSize))
                        flattenedIndices = np.argmax(subMatrix.reshape(self.windowSize*self.windowSize))

                        out[batchItem, d, currentRow + (flattenedIndices // self.windowSize), currentCol + (flattenedIndices % self.windowSize)] = errorSize[batchItem, d, (currentRow // self.stride), (currentCol // self.stride)]

                        #Move section being considered to right by stride length
                        currentCol += self.stride
                    
                    #Move section being considered down by stride length
                    currentRow += self.stride
        
        #Define output deltas
        self.outputDeltas = out

    def getErrorGradients(self):
        """There are no error gradients to return."""
        return None

    def updateWeights(self, gradientSums, learningRate):
        """A pooling layer has no weights to update, so calling this method has no effect."""
        pass