import numpy as np

class Layer:
    """Base class describing shared functionality of Dense, MaxPool and Convolutional layers."""

    def updateWeights(self, gradientSums, learningRate):
        """Updates the weights according to the accumulated gradients and the learning rate."""
        self.weights = np.subtract(self.weights, learningRate * gradientSums)
        # Sum the output deltas for each data item in the batch
        accumulatedDeltas = np.reshape(np.sum(self.outputDeltas, axis=0), (1, -1))
        # self.biases = np.subtract(self.biases, learningRate * accumulatedDeltas)

    def getOutputDeltas(self, expectedValues, costFunction):
        """Calculates delta values for each neuron in the output layer."""
        self.outputDeltas = costFunction.getDerivative(self.outputs, expectedValues)

    def getErrorGradients(self):
        return self.errorGradients

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
        if inputs.ndim > 2:
            inputs = inputs.reshape(-1, np.product(inputs.shape) // inputs.shape[0])
        self.inputs = inputs
        #Calculate weighted sum
        self.weightedSum = np.matmul(inputs, self.weights) + self.biases
        #Apply non-linear activation function
        self.outputs = self.activationFunction.forward(self.weightedSum)

    def getDeltas(self, outputDeltas):
        """Calculates delta values for each neuron, assuming that this is a hidden layer
        (i.e. not the output layer).
        """
        # Get the rate of change of the output with respect to the weighted sum
        activationDerivative = self.activationFunction.getDerivative(self.weightedSum)

        # Get the rate of change of the error with respect to the weighted sum, then multiply with the weights to
        # get the rate of change of the error with respect to the inputs
        weightedSumDerivative = np.multiply(outputDeltas, activationDerivative)
        self.inputDeltas = np.matmul(weightedSumDerivative, self.weights.T)

        self.errorGradients = np.matmul(weightedSumDerivative.T, self.inputs).T

class ConvolutionalLayer(Layer):
    def __init__(self, nOfFilters, filterSize, previousNOfFilters, activationFunction, stride=1):
        super().__init__()
        self.nOfFilters = nOfFilters
        self.filterSize = filterSize
        self.previousNOfFilters = previousNOfFilters
        self.activationFunction = activationFunction
        self.stride = stride

        # Randomly initialise filters of layer with He initialisation
        self.weights = np.random.randn(nOfFilters, previousNOfFilters, filterSize, filterSize) * (2.0 / (previousNOfFilters * filterSize ** 2))**0.5
        # Biases are initialised as zeros
        self.biases = np.zeros((nOfFilters), dtype=np.float32)

    def forward(self, inputs):
        """Performs the convolution and stores the result after applying
        the activation function as the output of the layer.
        """
        self.inputs = inputs
        newImageSize = (inputs.shape[-1] - self.filterSize) // self.stride + 1
        self.convolutionOutputs = np.zeros((inputs.shape[0], self.nOfFilters, newImageSize, newImageSize), dtype=np.float32)
        #Perform the cross-correlation operation for each input and filter
        for i, input in enumerate(inputs):
            for j, filter in enumerate(self.weights):
                self.convolutionOutputs[i, j] = self.crossCorrelate(input, filter, self.stride, self.biases)

        #Get the output of the activation function
        self.outputs = self.activationFunction.forward(self.convolutionOutputs)

    def crossCorrelate(self, image, filter, stride, biases = None):
        if image.ndim < 3:
            image = np.expand_dims(image, 0)
        imageDepth, imageSize, _ = image.shape
        if filter.ndim < 3:
            filter = np.expand_dims(filter, 0)
        filterDepth, filterSize, _ = filter.shape

        # Filter must should not go outside image bounds when moved by stride distance.
        assert ((imageSize - filterSize) / stride) % 1 == 0
        # Filter depth must match image depth for convolution to be defined.
        assert filterDepth == imageDepth

        outImageSize = (imageSize - filterSize) // stride + 1
        out = np.zeros((outImageSize, outImageSize), dtype=np.float32)
        for outRow, maskedRow in enumerate(range(0, imageSize - filterSize + 1, stride)):
            for outCol, maskedColumn in enumerate(range(0, imageSize - filterSize + 1, stride)):
                maskedImage = image[:, maskedRow:maskedRow+filterSize, maskedColumn:maskedColumn+filterSize]
                res = np.sum(np.multiply(maskedImage, filter)) # + np.tensordot(biases, np.ones((outImageSize, outImageSize)))
                out[outRow, outCol] = res

        return out

    def getDeltas(self, outputDeltas):
        """Calculates delta values for each neuron, assuming that this is a hidden layer
        (i.e. not the output layer).
        """
        activationDerivative = self.activationFunction.getDerivative(self.convolutionOutputs)
        weightedSumDerivative = np.multiply(outputDeltas.reshape(activationDerivative.shape), activationDerivative)

        dilatedDerivative = np.zeros((weightedSumDerivative.shape[0], 
        weightedSumDerivative.shape[1], 
        weightedSumDerivative.shape[2] + (weightedSumDerivative.shape[2] - 1) * (self.stride - 1), 
        weightedSumDerivative.shape[3] + (weightedSumDerivative.shape[3] - 1) * (self.stride - 1)), 
        dtype=np.float32)

        dilatedDerivative[:,:,::self.stride,::self.stride] = weightedSumDerivative

        #The error gradient of the filter is the convolution of the dilated output derivative over the inputs
        self.errorGradients = np.zeros(self.weights.shape, dtype=np.float32)
        for i in range(self.nOfFilters):
            for j in range(self.previousNOfFilters):
                #for i in range(self.nOfFilters):
                self.errorGradients[i][j] = sum([self.crossCorrelate(self.inputs[k, j], dilatedDerivative[k, i], 1) for k in range(self.inputs.shape[0])])

        #Rotate filters by 180 degrees
        rotatedFilters = np.rot90(self.weights, k = 2, axes=(-1,-2))

        #Pad the dilated output derivative
        paddedDerivative = np.pad(dilatedDerivative, ((0,0), (0,0), (rotatedFilters.shape[-2] - 1, rotatedFilters.shape[-2] - 1), (rotatedFilters.shape[-1] - 1, rotatedFilters.shape[-1] - 1)), 'constant', constant_values=(0))
        self.inputDeltas = np.zeros(self.inputs.shape, dtype=np.float32)
        for k in range(self.inputs.shape[0]):
            for j in range(self.previousNOfFilters):
                self.inputDeltas[k, j] = sum([self.crossCorrelate(paddedDerivative[k, i], rotatedFilters[i, j], 1) for i in range(self.nOfFilters)])

class MaxPoolLayer(Layer):
    def __init__(self, windowSize, stride = 1):
        super().__init__()
        # A pooling layer has no weights, so we just assign a zero array
        self.weights = np.zeros((1))
        self.windowSize = windowSize
        self.stride = stride
        self.errorGradients = None

    def forward(self, inputs):
        """Performs max pooling to generate the layer's output."""
        self.inputs = inputs
        self.outputs = self.maxPool(inputs)

    def maxPool(self, images):
        """Performs max pooling to downsample the matrix."""
        (batchSize, depth, imageSize, _) = images.shape

        #Get size of output tensor
        outputSize = (imageSize - self.windowSize) // self.stride + 1

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
                out[:, :, currentRow // self.stride, currentCol // self.stride] = np.max(subImages, axis=(-2, -1))

                #Move section being considered to right by stride length
                currentCol += self.stride
            
            #Move section being considered down by stride length
            currentRow += self.stride
        
        #Return the output matrix
        return out

    def getDeltas(self, outputDeltas):
        """Calculates delta values, assuming that this is a hidden layer.
        Even though this layer has no weights, we still need to propagate derivatives
        backwards from subsequent layers.
        """
        '''
        if isinstance(nextLayer, DenseLayer):
            #Get the rate of change of the error with respect to the output
            errorSize = np.dot(nextLayer.outputDeltas, nextLayer.weights.T)

            #Reshape to rank 4 tensor
            errorSize = np.reshape(errorSize, self.outputs.shape)

        else:
            errorSize = np.multiply(nextLayer.outputDeltas, np.ones(self.outputs.shape))
        '''
        (batchSize, depth, imageSize, _) = self.inputs.shape

        #Get size of output tensor
        outputSize = (imageSize - self.windowSize) // self.stride + 1

        #Create rank 4 tensor to store output of max pooling
        out = np.zeros((batchSize, depth, imageSize, imageSize), dtype=np.float32)

        #Move window vertically across image

        reshapedOutputDeltas = outputDeltas.reshape((batchSize, depth, outputSize, outputSize))

        for batchItem in range(batchSize):
            for d in range(depth):
                currentRow = 0
                while currentRow < imageSize - self.windowSize:
                    #Move window horizontally across image
                    currentCol = 0

                    while currentCol < imageSize - self.windowSize:
                        #Get submatrix
                        subMatrix = self.inputs[batchItem, d, currentRow:currentRow + self.windowSize, currentCol:currentCol + self.windowSize]

                        #Get indices of maximum values within each sub image
                        #print(subMatrix.reshape(self.windowSize*self.windowSize))
                        flattenedIndices = np.argmax(subMatrix.reshape(self.windowSize*self.windowSize))

                        out[batchItem, d, currentRow + (flattenedIndices // self.windowSize), currentCol + (flattenedIndices % self.windowSize)] = reshapedOutputDeltas[batchItem, d, (currentRow // self.stride), (currentCol // self.stride)]

                        #Move section being considered to right by stride length
                        currentCol += self.stride
                    
                    #Move section being considered down by stride length
                    currentRow += self.stride
        
        #Define output deltas
        self.inputDeltas = out