import numpy as np

class Layer:
    """Base class describing shared functionality of Dense, MaxPool and Convolutional layers."""

    def updateWeights(self, learningRate):
        """Updates the weights and biases of this layer according to the accumulated gradients and the learning rate.

        If this ``Layer`` object does not have any associated weights or biases (e.g. ``MaxPoolLayer``), then calling this function has no effect.

        Parameters
        ----------
        learningRate : float
            Controls the amount by which the weights and biases of the layer are changed on each step. 
            Both the weights and biases are updated at the same rate.

        """
        if type(self.weights) == np.ndarray:
            self.weights = np.subtract(self.weights, learningRate * self.weightGradients)
            self.biases = np.subtract(self.biases, learningRate * self.biasGradients)

    def getOutputDeltas(self, expectedValues, costFunction):
        """Calculates delta values for each neuron, assuming that this is the output layer.

        In other words, this function computes the derivative of the cost function with respect
        to each of this layer's outputs. This function should only be called if this ``Layer`` object
        is the last layer of some ``NeuralNetwork`` object.
        
        Parameters
        ----------
        expectedValues : numpy.ndarray
            The desired values for this layer's outputs. This parameter should have dimensions `(batch_size, out_1, out_2, ..., out_n)`,
            where `(out_1, out_2, ..., out_n)` are the dimensions of this layer's outputs.
        costFunction: CostFunction
            A cost function whose derivative is calculated with respect to the outputs of this layer.

        Notes
        -----
        ``CostFunction`` is any of ``MSE``, ``BinaryCrossEntropy``, or ``CategoricalCrossEntropy``.
        """
        self.outputDeltas = costFunction.getDerivative(self.outputs, expectedValues)

    def getWeightGradients(self):
        """Returns the derivative of the cost function with respect to this layer's weights.
        
        Returns
        -------
        numpy.ndarray
            The gradients with respect to the weights. Has dimensions equal to the dimensions of the weights.
        """
        return self.weightGradients
    
    def getBiasGradients(self):
        """Returns the derivative of the cost function with respect to this layer's biases.
        
        Returns
        -------
        numpy.ndarray
            The gradients with respect to the biases. Has dimensions equal to the dimensions of the biases.
        """
        return self.biasGradients

class DenseLayer(Layer):
    """A fully-connected layer of a neural network.
    
    Weights are initialised using He initialisation, and biases are zero-initialised.

    Attributes
    ----------
    nInputs : int
        The total number of inputs to this layer.
    nOutputs : int
        The total number of outputs (i.e. neurons) of this layer.
    activationFunction : Activation
        The activation function to be applied after taking the weighted sum of the inputs.
    """
    def __init__(self, nInputs, nOutputs, activationFunction):
        super().__init__()
        # Weights are initialised using He initialisation
        self._weights = np.random.randn(nInputs, nOutputs) * (2.0 / nInputs)**0.5
        # Biases are initialised as zero vectors
        self._biases = np.zeros((1, nOutputs), dtype=np.float32)
        self._activationFunction = activationFunction
        
    def forward(self, inputs):
        """Calculates the weighted sum of `inputs` and stores the output after applying the activation function to it.

        Parameters
        ----------
        inputs : numpy.ndarray
            The inputs to the layer, which should have dimensions `(batchSize, nInputs)`.
            Otherwise, the inputs will be flattened beyond axis `0`.
        """
        if inputs.ndim > 2:
            inputs = inputs.reshape(-1, np.product(inputs.shape) // inputs.shape[0])
        self.inputs = inputs
        # Calculate weighted sum
        self.weightedSum = np.matmul(inputs, self._weights) + self._biases
        # Apply non-linear activation function
        self.outputs = self._activationFunction.forward(self.weightedSum)

    def getDeltas(self, outputDeltas):
        """Calculates gradients for each neuron in this layer.

        These gradients include the derivative of the cost function with respect to
        the inputs, the weights, and the biases.

        Parameters
        ----------
        outputDeltas : numpy.ndarray
            The derivative of the cost function with respect to each of the outputs of this layer.
            Equivalently, the derivative of the cost function with respect to each of the inputs of the next layer
            (if there is a next layer).
            Has dimensions `(batchSize, nOutputs)`.
        """
        # Get the rate of change of the output with respect to the weighted sum
        activationDerivative = self._activationFunction.getDerivative(self.weightedSum)

        # Get the rate of change of the error with respect to the weighted sum
        weightedSumDerivative = np.multiply(outputDeltas, activationDerivative)
        # Derivative of cost w.r.t inputs
        self.inputDeltas = np.matmul(weightedSumDerivative, self._weights.T)
        # Derivative of cost w.r.t weights
        self.weightGradients = np.matmul(weightedSumDerivative.T, self.inputs).T
        # Derivative of cost w.r.t biases
        self.biasGradients = np.sum(weightedSumDerivative, axis=0)

class ConvolutionalLayer(Layer):
    """A convolutional layer of a neural network.
    
    Weights are initialised using He initialisation, and biases are zero-initialised.

    Attributes
    ----------
    nOfFilters : int
        The number of filters convolved over the input.
    inputSize : int
        Equal to `width` if the dimensions of the input are `(batchSize, depth, width, width)`.
    filterSize : int
        The width / height of each of the filters. Note that this means only square filters can be used.
    previousNOfFilters : int
        The number of filters used in the previous layer. If the previous layer was not a ``ConvolutionalLayer``,
        then this should be set to 1.
    activationFunction : Activation
        The activation function to be applied after convolving the filters over the inputs.
    stride : int, optional
        The step size to move the kernel by in each cross-correlation operation. Defaults to 1.
        
    """
    def __init__(self, nOfFilters, inputSize, filterSize, previousNOfFilters, activationFunction, stride=1):
        super().__init__()
        self._nOfFilters = nOfFilters
        self._filterSize = filterSize
        self._previousNOfFilters = previousNOfFilters
        self._activationFunction = activationFunction
        self._stride = stride

        # Randomly initialise filters of layer with He initialisation
        self._weights = np.random.randn(nOfFilters, previousNOfFilters, filterSize, filterSize) * (2.0 / (previousNOfFilters * filterSize ** 2))**0.5
        # Untied biases are initialised as zeros
        self._outputWidth = (inputSize - filterSize) // stride + 1
        self._biases = np.zeros((nOfFilters, self._outputWidth, self._outputWidth), dtype=np.float32)

    def forward(self, inputs):
        """Performs the cross-correlation of each filter over each input in the batch
        and stores the result after applying the activation function.

        Parameters
        ----------
        inputs : numpy.ndarray
            The inputs to which the cross-correlation operation is to be applied.
            Has dimensions `(batchSize, depth, inputSize, inputSize)`.
        """
        self.inputs = inputs
        self.convolutionOutputs = np.zeros((inputs.shape[0], self._nOfFilters, self._outputWidth, self._outputWidth), dtype=np.float32)
        # Perform the cross-correlation operation for each input and filter
        self.convolutionOutputs = self.crossCorrelate(inputs, self._weights, self._stride, self._biases)

        # Pass the output of the cross correlation into the activation function
        self.outputs = self._activationFunction.forward(self.convolutionOutputs)

    def crossCorrelate(self, images, filters, stride, biases = None):
        """Performs the cross-correlation of the filters over the inputs.

        If `images` has dimensions ``(batchSize, imagesDepth, inputSize, inputSize)``,
        and `filters` has dimensions ``(nFilters, filterDepth, filterSize, filterSize)``,
        we require that `imagesDepth = filterDepth`.
        Insert new axes at position 0 in `images` and `filters` while either has fewer than
        4 dimensions.
        
        Parameters
        ----------
        images : numpy.ndarray
            The images to be convolved over.
        filters : numpy.ndarray
            The filters to convolve over `images`.
        stride : int
            The step size to move each kernel by in the cross-correlation operations. Defaults to 1.
        biases : numpy.ndarray
            Added to the results after the cross-correlations has been applied.

        Returns
        -------
        numpy.ndarray
            The result of the cross-correlation. Has dimensions ``(batchSize, nFilters, outputSize, outputSize)``,
            where ``outputSize = (inputSize - filterSize) // stride + 1``.
        """
        while images.ndim < 4:
            images = np.expand_dims(images, 0)
        _, imageDepth, imageSize, _ = images.shape
        while filters.ndim < 4:
            filters = np.expand_dims(filters, 0)
        _, filterDepth, filterSize, _ = filters.shape

        # Filter must should not go outside image bounds when moved by stride distance.
        assert ((imageSize - filterSize) / stride) % 1 == 0
        # Filter depth must match image depth for convolution to be defined.
        assert filterDepth == imageDepth

        outImageSize = (imageSize - filterSize) // stride + 1
        out = np.zeros((images.shape[0], filters.shape[0], outImageSize, outImageSize), dtype=np.float32)
        for outRow, maskedRow in enumerate(range(0, imageSize - filterSize + 1, stride)):
            for outCol, maskedColumn in enumerate(range(0, imageSize - filterSize + 1, stride)):
                maskedImage = images[:, :, maskedRow:maskedRow+filterSize, maskedColumn:maskedColumn+filterSize]
                maskedImage = np.expand_dims(maskedImage, 1)
                res = np.sum(np.multiply(maskedImage, filters), axis=(-1, -2, -3)) # + np.tensordot(biases, np.ones((outImageSize, outImageSize)))
                out[:, :, outRow, outCol] = res
        if type(biases) == np.ndarray:
            return out + biases
        else:
            return out

    def getDeltas(self, outputDeltas):
        """Calculates gradients for each neuron in this layer.

        These gradients include the derivative of the cost function with respect to
        the inputs, the weights, and the biases.

        Parameters
        ----------
        outputDeltas : numpy.ndarray
            The derivative of the cost function with respect to each of the outputs of this layer.
            Equivalently, the derivative of the cost function with respect to each of the inputs of the next layer
            (if there is a next layer).
            Has dimensions `(batchSize, nFilters, outputWidth, outputWidth)`.
        """
        # Use the derivative of the activation function to calculate the derivative of the cost w.r.t to the convolution output
        activationDerivative = self._activationFunction.getDerivative(self.convolutionOutputs)
        weightedSumDerivative = np.multiply(outputDeltas.reshape(activationDerivative.shape), activationDerivative)

        # The bias gradients are just the sum of the convolution derivatives over all samples in the batch
        self.biasGradients = np.sum(weightedSumDerivative, axis=0)

        # Dilate the convolution derivative with zeros if the stride value is greater than 1
        dilatedDerivative = np.zeros((weightedSumDerivative.shape[0], 
        weightedSumDerivative.shape[1], 
        weightedSumDerivative.shape[2] + (weightedSumDerivative.shape[2] - 1) * (self._stride - 1), 
        weightedSumDerivative.shape[3] + (weightedSumDerivative.shape[3] - 1) * (self._stride - 1)), 
        dtype=np.float32)

        dilatedDerivative[:,:,::self._stride,::self._stride] = weightedSumDerivative

        #The gradient w.r.t each filter is the convolution of the dilated output derivative over the inputs
        self.weightGradients = np.zeros(self._weights.shape, dtype=np.float32)
        for i in range(self._nOfFilters):
            for j in range(self._previousNOfFilters):
                #for i in range(self.nOfFilters):
                self.weightGradients[i][j] = sum([self.crossCorrelate(self.inputs[k, j], dilatedDerivative[k, i], 1) for k in range(self.inputs.shape[0])])

        #Rotate filters by 180 degrees (to perform convolution instead of cross-correlation)
        rotatedFilters = np.rot90(self._weights, k = 2, axes=(-1,-2))

        #Pad the dilated output derivative and use this to backpropagate the gradient through the convolution
        paddedDerivative = np.pad(dilatedDerivative, ((0,0), (0,0), (rotatedFilters.shape[-2] - 1, rotatedFilters.shape[-2] - 1), (rotatedFilters.shape[-1] - 1, rotatedFilters.shape[-1] - 1)), 'constant', constant_values=(0))
        self.inputDeltas = np.zeros(self.inputs.shape, dtype=np.float32)
        for k in range(self.inputs.shape[0]):
            for j in range(self._previousNOfFilters):
                self.inputDeltas[k, j] = sum([self.crossCorrelate(paddedDerivative[k, i], rotatedFilters[i, j], 1) for i in range(self._nOfFilters)])

class MaxPoolLayer(Layer):
    """A max pooling layer of a neural network.

    Attributes
    ----------
    windowSize : int
        The maximum value in each windowSize x windowSize submatrix is propagated to the next layer.
    stride : int, optional
        The step size to move the window by after each pooling operation. Defaults to 1.
        
    """
    def __init__(self, windowSize, stride = 1):
        super().__init__()
        # A pooling layer has no weights, so we just assign None to it
        self._weights = None
        self._windowSize = windowSize
        self._stride = stride
        # self.errorGradients = None

    def forward(self, inputs):
        """Performs max pooling to generate this layer's output.

        Parameters
        ----------
        inputs : numpy.ndarray
            The inputs to which the max pooling operation is to be applied.
            Has dimensions `(batchSize, depth, inputSize, inputSize)`.
        """
        self.inputs = inputs
        self.outputs = self.maxPool(inputs)

    def maxPool(self, images):
        """Performs max pooling to downsample the matrix.

        Parameters
        ----------
        images : numpy.ndarray
            The images to apply max pooling to.

        Returns
        -------
        numpy.ndarray
            The result of the max pooling. Has dimensions ``(batchSize, depth, outputSize, outputSize)``,
            where ``(batchSize, depth, inputSize, inputSize)`` are the dimensions of the input, and
            ``outputSize = (inputSize - windowSize) // stride + 1``.
        """
        (batchSize, depth, imageSize, _) = images.shape

        # Get size of output tensor
        outputSize = (imageSize - self._windowSize) // self._stride + 1

        # Create 4D array to store output of max pooling
        out = np.zeros((batchSize, depth, outputSize, outputSize), dtype=np.float32)

        # Move window vertically across image
        currentRow = 0

        while currentRow < imageSize - self._windowSize:
            # Move window horizontally across image
            currentCol = 0
            while currentCol < imageSize - self._windowSize:
                # Get submatrices
                subImages = images[:, :, currentRow:currentRow + self._windowSize, currentCol:currentCol + self._windowSize]

                # Get maximum within each sub image
                out[:, :, currentRow // self._stride, currentCol // self._stride] = np.max(subImages, axis=(-2, -1))

                # Move section being considered to right by stride length
                currentCol += self._stride
            
            # Move section being considered down by stride length
            currentRow += self._stride
        
        return out

    def getDeltas(self, outputDeltas):
        """Calculates gradients for each neuron in this layer.

        These gradients include only the derivative of the cost function with respect to
        the inputs, since there are no weights or biases associated with this layer type.
        Note that these gradients still need to be calculated in order to propagate derivatives
        backwards from subsequent layers.

        Parameters
        ----------
        outputDeltas : numpy.ndarray
            The derivative of the cost function with respect to each of the outputs of this layer.
            Equivalently, the derivative of the cost function with respect to each of the inputs of the next layer
            (if there is a next layer).
            Has dimensions `(batchSize, depth, outputSize, outputSize)`.
        """
        (batchSize, depth, imageSize, _) = self.inputs.shape

        # Get size of output tensor
        outputSize = (imageSize - self._windowSize) // self._stride + 1

        # Create 4D array to store derivatives
        out = np.zeros((batchSize, depth, imageSize, imageSize), dtype=np.float32)

        # Move window vertically across image
        reshapedOutputDeltas = outputDeltas.reshape((batchSize, depth, outputSize, outputSize))

        for batchItem in range(batchSize):
            for d in range(depth):
                currentRow = 0
                while currentRow < imageSize - self._windowSize:
                    # Move window horizontally across image
                    currentCol = 0
                    while currentCol < imageSize - self._windowSize:
                        # Get submatrix
                        subMatrix = self.inputs[batchItem, d, currentRow:currentRow + self._windowSize, currentCol:currentCol + self._windowSize]

                        # Get indices of maximum values within each sub image
                        flattenedIndices = np.argmax(subMatrix.reshape(self._windowSize*self._windowSize))
                        out[batchItem, d, currentRow + (flattenedIndices // self._windowSize), currentCol + (flattenedIndices % self._windowSize)] = reshapedOutputDeltas[batchItem, d, (currentRow // self._stride), (currentCol // self._stride)]

                        # Move section being considered to right by stride length
                        currentCol += self._stride
                    
                    # Move section being considered down by stride length
                    currentRow += self._stride
        
        # Define derivatives of the cost function w.r.t the inputs
        self.inputDeltas = out