import numpy as np
from deeplearningnumpy.layers import Layer
from deeplearningnumpy.layers import DenseLayer

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
