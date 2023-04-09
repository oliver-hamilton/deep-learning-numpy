import numpy as np
from deeplearningnumpy.layers.base import Layer

WEIGHT_MULTIPLIER = 0.10

class DenseLayer(Layer):
    def __init__(self, nInputs, nNeurons, activationFunction):
        super().__init__()
        # Weights are multiplied by constant for normalisation
        self.weights = WEIGHT_MULTIPLIER * (np.random.randn(nInputs, nNeurons))
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