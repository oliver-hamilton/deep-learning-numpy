import numpy as np

class Layer:
    """Base class describing shared functionality of Dense, MaxPool and Convolutional layers."""
    def __init__(self):
        self.previousUpdate = 0

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
