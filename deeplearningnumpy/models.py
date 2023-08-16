import numpy as np
import os
from deeplearningnumpy.cost_functions import CategoricalCrossEntropy
from deeplearningnumpy.activations import ActivationSoftmax
import pickle

class NeuralNetwork:
    """Represents a neural network - this may be fully connected, convolutional, or some other type of network."""
    def __init__(self, name, layers):
        self.name = name
        self.layers = layers

    def forward(self, inputs):
        """Performs a forward pass of the network."""
        # Initialise inputs to starting values
        nextInputs = np.array(inputs)
        for i in range(len(self.layers)):
            self.layers[i].forward(nextInputs)
            # Initialise inputs of next layer as outputs from current layer
            nextInputs = self.layers[i].outputs

    def getOutputs(self):
        """Returns the outputs from the last layer of the network."""
        return self.layers[-1].outputs

    def saveWeights(self):
        """Saves the current weights for the network in a pickle file."""
        fileName = f"deeplearningnumpy/data/{self.name}.pkl"
        with open(fileName, 'wb') as fileHandle:
            pickle.dump(self.layers, fileHandle)

    def loadWeights(self):
        """Loads the weights for this network from the appropriate pickle file (if it exists).

        If the file does not exist, a FileNotFoundError is raised.
        """
        fileName = f"deeplearningnumpy/data/{self.name}.pkl"
        if not os.path.exists(fileName):
            raise FileNotFoundError(f"Weights for network '{self.name}' could not be loaded - have you trained your model?")

        else:
            with open(fileName, 'rb') as fileHandle:
                self.layers = pickle.load(fileHandle)

    def updateWeights(self, inputs, expectedOutputs, costFunction, learningRate):
        """Perform weight updates for all layers in the network."""
        self.errorGradientsSums = [None]*len(self.layers)

        #Iterate over each layer, starting with the output layer
        for i in range(len(self.layers)-1, -1, -1):
            currentLayer = self.layers[i]
            #Get the delta values for this layer
            if i == len(self.layers) - 1:
                currentLayer.getOutputDeltas(expectedOutputs, costFunction)
                currentLayer.getDeltas(currentLayer.outputDeltas)
            else:
                currentLayer.getDeltas(self.layers[i+1].inputDeltas)

            #Get error gradients
            self.errorGradientsSums[i] = (currentLayer.getErrorGradients())

        #Update all network weights and biases
        for i in range(len(self.layers)-1, -1, -1):
            currentLayer = self.layers[i]
            #Reshape error gradients so that they can be added to the weights
            #TODO: Add back in
            # currentLayer.updateWeights(self.errorGradientsSums[i], learningRate)

        #Make forward pass to get new network outputs that reflect updated weights
        self.forward(inputs)

    def train(self, X, y, costFunction, batchSize=None, epochs=10, learningRate = 0.1, checkGradients = False, testImages = [], testLabels = []):
        """Trains the network using stochastic gradient descent."""
        #If the cost function is CCE, the final activation must be softmax
        if not isinstance(costFunction, CategoricalCrossEntropy) and isinstance(self.layers[-1].activationFunction, ActivationSoftmax):
            raise Exception("The Softmax function can only be used with CCE loss")
        
        #Get number of training examples
        m = X.shape[0]

        if not batchSize:
            batchSize = m

        startIndex = 0

        for _ in range(epochs):
            #Iterate over all batches in the training set
            for _ in range(m // batchSize):
                try:
                    #Get the current batch data
                    currentBatch = X[startIndex:startIndex + batchSize]
                    currentLabels = y[startIndex:startIndex + batchSize]

                    #Perform a forward pass using the input data
                    self.forward(currentBatch)
                    self.updateWeights(currentBatch, currentLabels, costFunction, learningRate)
                    #Print cost for mini batch
                    print(f"Cost: {costFunction.getCost(self.layers[-1].outputs, currentLabels)}")

                    #Numerically estimate the gradients to verify backprop implementation
                    if checkGradients:
                        print(f"Error gradients (by backprop): {self.errorGradientsSums}")
                        print(f"Numerically estimated gradients: {self.getEstimatedGradients(currentBatch, currentLabels, costFunction)}")

                    # Evaluate accuracy on test set
                    '''
                    if testImages != [] and testLabels != []:
                        self.forward(testImages)
                        testOutputs = np.argmax(self.getOutputs(), -1)
                        matchingCount = 0
                        for i in range(testOutputs.size):
                            if testOutputs[i] == np.argmax(testLabels[i], -1):
                                matchingCount += 1

                        print("Accuracy on test set: {}%".format((matchingCount / testOutputs.size) * 100))
                    '''
                    startIndex += batchSize

                    if startIndex + batchSize > m:
                        startIndex = 0
                    
                except KeyboardInterrupt:
                    break

    def getEstimatedGradients(self, inputs, yReal, costFunction):
        """Numerically estimate the partial derivatives of the cost function w.r.t each weight"""

        #Assemble a list of computed gradient matrices
        estimatedGradients = [None] * len(self.layers)

        #Iterate over each parameter in each layer
        for i in range(len(self.layers)):
            flattenedWeights = self.layers[i].weights.reshape(-1)
            estimatedGradients[i] = np.zeros(flattenedWeights.size)
            for j in range(flattenedWeights.size):
                #A small value so that the gradient can be calculated between 2 points
                EPSILON = 1e-4
                #Copy the weights matrix so that it can be replaced once gradient checking completes
                weightsCopy = self.layers[i].weights.copy()
                flattenedWeights[j] += EPSILON

                self.forward(inputs)
                gradPlus = costFunction.getCost(self.layers[-1].outputs, yReal)
                flattenedWeights[j] -= 2 * EPSILON
                self.forward(inputs)
                gradMinus = costFunction.getCost(self.layers[-1].outputs, yReal)

                estimatedGradients[i][j] = (gradPlus - gradMinus) / (2 * EPSILON)

                #Re-instate original layer weights
                self.layers[i].weights = weightsCopy
                flattenedWeights = self.layers[i].weights.reshape(-1)

        return estimatedGradients