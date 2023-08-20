import numpy as np
import os
from deeplearningnumpy.cost_functions import CategoricalCrossEntropy
from deeplearningnumpy.activations import ActivationSoftmax
import pickle

class NeuralNetwork:
    """Represents a neural network, which may be fully connected or convolutional.
    
    Attributes
    ----------
    name : str
        A string identifier for the neural network e.g. for naming weights files.
    layers : array_like
        A list of layers that comprise the neural network. Inputs are propagated through this set of layers
        sequentially from start to finish when the `forward` method is called.
    """
    def __init__(self, name, layers):
        self._name = name
        self._layers = layers

    def forward(self, inputs):
        """Performs a forward pass of the network.
        
        Parameters
        ----------
        inputs : numpy.ndarray
            The inputs to the network, which should match the dimensions required for
            the input to the `forward` method of the first layer in the network.
        """
        # Initialise inputs to starting values
        nextInputs = np.array(inputs)
        for i in range(len(self._layers)):
            self._layers[i].forward(nextInputs)
            # Initialise inputs of next layer as outputs from current layer
            nextInputs = self._layers[i].outputs

    def getOutputs(self):
        """Returns the outputs from the last layer of the network produced by the
        most recent call to `forward`.
        
        Returns
        -------
        numpy.ndarray
            The outputs of the last layer, whose dimensions match those of the
            output of the `forward` method in the last layer of the network.
        """
        return self._layers[-1].outputs

    def saveWeights(self):
        """Saves the current weights for the network in a pickle file.

        The file is named as ``{network_name}.pkl``, and saved in the
        ``deeplearningnumpy/data/`` directory. 
        """
        fileName = f"deeplearningnumpy/data/{self._name}.pkl"
        with open(fileName, 'wb') as fileHandle:
            pickle.dump(self._layers, fileHandle)

    def loadWeights(self):
        """Loads the weights for this network from the appropriate pickle file (if it exists).

        The file that is read is ``deeplearningnumpy/data/{network_name}.pkl``.

        Raises
        ------
        FileNotFoundError
            If the file for the network does not exist.
        """
        fileName = f"deeplearningnumpy/data/{self._name}.pkl"
        if not os.path.exists(fileName):
            raise FileNotFoundError(f"Weights for network '{self._name}' could not be loaded - have you trained your model?")

        else:
            with open(fileName, 'rb') as fileHandle:
                self._layers = pickle.load(fileHandle)

    def updateWeights(self, inputs, expectedOutputs, costFunction, learningRate):
        """Perform weight updates for all layers in the network.
        
        A forward pass is made prior to updating weights to ensure that the layer outputs
        are those produced by inputting `inputs`. The weights are then updated to
        bring the outputs closer to `expectedOutputs`, according to `costFunction` and 
        controlled by the `learningRate`.

        Parameters
        ----------
        inputs : numpy.ndarray
            The inputs to the network, which should match the dimensions required for
            the input to the `forward` method of the first layer in the network.
        expectedOutputs : numpy.ndarray
            The desired values for the network's outputs. This parameter should have the same dimensions as that
            which would be produced by the network's `forward` method.
        costFunction : CostFunction
            A cost function whose derivative is calculated with respect to the outputs of the network
            (i.e. the outputs of the last layer).
        learningRate : float
            Controls the amount by which the weights and biases of each layer are changed on each step. 
            Both the weights and biases of each layer are updated at the same rate.

        Notes
        -----
        ``CostFunction`` is any of ``MSE``, ``BinaryCrossEntropy``, or ``CategoricalCrossEntropy``.
        """

        self.forward(inputs)
        self.errorGradientsSums = [None]*len(self._layers)

        #Iterate over each layer, starting with the output layer
        for i in range(len(self._layers)-1, -1, -1):
            currentLayer = self._layers[i]
            #Get the delta values for this layer
            if i == len(self._layers) - 1:
                currentLayer.getOutputDeltas(expectedOutputs, costFunction)
                currentLayer.getDeltas(currentLayer.outputDeltas)
            else:
                currentLayer.getDeltas(self._layers[i+1].inputDeltas)

            #Get error gradients
            self.errorGradientsSums[i] = (currentLayer.getErrorGradients())

        #Update all network weights and biases
        for i in range(len(self._layers)-1, -1, -1):
            currentLayer = self._layers[i]
            #Reshape error gradients so that they can be added to the weights
            currentLayer.updateWeights(self.errorGradientsSums[i], learningRate)


    def train(self, X, y, costFunction, batchSize=None, epochs=10, learningRate = 0.1, checkGradients = False, testImages = [], testLabels = []):
        """Trains the network using stochastic gradient descent.
        
        Parameters
        ----------
        X : numpy.ndarray
            The training inputs. The first dimension equals the total number of input samples (not batches).
        Y : numpy.ndarray
            The desired training outputs. The first dimension equals the total number of input samples (not batches).
        costFunction : CostFunction
            A cost function to evaluate the performance of the neural network.
        batchSize : int
            The number of input samples to include in each batch.
        epochs : int, optional
            The number of times to iterate through the full training set. Defaults to 10.
        learningRate : float, optional
            Controls the rate of learning. A small value of `learningRate` will lead to slow training,
            but a large value will cause training instability. Defaults to 0.1.
        checkGradients : bool, optional
            Whether to perform gradient checking. This was used for verifying the gradients computed through
            backpropagation were correct, and is therefore unlikely to be useful to a user. Defaults to False.
        testImages : np.ndarray, optional
            Test data to evaluate the accuracy of the model on a test set during the training process.
            Defaults to [].
        testLabels : np.ndarray, optional
            Corresponding labels for `testImages` to evaluate the accuracy of the model on a test set during
            the training process. Defaults to [].

        Notes
        -----
        ``CostFunction`` is any of ``MSE``, ``BinaryCrossEntropy``, or ``CategoricalCrossEntropy``.
        """
        #If the cost function is CCE, the final activation must be softmax
        if not isinstance(costFunction, CategoricalCrossEntropy) and isinstance(self._layers[-1].activationFunction, ActivationSoftmax):
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
                    print(f"Cost: {costFunction.getCost(self._layers[-1].outputs, currentLabels)}")

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
        """Numerically estimates the partial derivatives of the cost function w.r.t each weight.
        
        This is used in gradient checking as an alternative way of computing gradients other
        than backpropagation, but is unlikely to be useful to a user.
        
        Parameters
        ----------
        inputs : numpy.ndarray
            The inputs used for evaluating cost, and hence estimating derivatives.
        yReal : numpy.ndarray
            The correct outputs for the given inputs, used for evaluating cost.
        costFunction : CostFunction
            The cost function used for taking derivatives.

        Returns
        -------
        array-like
            The estimated gradients with respect to the weights and biases of each layer.
        
        Notes
        -----
        ``CostFunction`` is any of ``MSE``, ``BinaryCrossEntropy``, or ``CategoricalCrossEntropy``.
        """

        #Assemble a list of computed gradient matrices
        estimatedGradients = [None] * len(self._layers)

        # Iterate over each bias parameter in each layer
        for i in range(len(self._layers)):
            flattenedBiases = self._layers[i].biases.reshape(-1)
            estimatedGradients[i] = np.zeros(flattenedBiases.size)
            for j in range(flattenedBiases.size):
                #A small value so that the gradient can be calculated between 2 points
                EPSILON = 1e-3
                #Copy the biases matrix so that it can be replaced once gradient checking completes
                biasesCopy = self._layers[i].biases.copy()
                flattenedBiases[j] += EPSILON

                self.forward(inputs)
                gradPlus = costFunction.getCost(self._layers[-1].outputs, yReal)
                flattenedBiases[j] -= 2 * EPSILON
                self.forward(inputs)
                gradMinus = costFunction.getCost(self._layers[-1].outputs, yReal)

                estimatedGradients[i][j] = (gradPlus - gradMinus) / (2 * EPSILON)

                #Re-instate original layer biases
                self._layers[i].biases = biasesCopy
                flattenedBiases = self._layers[i].biases.reshape(-1)

        #Iterate over each weight parameter in each layer
        for i in range(len(self._layers)):
            flattenedWeights = self._layers[i].weights.reshape(-1)
            estimatedGradients[i] = np.zeros(flattenedWeights.size)
            for j in range(flattenedWeights.size):
                #A small value so that the gradient can be calculated between 2 points
                EPSILON = 1e-4
                #Copy the weights matrix so that it can be replaced once gradient checking completes
                weightsCopy = self._layers[i].weights.copy()
                flattenedWeights[j] += EPSILON

                self.forward(inputs)
                gradPlus = costFunction.getCost(self._layers[-1].outputs, yReal)
                flattenedWeights[j] -= 2 * EPSILON
                self.forward(inputs)
                gradMinus = costFunction.getCost(self._layers[-1].outputs, yReal)

                estimatedGradients[i][j] = (gradPlus - gradMinus) / (2 * EPSILON)

                #Re-instate original layer weights
                self._layers[i].weights = weightsCopy
                flattenedWeights = self._layers[i].weights.reshape(-1)

        return estimatedGradients