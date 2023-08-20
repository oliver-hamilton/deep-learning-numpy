import numpy as np

class MSE:
    """Mean squared error cost function."""
    def getCost(self, yPred, yReal):
        """Returns the cost when comparing the predicting outputs with the target outputs.
        
        Parameters
        ----------
        yPred : numpy.ndarray
            The outputs generated by the neural network, where the first dimension should be
            the number of outputs.
        yReal : numpy.ndarray
            The desired outputs of the neural network, with the same dimensions as `yPred`.

        Returns
        -------
        float
            The total cost, averaged over all of the outputs generated by the neural network.
        """
        # Number of examples
        m = yReal.shape[0]

        return (1/(2*m)) * np.sum((yPred - yReal)**2)

    def getDerivative(self, yPred, yReal):
        """Returns the derivative of the cost function w.r.t the output values.
        
        Parameters
        ----------
        yPred : numpy.ndarray
            The outputs generated by the neural network, where the first dimension should be
            the number of outputs.
        yReal : numpy.ndarray
            The desired outputs of the neural network, with the same dimensions as `yPred`.
        Returns
        -------
        numpy.ndarray
            The derivative of the cost function with respect to each of the neural network outputs.
        """
        # Number of examples
        m = yReal.shape[0]

        return (1/m) * (yPred - yReal)


class BinaryCrossEntropy:
    """Binary cross entropy cost function."""
    def getCost(self, yPred, yReal):
        """Returns the cost when comparing the predicting outputs with the target outputs.
        
        Parameters
        ----------
        yPred : numpy.ndarray
            The outputs generated by the neural network, where the first dimension should be
            the number of outputs.
        yReal : numpy.ndarray
            The desired outputs of the neural network, with the same dimensions as `yPred`.

        Returns
        -------
        float
            The total cost, averaged over all of the outputs generated by the neural network.
        """
        # Number of examples
        m = yReal.shape[0]

        return (-1/m) * np.sum(yReal * np.log(yPred) + (1 - yReal) * np.log(1 - yPred))

    def getDerivative(self, yPred, yReal):
        """Returns the derivative of the cost function w.r.t the output values.
        
        Parameters
        ----------
        yPred : numpy.ndarray
            The outputs generated by the neural network, where the first dimension should be
            the number of outputs.
        yReal : numpy.ndarray
            The desired outputs of the neural network, with the same dimensions as `yPred`.
        Returns
        -------
        numpy.ndarray
            The derivative of the cost function with respect to each of the neural network outputs.
        """
        # Number of examples
        m = yReal.shape[0]

        return (1/m) * ((yPred - yReal) / (yPred * (1 - yPred)))


class CategoricalCrossEntropy:
    """Categorical cross entropy cost function."""
    def getCost(self, yPred, yReal):
        """Returns the cost when comparing the predicting outputs with the target outputs.
        
        Parameters
        ----------
        yPred : numpy.ndarray
            The outputs generated by the neural network, where the first dimension should be
            the number of outputs.
        yReal : numpy.ndarray
            The desired outputs of the neural network, with the same dimensions as `yPred`.

        Returns
        -------
        float
            The total cost, averaged over all of the outputs generated by the neural network.
        """
        # Number of examples
        m = yReal.shape[0]

        return (-1/m) * np.sum(yReal * np.log(yPred))

    # Incorporates in the softmax derivative
    def getDerivative(self, yPred, yReal):
        """Returns the derivative of the cost function w.r.t the output values before the
        softmax activation function is applied. Thus this incorporates the derivative of
        softmax with that of the categorical cross entropy cost function.
        
        Parameters
        ----------
        yPred : numpy.ndarray
            The outputs generated by the neural network, where the first dimension should be
            the number of outputs.
        yReal : numpy.ndarray
            The desired outputs of the neural network, with the same dimensions as `yPred`.
        Returns
        -------
        numpy.ndarray
            The derivative of the cost function with respect to each of the neural network outputs
            (before softmax activation).
        """
        # Number of examples
        m = yReal.shape[0]

        return (1/m) * (yPred - yReal)