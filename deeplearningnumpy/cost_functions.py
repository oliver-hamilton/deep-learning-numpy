import numpy as np

class MSE:
    def getCost(self, yPred, yReal):
        """Get the cost when comparing the predicting outputs with the target outputs"""
        # Number of examples
        m = yReal.shape[0]

        # Return cost
        return (1/(2*m)) * np.sum((yPred - yReal)**2)

    def getDerivative(self, yPred, yReal):
        """Get the derivative of the cost w.r.t the output values"""
        # Number of examples
        m = yReal.shape[0]

        # Return derivative of cost w.r.t output values
        return (1/m) * (yPred - yReal)


class BinaryCrossEntropy:
    def getCost(self, yPred, yReal):
        # Number of examples
        m = yReal.shape[0]

        # Return cost
        return (-1/m) * np.sum(yReal * np.log(yPred) + (1 - yReal) * np.log(1 - yPred))

    def getDerivative(self, yPred, yReal):
        # Number of examples
        m = yReal.shape[0]

        # Return derivative of cost w.r.t output values
        return (1/m) * ((yPred - yReal) / (yPred * (1 - yPred)))


class CategoricalCrossEntropy:
    def getCost(self, yPred, yReal):
        #Number of examples
        m = yReal.shape[0]
        return (-1/m) * np.sum(yReal * np.log(yPred))

    #Builds in the softmax derivative
    def getDerivative(self, yPred, yReal):
        #Number of examples
        m = yReal.shape[0]
        return (1/m) * (yPred - yReal)