import numpy as np

class MSE:
    """Mean Squared Error loss function."""
    def getLoss(self, predictions, expected):
        return 0.5 * np.sum(predictions - expected) ** 2

class CCEWithSoftmax:
    """Categorical Cross Entropy Loss Function.
    As this incoroporates the derivative of softmax, this
    loss can only be used with a softmax activation function
    in the output layer."""
    def getLoss(self, predictions, expected):
        # Prevent overflow
        beta = 0.000000001
        return -1 * np.sum(np.einsum('ij,ij', expected, np.log(np.clip(predictions, beta, 1 - beta))))
