import numpy as np

class ActivationReLU:
    """Rectified Linear Unit activation function.
    Derivative is 1 if x > 0, and 0 otherwise.
    """
    def forward(self, inputs):
        """Propagates the inputs through the ReLU activation."""
        return np.maximum(0, inputs)

    def getDerivative(self, inputs):
        """Gets the derivative of the activation function w.r.t the inputs."""
        return np.where(inputs > 0, 1, 0)


class ActivationLeakyReLU:
    """Leaky ReLU activation function.
    Like ReLU, but the derivative is some small negative value, determined by
    slope, for x <= 0.
    """
    def __init__(self, slope):
        self.slope = slope

    def forward(self, inputs):
        """Propagates the inputs through the Leaky ReLU activation."""
        return np.maximum(self.slope * inputs, inputs)

    def getDerivative(self, inputs):
        """Gets the derivative of the activation function w.r.t the inputs."""
        return np.where(inputs > 0, 1, self.slope)

class ActivationLinear:
    """Linear activation function.
    The derivative is 1 for all inputs.
    """
    def forward(self, inputs):
        """Propagates the inputs through the Linear activation, which has no effect"""
        return inputs

    def getDerivative(self, inputs):
        """Gets the derivative of the activation function w.r.t the inputs."""
        return np.ones(inputs.shape)

class ActivationSoftmax:
    """Softmax activation function.
    The derivative is built into the cross-entropy loss, so we just
    assign a derivative of 1. The effect of this is that the softmax
    activation function can only be used with cross-entropy loss.
    """
    def forward(self, inputs):
        """Propagates the inputs through the Softmax activation."""
        #Subtract max value in each batch to prevent overflow
        expValues = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        #Normalise output values so values sum to 1
        probabilities = expValues / np.sum(expValues, axis=-1, keepdims=True)
        return probabilities

    def getDerivative(self, inputs):
        """Gets the derivative of the activation function w.r.t the inputs."""
        return np.ones(inputs.shape)

class ActivationLogistic:
    """Logistic activation function.
    The derivative is logistic(x) * (1 - logistic(x))
    """
    def forward(self, inputs):
        """Propagates the inputs through the Logistic activation."""
        return 1 / (1 + np.exp(-1 * (inputs)))

    def getDerivative(self, inputs):
        """Gets the derivative of the activation function w.r.t the inputs."""
        logistic = self.forward(inputs)
        return logistic * (1 - logistic)