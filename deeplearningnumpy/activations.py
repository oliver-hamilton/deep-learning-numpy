import numpy as np

class ActivationReLU:
    """Rectified Linear Unit activation function."""
    def forward(self, inputs):
        """Propagates the inputs through the ReLU activation.
        
        Parameters
        ----------
        inputs : np.ndarray
            The inputs to apply the activation function to.
            Negative values become 0, and positive values
            remain unchanged.

        Returns
        -------
        np.ndarray
            The output of the activation function. 
            The function is applied elementwise, and so the
            output has the same dimensions as the input.
        """
        return np.maximum(0, inputs)

    def getDerivative(self, inputs):
        """Gets the derivative of the activation function w.r.t the inputs.
        
        Parameters
        ----------
        inputs : np.ndarray
            The inputs to take the derivative with respect to.

        Returns
        -------
        np.ndarray
            The resulting derivatives. Since the activation is applied elementwise,
            the result has the same dimensions as the input. Positive values are
            replaced by 1, and negative values by 0.
        """
        return np.where(inputs > 0, 1, 0)


class ActivationLeakyReLU:
    """Leaky ReLU activation function.

    Attributes
    ----------
    slope : int
        The derivative for negative inputs. Setting `slope` to 0
        gives the same behaviour as ReLU activation. It is recommended
        that slope is set at a value between 0 and 1 (inclusive) for stable
        training, for example ``slope = 0.2``.
    """
    def __init__(self, slope):
        self._slope = slope

    def forward(self, inputs):
        """Propagates the inputs through the Leaky ReLU activation.
        
        Parameters
        ----------
        inputs : np.ndarray
            The inputs to apply the activation function to.
            Negative values ``x`` become ``slope * x``, and positive values
            remain unchanged. Thus, for positive values of `slope`, negative
            values remain negative and positive values remain positive.

        Returns
        -------
        np.ndarray
            The output of the activation function. 
            The function is applied elementwise, and so the
            output has the same dimensions as the input.
        """
        return np.maximum(self._slope * inputs, inputs)

    def getDerivative(self, inputs):
        """Gets the derivative of the activation function w.r.t the inputs.
        
        Parameters
        ----------
        inputs : np.ndarray
            The inputs to take the derivative with respect to.

        Returns
        -------
        np.ndarray
            The resulting derivatives. Since the activation is applied elementwise,
            the result has the same dimensions as the input. Positive values are
            replaced by 1, and negative values by the value of `slope`.
        """
        return np.where(inputs > 0, 1, self._slope)

class ActivationLinear:
    """Linear activation function."""
    def forward(self, inputs):
        """Propagates the inputs through the Linear activation.
        
        Parameters
        ----------
        inputs : np.ndarray
            The inputs to apply the activation function to.
            All values remain unchanged.

        Returns
        -------
        np.ndarray
            The output of the activation function. 
            The function is applied elementwise, and so the
            output has the same dimensions as the input.
        """
        return inputs

    def getDerivative(self, inputs):
        """Gets the derivative of the activation function w.r.t the inputs.
        
        Parameters
        ----------
        inputs : np.ndarray
            The inputs to take the derivative with respect to.

        Returns
        -------
        np.ndarray
            The resulting derivatives. Since the activation is applied elementwise,
            the result has the same dimensions as the input. All values are replaced
            by 1.
        """
        return np.ones(inputs.shape)

class ActivationLogistic:
    """Logistic activation function."""
    def forward(self, inputs):
        """Propagates the inputs through the Logistic activation.
        
        Parameters
        ----------
        inputs : np.ndarray
            The inputs to apply the activation function to.
            Each value ``x`` is replaced by ``1 / (1 + e^-x)``

        Returns
        -------
        np.ndarray
            The output of the activation function. 
            The function is applied elementwise, and so the
            output has the same dimensions as the input.
        """
        return 1 / (1 + np.exp(-1 * (inputs)))

    def getDerivative(self, inputs):
        """Gets the derivative of the activation function w.r.t the inputs.
        
        Parameters
        ----------
        inputs : np.ndarray
            The inputs to take the derivative with respect to.

        Returns
        -------
        np.ndarray
            The resulting derivatives. Since the activation is applied elementwise,
            the result has the same dimensions as the input. All values ``x`` are replaced
            by ``logistic(x) * (1 - logistic(x))``.
        """
        logistic = self.forward(inputs)
        return logistic * (1 - logistic)
    
class ActivationTanh:
    """Tanh activation function."""
    def forward(self, inputs):
        """Propagates the inputs through the Tanh activation.
        
        Parameters
        ----------
        inputs : np.ndarray
            The inputs to apply the activation function to.
            Each value ``x`` is replaced by ``(e^x - e^-x) / (e^x + e^-x)``

        Returns
        -------
        np.ndarray
            The output of the activation function. 
            The function is applied elementwise, and so the
            output has the same dimensions as the input.
        """
        return (np.exp(inputs) - np.exp(-1 * inputs)) / (np.exp(inputs) + np.exp(-1 * inputs))

    def getDerivative(self, inputs):
        """Gets the derivative of the activation function w.r.t the inputs.
        
        Parameters
        ----------
        inputs : np.ndarray
            The inputs to take the derivative with respect to.

        Returns
        -------
        np.ndarray
            The resulting derivatives. Since the activation is applied elementwise,
            the result has the same dimensions as the input. All values ``x`` are replaced
            by ``sech^2(x) = 1 - tanh^2(x)``.
        """
        tanh = self.forward(inputs)
        return 1 - tanh**2

class ActivationSoftmax:
    """Softmax activation function.
    
    The derivative is built into the cross-entropy loss, so we just
    assign a derivative of 1. The effect of this is that the softmax
    activation function can only be used with cross-entropy loss.
    """
    def forward(self, inputs):
        """Propagates the inputs through the Softmax activation.
        
        Parameters
        ----------
        inputs : np.ndarray
            The inputs to apply the activation function to.
            Each value is divided by the sum of the values over
            that last dimension.

        Returns
        -------
        np.ndarray
            The output of the activation function. 
            This is a vector function that operates over the last dimension,
            and the output has the same dimensions as the input.
        """
        # Subtract max value in each batch to prevent overflow
        # We can do this since softmax(x + c) = softmax(x)
        expValues = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        # Normalise exponential values so they sum to 1
        probabilities = expValues / np.sum(expValues, axis=-1, keepdims=True)
        return probabilities

    def getDerivative(self, inputs):
        """Gets the derivative of the activation function w.r.t the inputs.
        
        Parameters
        ----------
        inputs : np.ndarray
            The inputs to take the derivative with respect to.

        Returns
        -------
        np.ndarray
            The resulting derivatives. Since the activation is applied elementwise,
            the result has the same dimensions as the input. All values are replaced
            by 1 since the true derivative is accounted for by the categorical cross-
            entropy loss.
        """
        return np.ones(inputs.shape)

