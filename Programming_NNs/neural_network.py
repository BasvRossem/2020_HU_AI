"""A neural network."""

import numpy as np

from neuron import Neuron


class NeuralNetwork:
    """
    A neural network class.
    """

    def __init__(self, layer_layout, learning_rate=1):
        """
        Creates a neural network usinig the input layout.
        """
        self.learning_rate = learning_rate
        self.input_size = layer_layout[0]
        self.layers = []

        for i, neuron_count in enumerate(layer_layout[1:]):
            layer = []
            for _j in range(neuron_count):
                weight_size = self.input_size
                if i > 0:
                    weight_size = len(self.layers[i-1])
                weights = np.random.rand(1, weight_size)
                bias = np.random.random()
                neuron = Neuron(np.array([]), weights[0], bias)

                layer.append(neuron)

            self.layers.append(layer)

        print(self.layers)

    def train(self, values, label):
        """
        Do a single training step. One setp is one
        feed forward and one backward propegation step.
        """
        assert len(values) == self.input_size, "Input is not the same size as NN inputs"
        assert len(label) == len(self.layers[-1]), "Label not the same size as NN outputs"

        self.feed_forward(values)
        self.back_propagation(label)

    def feed_forward(self, neuron_inputs):
        """
        Calculates all new values of the neurons in every
        layer, and sets those values for the neurons.
        """
        for layer in self.layers:
            new_neuron_inputs = []
            for neuron in layer:
                neuron.inputs = neuron_inputs
                new_neuron_inputs.append(neuron.activation())
            neuron_inputs = new_neuron_inputs

    def back_propagation(self, label):
        """
        Calculate and set the new weighhts for the neurons
        in every layer.
        """
        all_delta_weights = []

        # Loop through the neural network backwards
        for i, layer in enumerate(reversed(self.layers)):
            current_weights = []
            for j, neuron in enumerate(layer):
                # Calculating a new delta
                if i == 0:
                    neuron.calculate_delta(label[j] - neuron.activation())
                else:
                    delta_sum = sum(np.array(all_delta_weights).T[j])
                    neuron.calculate_delta(delta_sum)

                # Calculate the new weights
                self.calculate_weights(neuron)

                neuron_weights = [neuron.delta * weight for weight in neuron.weights]
                current_weights.append(neuron_weights)

            all_delta_weights = current_weights

    def calculate_weights(self, neuron):
        """
        Calculates the new weight for a neuron.
        """
        new_weights = []
        for i, weight in enumerate(neuron.weights):
            new_weights.append(weight + (self.learning_rate * neuron.delta * neuron.inputs[i]))
        neuron.weights = np.array(new_weights)
        neuron.bias = neuron.bias + (self.learning_rate * neuron.delta)

    def predict(self, data):
        """
        Make a prediction and return the output.
        """
        self.feed_forward(data)
        return self.get_output()

    def get_output(self):
        """
        Returns the lasty values calculated by the neural network.
        """
        return [neuron.activation() for neuron in self.layers[-1]]
