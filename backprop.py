import numpy as np


class Layer:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        z = np.dot(self.weights, inputs) + self.bias
        return self.activation(z)

    def activation(self, x):
        return np.maximum(0, x)


class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = [
            Layer(
                np.random.randn(layer_sizes[i + 1], layer_sizes[i]),
                np.zeros(layer_sizes[i + 1]),
            )
            for i in range(len(layer_sizes) - 1)
        ]

    def train(self, X_all, y_all, batch_size, num_epochs, learning_rate):
        N = len(X_all)
        indices = np.arange(N)

        for epoch in range(num_epochs):
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_indices = indices[start:end]

                X_batch = X_all[batch_indices]
                y_batch = y_all[batch_indices]
                dw_batch, db_batch = [], []

                for x, y in zip(X_batch, y_batch):
                    pred, activations = self.forward(x)
                    dw, db = self.backward(y, pred, activations)
                    dw_batch.append(dw)
                    db_batch.append(db)

                self.step(dw_batch, db_batch, learning_rate)

            np.random.shuffle(indices)

    def forward(self, x):
        activations = []
        for layer in self.layers:
            x = layer.forward(x)
            activations.append(x)
        return x, activations

    def backward(self, y_true, y_pred, activations):
        pass

    def step(self, dw_batch, db_batch, learning_rate):
        dw = np.mean(np.stack(dw_batch, axis=0), axis=0)
        db = np.mean(np.stack(db_batch, axis=0), axis=0)

        for i, layer in enumerate(self.layers):
            layer.weights -= learning_rate * dw[i]
            layer.bias -= learning_rate * db[i]


def main():
    data = np.load("./data/data.npy")
    target = np.load("./data/target.npy")

    input_size = data.shape[-1]
    output_size = target.shape[-1]
    hidden_layers = [10, 10, 10]

    nn_shape = [input_size] + hidden_layers + [output_size]

    nn = NeuralNetwork(nn_shape)
