import numpy as np
from enum import Enum


class LayerType(Enum):
    HIDDEN = "hidden"
    OUTPUT = "output"


class Layer:

    def __init__(self, weights, bias, layer_type=LayerType.HIDDEN):
        self.weights = weights
        self.bias = bias
        self.layer_type = layer_type

    def forward(self, inputs):
        z = inputs @ self.weights.T + self.bias
        a = self.softmax(z) if self.layer_type == LayerType.OUTPUT else self.activation(z)
        return z, a

    def softmax(self, z):
        exps = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def activation(self, x):
        return np.maximum(0, x)

    def activation_derivative(self, z):
        return np.ones_like(z) if self.layer_type == LayerType.OUTPUT else (z > 0).astype(float)


class NeuralNetwork:
    def __init__(self, layer_sizes):
        layer_pairs = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        self.layers = [
            Layer(
                (
                    np.random.randn(out_size, in_size) * np.sqrt(2.0 / in_size)
                    if i < len(layer_pairs) - 1
                    else np.random.randn(out_size, in_size)
                ),
                np.zeros(out_size),
                layer_type=LayerType.OUTPUT if i == len(layer_pairs) - 1 else LayerType.HIDDEN,
            )
            for i, (in_size, out_size) in enumerate(layer_pairs)
        ]

    def train(self, X, y, batch_size, epochs, lr):
        n_samples = len(X)
        indices = np.arange(n_samples)

        for epoch in range(epochs):
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                a, z = self.forward(X_batch)
                dw, db = self.backprop(a, z, y_batch)

                self.step(dw, db, lr)

            loss = self.compute_loss(X, y)
            print(f"Epoch {epoch}: loss={loss:.4f}")

            np.random.shuffle(indices)

    def forward(self, x):
        Z, A = [], [x]
        for layer in self.layers:
            z, a = layer.forward(A[-1])
            Z.append(z)
            A.append(a)
        return A, Z

    def backprop(self, a, z, target):
        dw, db = [], []
        n_layers = len(self.layers)
        batch_size = target.shape[0]
        dc_da = a[-1] - target

        for i in range(1, n_layers + 1):
            layer = self.layers[-i]
            da_dz = layer.activation_derivative(z[-i])
            dz_da = layer.weights
            dz_dw = a[-i - 1]

            dc_dz = dc_da * da_dz

            dw.append(dc_dz.T @ dz_dw / batch_size)
            db.append(np.mean(dc_dz, axis=0))

            dc_da = dz_da.T @ dc_dz

        return dw[::-1], db[::-1]

    def compute_loss(self, X, y):
        total_loss = 0.0
        for x, target in zip(X, y):
            y_pred, _ = self.forward(x)
            y_pred = y_pred[-1]
            total_loss -= np.sum(target * np.log(y_pred + 1e-15))
        return total_loss / len(X)

    def step(self, dw, db, lr):
        for i, layer in enumerate(self.layers):
            layer.weights -= lr * dw[i]
            layer.bias -= lr * db[i]


def main():
    X = np.load("./data/data.npy")
    y = np.load("./data/target.npy")

    input_size = X.shape[-1]
    output_size = y.shape[-1]
    hidden_layers = [10, 10, 10]
    layer_sizes = [input_size] + hidden_layers + [output_size]

    model = NeuralNetwork(layer_sizes)

    model.train(X=X, y=y, batch_size=50, epochs=10000, lr=0.001)


main()
