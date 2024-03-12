
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This class is for multi layer perceptron. While training, firstly it does forward algorithm, then, it does backward algorithm. Finally it calculates loss.
class MultiLayerPerceptron:
    def __init__(self,
                 input_size:int = None,
                 hidden_layers = None,
                 output_size:int = None,
                 epochs:int = 30,
                 learning_rate:float = 0.01 ,
                 batch_size:int = 16,
                 lr_decay_rate:float = 0.1,
                 layer_activation_function:str = "sigmoid",
                 loss_function:str = "cross_entropy"
                 ):

        if input_size == None or output_size == None:
            raise ValueError("'input_size' and 'output_size' must be positive integer. 'output_size' must equals with the number of class !")

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        if hidden_layers != None:
            layer_sizes = [input_size] + hidden_layers + [output_size]
        else:
            layer_sizes = [input_size] + [output_size]

        # Set weights and biases randomly
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        self.biases = [np.random.randn(layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_decay_rate = lr_decay_rate

        if layer_activation_function.lower() in ["relu", "sigmoid", "tanh"]:
            self.layer_activation_function = layer_activation_function.lower()
        else:
            raise ValueError("Layer activation function must be one of the 'relu', 'sigmoid' or 'tanh' !")

        if loss_function.lower() in ["squared_error", "cross_entropy"]:
            self.loss_function = loss_function.lower()
        else:
            raise ValueError("Loss function must be one of the 'squared_error' or 'cross_entropy' !")

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu_derivative(self, x):
        return np.greater(x, 0).astype(int)

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh_derivative(self, x):
        return 1-self.tanh(x)**2 # sech^2{x}

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def squared_error(self, y_true, y_pred):
        return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))

    def cross_entropy(self, y_true, y_pred):
        epsilon = 1e-10  # To avoid the log(0)
        #y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))

    def forward(self, x):
        activations = [x]
        layer_results = []

        for i in range(len(self.weights)):
            layer_result = np.dot(activations[i], self.weights[i]) + self.biases[i]

            # If it is on the last layer, it is the output layer. It must use softmax.
            if i == len(self.weights) - 1:
                activation = self.softmax(layer_result)

            # It is based on the self.layer_actiovation_function
            else:
                if self.layer_activation_function == "relu":
                    activation = self.relu(layer_result)
                elif self.layer_activation_function == "sigmoid":
                    activation = self.sigmoid(layer_result)
                else:# self.layer_activation_function == "tanh":
                    activation = self.tanh(layer_result)

            activations.append(activation)
            layer_results.append(layer_result)

        return activations, layer_results


    def backward(self, x, y, activations):
        output_error = activations[-1] - y

        deltas = [output_error]
        for i in range(len(self.weights) - 1, 0, -1):
            error = np.dot(deltas[-1], self.weights[i].T)

            if self.layer_activation_function == "relu":
                delta = np.where(activations[i] <= 0, 0, error)

            elif self.layer_activation_function == "sigmoid":
                delta = error * self.sigmoid_derivative(activations[i])

            else: #self.activation == "tanh":
                delta = error * self.tanh_derivative(activations[i])

            deltas.append(delta)
        deltas = list(reversed(deltas))


        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= np.dot(activations[i].T, deltas[i]) * self.learning_rate
            self.biases[i] -= np.sum(deltas[i], axis=0) * self.learning_rate

    def calculate_accuracy(self, y_true, y_pred):
        predicted_labels = to_categorical(np.argmax(y_pred, axis=1), 6)
        return np.mean(np.all(y_true == predicted_labels, axis=1))

    def train(self, X, y):
        loss_list = []
        accuracy_list = []

        for epoch in range(self.epochs):
            for i in range(0, len(X), self.batch_size):
                x_batch = X[i : i + self.batch_size]
                y_batch = y[i : i + self.batch_size]

                # Feed Forward
                activations, layer_results = self.forward(x_batch)

                # Back Propogation
                self.backward(x_batch, y_batch, activations)

            y_pred = self.forward(X)[0][-1]

            # Loss for each epoch
            if self.loss_function == "cross_entropy":
                loss = self.cross_entropy(y, y_pred)
            else: # loss_function == "squared_error":
                loss = self.squared_error(y, y_pred)

            loss_list.append(loss)

            accuracy = self.calculate_accuracy(y, y_pred)
            accuracy_list.append(accuracy)

            # Reduce the learning rate
            self.learning_rate = self.learning_rate - (self.learning_rate * self.lr_decay_rate )

            print(f"Epoch {epoch+1}/{self.epochs} - Accuracy: {accuracy} - Loss: {loss} - Learning Rate: {self.learning_rate}")

        return accuracy_list, loss_list

    def predict(self, X):
        return self.forward(X)[0][-1]

# This creates the accuracy and loss plots for models.
def plot_accuracy_loss(data, generalName):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    fig.suptitle(f"Accuracy-Loss for {generalName}")

    for model in data.keys():
        model_name = data[model][0]

        ax1.plot(data[model][1], label=model_name)
        ax1.set_label(model_name)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("accuracy")
        ax1.set_title("accuracy-epoch")
        ax1.legend()

        ax2.plot(data[model][2], label=model_name)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("loss")
        ax2.set_title("loss-epoch")
        ax2.legend()

    plt.legend( loc="lower left")
    plt.show()


