'''
Expectation Backpropagation with binary weights

Adapted from:
https://github.com/ExpectationBackpropagation/EBP_Matlab_Code/blob/master/RunMe.m
'''

import numpy as np
from scipy.stats import norm

class ExpectationBackpropagationNetworkBinary:

    def __init__(self, data, network_size, epochs=20, batch_size=10, sigma_w=1.):
        
        self.epochs = epochs
        self.sigma_w = sigma_w
        self.eta = 1.
        self.batch_size = batch_size
        self.network_size = network_size
        self.num_layers = len(network_size) - 1

        self.train_data = data['train']
        self.valid_data = data['valid']
        self.test_data = data['test']

        self.mean_u_history = {}
        self.var_u_history = {}
        self.mean_v_prev_history = {}

        self.weights = {}
        self.tanh_weights = {}
        self.biases = {}
        for l in range(0, self.num_layers):
            self.weights[l] = (np.random.rand(self.network_size[l + 1], self.network_size[l]) - 0.5) * \
                np.sqrt(self.sigma_w * 12 / self.network_size[l])
            self.tanh_weights[l] = np.tanh(self.weights[l])
            self.biases[l] = np.zeros((self.network_size[l + 1], 1))

    def forward_pass(self, x, y, batch_size=10):

        shuffle = np.random.permutation(len(y))
        x, y = x[shuffle], y[shuffle]

        prob_pred = np.zeros((y.shape[0], y.shape[1]))
        deterministic_pred = np.zeros((y.shape[0], y.shape[1]))

        for i in range(0, len(y), batch_size):

            x_mb, y_mb = np.transpose(x[i:i + batch_size]), np.transpose(y[i:i + batch_size])

            # Forward pass through network
            
            mean_v = x_mb
            tanh = self.tanh_weights[0]
            bias = self.biases[0]

            mean_u = (np.matmul(tanh, mean_v) + bias) / np.sqrt(self.network_size[0] + 1)
            var_u = (np.matmul((1 - tanh ** 2), mean_v ** 2) + 1) / (self.network_size[0] + 1)
            prob_v = norm.cdf(mean_u / np.sqrt(var_u), 0, 1)

            mean_v = 2 * prob_v - 1
            var_v = 4 * (prob_v - prob_v ** 2)

            for l in range(1, self.num_layers):

                bias = self.biases[l]
                tanh = self.tanh_weights[l]

                mean_u = (np.matmul(tanh, mean_v) + bias) / np.sqrt(self.network_size[l] + 1)
                var_u = (np.sum(var_v, 0) + np.matmul((1 - tanh ** 2), (1 - var_v)) + 1) / (self.network_size[l] + 1)
                prob_v = norm.cdf(mean_u / np.sqrt(var_u), 0, 1)

                mean_v = 2 * prob_v - 1
                var_v = 4 * (prob_v - prob_v ** 2)

            # Compute probabilistic prediction
            prob_pred[i:i + batch_size, :] = np.transpose(mean_v)

            # Compute deterministic prediction
            v = np.copy(x_mb)
            for l in range(self.num_layers - 1):
                h = self.weights[l]
                bias = self.biases[l]
                v = np.sign(np.matmul(np.sign(h), v) + bias)

            h = self.weights[self.num_layers - 1]
            bias = self.biases[self.num_layers - 1]
            v = np.matmul(np.sign(h), v) + bias
            deterministic_pred[i:i + batch_size, :] = np.transpose(v)

        prob_accuracy = self.compute_accuracy(y, prob_pred)
        print('Probabilistic Accuracy: {0}'.format(prob_accuracy))
        determ_accuracy = self.compute_accuracy(y, deterministic_pred)
        print('Deterministic Accuracy: {0}\n'.format(determ_accuracy))

        return prob_accuracy, determ_accuracy

    def compute_accuracy(self, labels, predictions):

        assert len(labels) == len(predictions)

        acc = np.zeros(len(labels))
        for i in range(len(labels)):
            true_class = np.argmax(labels[i, :])
            pred_class = np.argmax(predictions[i, :])
            acc[i] = true_class == pred_class

        return np.mean(acc)

    def train(self):

        x_train, y_train = self.train_data['x'], self.train_data['y']

        for e in range(1, self.epochs + 1):

            shuffle = np.random.permutation(len(y_train))
            x_train, y_train = x_train[shuffle], y_train[shuffle]

            prob_acc, determ_acc = self.forward_pass(self.valid_data['x'], self.valid_data['y'])

            for i in range(0, len(y_train), self.batch_size):

                x_train_mb, y_train_mb = np.transpose(x_train[i:i + self.batch_size]), np.transpose(y_train[i:i + self.batch_size])

                # Forward pass through network
                
                mean_v = np.copy(x_train_mb)
                tanh = self.tanh_weights[0]
                bias = self.biases[0]

                mean_u = (np.matmul(tanh, mean_v) + bias) / np.sqrt(self.network_size[0] + 1)
                var_u = (np.matmul((1 - np.square(tanh)), np.square(mean_v)) + 1) / (self.network_size[0] + 1)
                prob_v = norm.cdf(mean_u / np.sqrt(var_u), 0, 1)

                self.mean_u_history[0] = mean_u
                self.var_u_history[0] = var_u
                self.mean_v_prev_history[0] = mean_v

                mean_v = 2 * prob_v - 1
                var_v = 4 * (prob_v - np.square(prob_v))
                self.mean_v_prev_history[1] = mean_v

                for l in range(1, self.num_layers):

                    bias = self.biases[l]
                    tanh = self.tanh_weights[l]

                    mean_u = (np.matmul(tanh, mean_v) + bias) / np.sqrt(self.network_size[l] + 1)
                    var_u = (np.sum(var_v, 0) + np.matmul((1 - np.square(tanh)), (1 - var_v)) + 1) / (self.network_size[l] + 1)
                    prob_v = norm.cdf(mean_u / np.sqrt(var_u), 0, 1)

                    self.mean_u_history[l] = mean_u
                    self.var_u_history[l] = var_u

                    mean_v = 2 * prob_v - 1
                    var_v = 4 * (prob_v - np.square(prob_v))
                    self.mean_v_prev_history[l + 1] = mean_v

                # Backward pass through network

                delta = None

                for l in range(self.num_layers - 1, -1, -1):
                    mean_v_prev = self.mean_v_prev_history[l]
                    mean_u = self.mean_u_history[l]
                    var_u = self.var_u_history[l]
                    bias = self.biases[l]
                    h = self.weights[l]
                    tanh = self.tanh_weights[l]

                    if l == self.num_layers - 1:
                        grad = 2 * (norm.pdf(0, mean_u, np.sqrt(var_u)) / norm.cdf(0, -y_train_mb * mean_u, np.sqrt(var_u))) \
                            / np.sqrt(self.network_size[l] + 1)
                        non_finite_indices = np.logical_not(np.isfinite(grad))
                        grad[non_finite_indices] = -2 * ((y_train_mb[non_finite_indices] * mean_u[non_finite_indices] < 0) \
                            * (mean_u[non_finite_indices] / var_u[non_finite_indices])) / np.sqrt(self.network_size[l] + 1)  
                        delta_next = np.copy(y_train_mb)
                    else:
                        delta_next = delta
                        grad = 2 * norm.pdf(0, mean_u, np.sqrt(var_u)) / np.sqrt(self.network_size[l] + 1)
                    
                    delta = np.matmul(np.transpose(tanh), (delta_next * grad))
                    h = h + 0.5 * np.matmul((delta_next * grad), np.transpose(mean_v_prev))

                    self.weights[l] = h
                    self.tanh_weights[l] = np.tanh(h)
                    self.biases[l] = bias + 0.5 * np.expand_dims(np.sum(delta_next * grad, 1), axis=1)
            
        return self.forward_pass(self.test_data['x'], self.test_data['y'])
