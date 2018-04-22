import tensorflow as tf
import numpy as np

class ClassicBackpropagation:

    def __init__(self, data, network_size, epochs=150, batch_size=128, eta=0.001):

        self.epochs = epochs
        self.eta = eta
        self.batch_size = batch_size
        self.network_size = network_size
        self.num_layers = len(network_size) - 1
        
        self.train_data = data['train']
        self.valid_data = data['valid']
        self.test_data = data['test']

        # Network parameters
        self.weights = {}
        self.biases = {}
        for i in range(1, len(self.network_size)):
            if i < len(network_size) - 1:
                self.weights[i - 1] = tf.Variable(tf.random_normal([self.network_size[i - 1], self.network_size[i]]))
                self.biases[i - 1] = tf.Variable(tf.random_normal([self.network_size[i]]))
            else:
                self.weights[i - 1] = tf.Variable(tf.random_normal([self.network_size[i - 1], 10]))
                self.biases[i - 1] = tf.Variable(tf.random_normal([10]))

    def model(self):
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.int64, [None, 10])

        # Construct model

        # Fully connected layer 1
        fc1 = tf.add(tf.matmul(self.x, self.weights[0]), self.biases[0], name="fully_connected1")
        fc1 = tf.nn.tanh(fc1, name="fully_connected_relu1")

        if len(self.network_size) > 3:
            # Fully connected layer 2
            fc2 = tf.add(tf.matmul(fc1, self.weights[1]), self.biases[1], name="fully_connected2")
            fc2 = tf.nn.tanh(fc2, name="fully_connected_relu2")

            # Output, class prediction
            pred = tf.add(tf.matmul(fc2, self.weights[2]), self.biases[2], name="output")
        else:
            # Output, class prediction
            pred = tf.add(tf.matmul(fc1, self.weights[1]), self.biases[1], name="output")

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        num_correct = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return cost, num_correct, accuracy, optimizer

    def train(self):
        x_train, y_train = self.train_data['x'], self.train_data['y']
        x_valid, y_valid = self.valid_data['x'], self.valid_data['y']
        x_test, y_test = self.test_data['x'], self.test_data['y']

        _, num_correct, _, optimizer = self.model()

        # Initializing the variables
        init = tf.global_variables_initializer()

        def batch_eval(x, y, batch_size):
            total_correct = 0
            for i in range(0, len(y), batch_size):
                x_mb, y_mb = x[i:i + batch_size], y[i:i + batch_size]
                c = sess.run(num_correct, feed_dict={self.x: x_mb, self.y: y_mb})
                total_correct += c
            return total_correct / float(len(y))

        # Launch the graph
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init)

            for epoch in range(1, self.epochs + 1):

                shuffle = np.random.permutation(len(y_train))
                x_train, y_train = x_train[shuffle], y_train[shuffle]

                for i in range(0, len(y_train), self.batch_size):
                    x_train_mb, y_train_mb = x_train[i:i + self.batch_size], y_train[i:i + self.batch_size]

                    sess.run(optimizer, feed_dict={self.x: x_train_mb, self.y: y_train_mb})

                valid_acc = batch_eval(x_valid, y_valid, self.batch_size)
                print('Validation Accuracy: {0}'.format(valid_acc))

            print('Training done \n')

            test_acc = batch_eval(x_test, y_test, self.batch_size)
            print("Test Accuracy: {0}".format(test_acc))

            return test_acc