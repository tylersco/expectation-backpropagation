import random
import sys
import tensorflow as tf
import numpy as np

from read_mnist import MNIST
from ebp import ExpectationBackpropagationNetwork

def main():
    replications = 1
    network_size = [784, 200, 10]
    global_random_seed = 42
    random.seed(global_random_seed)
    initialization_seq = random.sample(range(50000), replications)

    #performance = []
    for i in range(replications):
        tf.set_random_seed(initialization_seq[i])
        np.random.seed(initialization_seq[i])
        mnist = MNIST(sys.argv[1], initialization_seq[i])
        d = mnist.get_data()
        ebpn = ExpectationBackpropagationNetwork(d, network_size)
        ebpn.train()

if __name__ == '__main__':
    main()