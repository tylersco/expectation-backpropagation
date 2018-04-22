import random
import sys
import tensorflow as tf
import numpy as np

from read_mnist import MNIST
from ebp_binary import ExpectationBackpropagationNetworkBinary
from bp import ClassicBackpropagation

def main():
    replications = 10
    network_size = [784, 500, 10]
    global_random_seed = 42
    random.seed(global_random_seed)
    initialization_seq = random.sample(range(50000), replications)

    performance = {
        'ebp': [],
        'bp': []
    }
    for i in range(replications):
        tf.set_random_seed(initialization_seq[i])
        np.random.seed(initialization_seq[i])

        mnist = MNIST(sys.argv[1], initialization_seq[i], prob=True)
        d = mnist.get_data()
        ebpn = ExpectationBackpropagationNetworkBinary(d, network_size)
        performance['ebp'].append(ebpn.train())

        mnist = MNIST(sys.argv[1], initialization_seq[i], prob=False)
        d = mnist.get_data()
        bpn = ClassicBackpropagation(d, network_size)
        performance['bp'].append(bpn.train())

    print(performance)

if __name__ == '__main__':
    main()