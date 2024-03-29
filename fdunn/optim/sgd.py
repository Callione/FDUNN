"""
sgd optimizer

https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from .base import Optimizer

class SGD(Optimizer):
    def __init__(self, model, lr=0.0):
        self.model = model
        self.lr = lr

    def step(self):
        """
        Performs a single optimization step.
        """
        ###########################################################################
        # TODO:                                                                   #
        # Implement the step method.                                              #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        for layer in self.model.layers:
            if hasattr(layer,'params'):
                if isinstance(layer.params, dict):
                    for key in layer.params.keys():
                        #print(f'{key} before:{layer.params[key]}')
                        #print(f'grad({key})={layer.grads[key]}')
                        #print(f'd key:={self.lr * layer.grads[key]}')
                        layer.params[key] = layer.params[key] - self.lr * layer.grads[key]
                        #print(f'{key} after:{layer.params[key]}')
                        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****