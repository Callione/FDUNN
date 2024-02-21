import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from .base import Optimizer

class Adam(Optimizer):
    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # 初始化时间步

        # 初始化一些状态变量
        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1

        for layer in self.model.layers:
            if hasattr(layer, 'params') and isinstance(layer.params, dict):
                for key in layer.params.keys():
                    if key not in self.m:
                        self.m[key] = np.zeros_like(layer.params[key])
                        self.v[key] = np.zeros_like(layer.params[key])

                    # 更新m和v
                    self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * layer.grads[key]
                    self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (layer.grads[key] ** 2)

                    # 偏差修正
                    m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[key] / (1 - self.beta2 ** self.t)

                    # 更新参数
                    layer.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

