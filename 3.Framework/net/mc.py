# 微簇，注意需要修改为向量模式
import math
import numpy as np


class MicroCluster:
    def __init__(self, data, re=1, label=-1, radius=-1., lmbda=1e-4):
        self.n = 1
        self.nl = 0 if label == -1 else 1
        self.ls = data
        self.ss = np.square(data)
        self.t = 0
        self.re = re
        self.label = label
        self.radius = radius

        self.lmbda = lmbda
        self.epsilon = 0.00005
        self.radius_factor = 1.1

    def insert(self, data, labeled=False):
        self.n += 1
        self.nl += 1 if labeled else 0
        self.ls += data
        self.ss += np.square(data)
        self.t = 0
        # self.re = 1 if labeled else self.re       # 添加了这个地方:0901-18:34
        self.radius = self.get_radius()

    def update_reliability(self, probability, increase=True):
        if increase:
            self.re += max(1 - self.re, (1 - self.re) * math.pow(math.e, probability - 1))
        else:
            self.re -= (1 - self.re) * math.pow(math.e, probability)
            # self.re -= 1 - math.pow(math.e, - probability)

    def update(self):
        self.t += 1
        self.re = self.re * math.pow(math.e, - self.lmbda * self.epsilon * self.t)
        return self.re

    # 查
    def get_deviation(self):
        ls_mean = np.sum(np.square(self.ls / self.n))
        ss_mean = np.sum(self.ss / self.n)
        variance = ss_mean - ls_mean
        variance = 1e-6 if variance < 1e-6 else variance
        radius = np.sqrt(variance)
        return radius

    def get_center(self):
        return self.ls / self.n

    def get_radius(self):
        if self.n <= 1:
            return self.radius
        return max(self.radius, self.get_deviation() * self.radius_factor)

    def __str__(self):
        return f"n = {self.n}; nl = {self.nl}; label = {self.label}; ls = {self.ls.shape}; ss = {self.ss.shape}; " \
               f"t = {self.t}; re = {self.re}; ra = {self.get_radius()}\n "


if __name__ == '__main__':
    mc = MicroCluster(np.array([1, 2, 3, 4, 5]))
    print(mc)
