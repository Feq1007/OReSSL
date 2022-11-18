import math

import torch

class MicroCluster(object):
    def __init__(self, data, re=1, label=-1, radius=-1.0, lmbda=1e-4):
        self.n = 1
        self.nl = 0 if label==-1 else 1
        self.ls = data
        self.ss = torch.square(data)
        self.t = 0
        self.re = re
        self.label = label
        self.radius = radius

        self.lmbda = lmbda
        self.espilon = 0.00005
        self.radius_factor = 5
        
    def insert(self, data, labeled=False):
        self.n += 1
        self.nl += 1 if labeled else 0
        self.ls += data
        self.ss += torch.square(data)
        self.t = 0
        # self.re = 1 if labeled else self.re       # 添加了这个地方:0901-18:34
        self.radius = self.get_radius()        

    def replace(self, data, label, radius):
        self.n = 1
        self.nl = 1 if label!=-1 else 0
        self.label = label
        self.ls += data
        self.ss += torch.square(data)
        self.t = 0
        self.re = 1
        self.radius = radius

    def update_reliability(self, probability, increase=True):
        if increase:
            self.re += max(1-self.re, (1 - self.re) * math.pow(math.e, probability - 1))
        else:
            # self.re -= (1 - self.re) * math.pow(10, probability)
            self.re -= 1 - math.pow(math.e, -probability)

    def update(self):
        self.t += 1
        self.re = self.re * math.pow(math.e, - self.lmbda * self.espilon * self.t)
        return self.re

    # 查
    def get_deviation(self):
        ls_mean = torch.sum(torch.square(self.ls / self.n))
        ss_mean = torch.sum(self.ss / self.n)
        variance = ss_mean - ls_mean
        torch.square(self.ls / self.n)
        variance = 1e-6 if variance < 1e-6 else variance
        radius = torch.sqrt(variance)
        return radius

    def get_center(self):
        return self.ls / self.n

    def get_radius(self):
        if self.n <= 1:
            return self.radius
        return max(self.radius,self.get_deviation() * self.radius_factor)

    def __str__(self):
        return f"n = {self.n}; nl = {self.nl}; label = {self.label}; ls = {self.ls.shape}; ss = {self.ss.shape}; t = {self.t}; re = {self.re}; ra = {self.get_radius()}\n"
        