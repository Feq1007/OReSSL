import pandas as pd

# 普通数据若增强
import numpy as np

class WeakAugment(object):
    def __init__(self, data):
        labeled_data = data[data[:,-1]!=-1]

        label, data = self._get_min_class(labeled_data)

        self.var = np.var(data, axis=0)

    def _get_min_class(self, labeled_data):
        labeled_data_df = pd.DataFrame(labeled_data)
        info = labeled_data_df.groupby(labeled_data_df.columns[-1])
        num = 10086
        label, data = 0, 0
        for l, df in info:
            if df.shape[0] > num or df.shape[0] == 1:
                continue
            else:
                label = l
                data = df.values
                num = df.shape[0]
        return label, data[:,:-2]

    def __call__(self, x):
        return x + np.random.normal(loc=0, scale=self.var).astype(np.float32)

# 普通数据强增强
class StrongAugment(object):
    def __init__(self, data):
        labeled_data = data[data[:,-1]!=-1]

        label, data = self._get_max_class(labeled_data)

        self.var = np.var(data, axis=0)

    def _get_max_class(self, labeled_data):
        labeled_data_df = pd.DataFrame(labeled_data)
        info = labeled_data_df.groupby(labeled_data_df.columns[-1])
        num = -1
        label, data = 0, 0
        for l, df in info:
            if df.shape[0] < num:
                continue
            else:
                label = l
                data = df.values
                num = df.shape[0]
        return label, data[:,:-2]

    def __call__(self, x):
        return (x + np.random.normal(loc=0, scale=self.var)).astype(np.float32)

if __name__=="__main__":
    data = np.load("../data/init/shuttle.npy")
    weak = WeakAugment(data)
    strong = StrongAugment(data)
