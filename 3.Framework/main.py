import argparse
import logging
import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import euclidean_distances
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import config.config as cfg
from net.base import NoModel
from net.mc import MicroCluster
from utils.data import DataSet


class OnlineLearning(object):
    def __init__(self, args):
        # hyperparameters
        self.args = args

        # main framework
        self.lmcs = []
        self.umcs = []
        self.classes = []
        self.counter = {-1: 0}
        self.avg_radius = 0.0

        # statistic info
        self.label_true = []
        self.label_predict = []
        self.propagation_times = 0
        self.propagation_success = 0
        self.labeled_num = 0

        self.create_num = 0
        self.re_list = []

        # logging
        self.writer = SummaryWriter()

        # model and scaler
        self.model, self.scaler = self.load_model()

        # initial
        self.initialization()
        self.dataloader = self.ready_data()
        self.describe(0)

    def start(self):
        device = torch.device('cuda' if self.args.cuda & torch.cuda.is_available() else 'cpu')

        self.model.to(device)

        start = time.time()
        for i, (data, true_label, semi_label) in enumerate(self.dataloader):
            data = data.to(device)
            x = self.model.trunk(data)
            x = self.model.embedder(x).cpu().detach()[0].numpy()

            # compute distance
            dis = self.cal_distance(x)
            pl, pr = self.classify(dis, semi_label.item())
            self.label_true.append(true_label.item())
            self.label_predict.append(pl)

            self.insert_data(x, pl, semi_label.item(), pr, dis)
            self.decay_time()

            if (i + 1) % 1000 == 0:
                self.describe(i + 1)

        end = time.time()
        print("total cost time: {cost:.3f}s".format(cost=end - start))

    def load_model(self):
        if self.args.use_metric:
            model_path = f"./model/{self.args.dataset}-model.pt"
            model = torch.load(model_path)
        else:
            model = NoModel()

        scaler_path = f"./model/{self.args.dataset}-scaler.pkl"
        scaler = joblib.load(scaler_path)

        model.eval()
        return model, scaler

    def initialization(self):
        # transform x from original space to embedding space
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(device)

        # get the embeded initial data
        path = f"data/init/{self.args.dataset}.npy"
        init_data = np.load(path).astype(np.float32)
        x = self.scaler.transform(init_data[:, :-2])
        y = init_data[:, -2:].astype(np.int64)
        dataset = DataSet(x, y)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=2)

        x_ebd = torch.Tensor()
        for i, (inputs, _, _) in enumerate(dataloader):
            inputs = inputs.to(device)

            trunk = self.model.trunk(inputs)
            embedding = self.model.embedder(trunk)

            x_ebd = torch.cat((x_ebd, embedding.cpu()), 0)

        # initialize the micro-clusters
        x = x_ebd.detach().numpy()
        y = y[:, 0]
        self.classes = list(set(y))

        for cls in self.classes:
            index = (y == cls)
            data = x[index]
            self.counter[cls] = 0

            if len(data) > self.args.init_k_per_class:  # samples number is smaller than specific parameter
                kmeans = KMeans(n_clusters=self.args.init_k_per_class)
                kmeans.fit(data)
                kmeans_labels = kmeans.labels_
                for _cls in range(self.args.init_k_per_class):
                    _data_cls = data[kmeans_labels == _cls]
                    if len(_data_cls) == 0:
                        continue
                    mc = MicroCluster(_data_cls[0], label=cls, lmbda=self.args.lmbda)
                    for _d in _data_cls[1:]:
                        mc.insert(_d, labeled=True)
                    self.lmcs.append(mc)
                    self.counter[cls] += 1
            else:
                mc = MicroCluster(data[0], label=cls, lmbda=self.args.lmbda)
                for d in data[1:]:
                    mc.insert(d, labeled=True)
                self.lmcs.append(mc)
                self.counter[cls] += 1

        # self.avg_radius = np.max(np.array([mc.radius for mc in lmcs if mc.n > 1]))
        self.avg_radius = np.average(np.array([mc.radius for mc in self.lmcs if mc.n > 1]))
        logging.info(f'average radius: {self.avg_radius}')
        # amend the radius
        for mc in self.lmcs:
            if mc.n <= 1:
                mc.radius = self.avg_radius

    def ready_data(self):
        data_path = f"data/eval/{self.args.dataset}.npy"

        data = np.load(data_path)
        labels = data[:, -2:].astype(np.int64)
        data = data[:, :-2].astype(np.float32)

        dataset = DataSet(data, labels)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        return dataloader

    def classify(self, dis, label_semi):
        mcs = np.array([[mc.label, mc.re] for mc in self.lmcs])

        # get topk of dis, class, reliability
        if dis.shape[0] == 1:  # the number of mcs may decrease, smaller than args.k
            topk_idx = np.array([0])
            topk_dis = dis[topk_idx] + 1e-5
        else:
            k = min(self.args.k, len(self.lmcs) - 1)
            topk_idx = np.argpartition(dis[:len(self.lmcs)], k)[:k]
            topk_dis = dis[topk_idx] + 1e-5
            topk_dis /= np.min(topk_dis)
        topk_cls = mcs[topk_idx, 0]
        topk_res = mcs[topk_idx, 1]

        # predict
        ret_cls = np.zeros(len(self.classes))
        ret_re = np.zeros(len(self.classes))
        probabilities = softmax(topk_res / topk_dis)  ############# reliable distance
        for i, cls in enumerate(topk_cls):
            index = self.classes.index(cls)
            ret_cls[index] += 1
            ret_re[index] += probabilities[i]  # sum of topk's reliability of class index
        label_pred = self.classes[np.argmax(ret_cls)]
        re_pred = max(ret_re)

        # update the reliability of topk if the true label is knownrt
        if label_semi != -1:
            correct = label_pred == label_semi
            for i, cls in enumerate(topk_cls):
                mc = self.lmcs[topk_idx[i]]
                mc.update_reliability(probabilities[i], increase=correct)
        return label_pred, re_pred

    def cal_distance(self, x):
        lcs = np.array([mc.get_center() for mc in self.lmcs])
        ucs = np.array([mc.get_center() for mc in self.umcs])

        if len(ucs) >= 1:
            centers = np.vstack([lcs, ucs])
        else:
            centers = lcs

        dis = euclidean_distances(centers, x.reshape((1, -1)))
        return dis.flatten()

    def insert_data(self, data, label_pred, label_semi, re, dis):
        known = False if label_semi == -1 else True

        min_idx = np.argmin(dis)
        if min_idx < len(self.lmcs):
            nearest_mc = self.lmcs[min_idx]
        else:
            nearest_mc = self.umcs[min_idx - len(self.lmcs)]

        if (dis[min_idx] < nearest_mc.radius) and (re >= self.args.minRE):
            if known and (nearest_mc.label == label_semi or nearest_mc.label == -1):
                nearest_mc.insert(data, labeled=known)
                if nearest_mc.label == -1:
                    self.counter[label_semi] += 1
                    self.counter[-1] -= 1
                nearest_mc.label = label_semi
            elif not known and (nearest_mc.label == label_pred or nearest_mc.label == -1):
                nearest_mc.insert(data, labeled=known)
                if nearest_mc.label == -1:
                    self.counter[label_pred] += 1
                    self.counter[-1] -= 1
                nearest_mc.label = label_pred
            else:
                if len(self.umcs) >= self.args.maxUMC:
                    self.drop(unlabeled=True)

                if len(self.lmcs) >= self.args.maxMC:
                    self.drop()

                re = 1 if known else re
                label = label_semi if known else label_pred
                mc = MicroCluster(data, re=re, label=label, radius=self.avg_radius, lmbda=self.args.lmbda)
                self.lmcs.append(mc)
                # self.mcs.append(mc)

                self.create_num += 1
                self.counter[label] += 1
        else:
            if len(self.umcs) > self.args.maxUMC:
                self.drop(unlabeled=True)

            if len(self.lmcs) >= self.args.maxMC:
                self.drop()

            mc = MicroCluster(data, label=label_semi, radius=self.avg_radius, lmbda=self.args.lmbda)
            if label_semi == -1:
                self.umcs.append(mc)
            else:
                self.lmcs.append(mc)

            self.create_num += 1
            self.counter[label_semi] += 1

    def drop(self, unlabeled=False):
        def key(elem):
            return elem.t

        if unlabeled:
            mcs = self.umcs
        else:
            mcs = self.lmcs

        mcs.sort(key=key, reverse=False)  # 是否需要通过排序来解决，并且一次只删除一个，基本上每次都会删除
        for mc in mcs[-self.args.k:]:
            self.counter[mc.label] -= 1
            if len(self.lmcs) == 0:
                pass
            mcs.remove(mc)

    def decay_time(self):
        radius = []
        for mcs in [self.lmcs, self.umcs]:
            for i, mc in enumerate(mcs):
                re = mc.update()
                if re < self.args.minRE:
                    self.counter[mc.label] -= 1
                    mcs.pop(i)
                else:
                    radius.append(mc.get_radius())

        # 重新计算avg_radius
        if len(radius) > 1:
            avg_radius = np.average(np.array(radius))
            for mcs in [self.lmcs, self.umcs]:
                for mc in mcs:
                    if mc.n <= 1:
                        mc.radius = avg_radius

        if self.counter[-1] >= self.args.maxUMC:
            if not (self.args.do_propagate and self.propagate()):
                self.drop(unlabeled=True)

    def propagate(self):
        # 具有传播效应，与先计算距离再传播不同
        flag = False
        self.propagation_times += 1
        # print('propagating ', self.counter)

        for i, umc in enumerate(self.umcs):
            ucenter = umc.get_center().reshape([1, -1])
            dis = self.cal_distance(ucenter)
            label, re = self.classify(dis[:len(self.lmcs)], -1)
            if re > self.args.minRE:
                umc.label = label
                umc.re = re
                flag = True
                self.counter[-1] -= 1
                self.counter[label] += 1
                self.umcs.pop(i)
                self.lmcs.append(umc)
        if flag:
            self.propagation_success += 1
        return flag

    def describe(self, i):
        print(f"the {i}-th instance")
        print('平均半径:', self.avg_radius)
        print('新建微簇个数：', self.create_num)
        data = {'n': [], 'nl': [], 'label': [], 'ls': [], 'ss': [], 't': [], 're': [], 'ra': []}
        for mc in self.lmcs:
            data['n'].append(mc.n)
            data['nl'].append(mc.nl)
            data['label'].append(mc.label)
            data['ls'].append(mc.ls)
            data['ss'].append(mc.ss)
            data['t'].append(mc.t)
            data['re'].append(mc.re)
            data['ra'].append(mc.get_radius().item())
            # print(mc)
        df_data = pd.DataFrame(data)
        print(df_data.describe())
        print(Counter(data['label']))
        print(self.counter)

    def evaluation(self):
        rep = classification_report(self.label_true, self.label_predict, digits=4)
        return rep

    def save_result(self):
        out_path = f"./result/{self.args.dataset}.npy"
        data = np.hstack([np.array(self.label_true).reshape((-1, 1)), np.array(self.label_predict).reshape((-1, 1))])
        np.save(out_path, data)


if __name__ == "__main__":
    args = cfg.get_options(argparse.ArgumentParser())
    model = OnlineLearning(args)
    model.start()
    res = model.evaluation()
    print(res)
    model.save_result()
