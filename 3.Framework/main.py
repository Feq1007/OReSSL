import argparse
import time
import config.config as cfg
import joblib
import numpy as np
import pandas as pd

import torch
from utils.data import DataSet
from torch.utils.data import DataLoader
from net.mc import MicroCluster
from tqdm import tqdm
from collections import Counter
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import euclidean_distances
import logging


opt = cfg.get_options(argparse.ArgumentParser())


lmcs = []
umcs = []
classes = []
counter = {-1:0}
avg_radius = 0.0
create_num = 0
propagation_times = 0

def load_model(opt):
    model_path = f"./model/{opt.dataset}-model.pt"
    scaler_path = f"./model/{opt.dataset}-scaler.pkl"    
    
    model = torch.load(model_path)
    scaler = joblib.load(scaler_path)

    model.eval()
    return model, scaler
    
def initialization(opt):
    global classes
    global avg_radius
    
    # transform x from original space to embedding space
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model, scaler = load_model(opt)
    model.to(device)
    
    path = f"data/eval/{opt.dataset}.npy"
    init_data = np.load(path).astype(np.float32)
    x = scaler.transform(init_data[:, :-2])
    y = init_data[:,-2:].astype(np.int64)
    dataset = DataSet(x, y)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=2)
    
    x_ebd = torch.Tensor()
    for i, (inputs, _, _) in enumerate(dataloader):
        inputs = inputs.to(device)
        
        trunk = model.trunk(inputs)
        embedding = model.embedder(trunk)
        
        x_ebd = torch.cat((x_ebd, embedding.cpu()), 0)
    
    # initialize the micro-clusters
    x = x_ebd.detach().numpy()
    y = y[:,0]
    classes = list(set(y))
    
    mcs_labeled = []
    counter = {-1:0}
    for cls in classes:
        index = (y == cls)
        data = x[index]
        counter[cls] = 0

        if len(data) > opt.init_k_per_class:  # samples number is smaller than specific parameter
            kmeans = KMeans(n_clusters=opt.init_k_per_class)
            kmeans.fit(data)
            kmeans_labels = kmeans.labels_
            for _cls in range(opt.init_k_per_class):
                _data_cls = data[kmeans_labels == _cls]
                if len(_data_cls) == 0:
                    continue
                mc = MicroCluster(_data_cls[0], label=cls, lmbda=opt.lmbda)
                for _d in _data_cls[1:]:
                    mc.insert(_d, labeled=True)
                mcs_labeled.append(mc)
                counter[cls] += 1
        else:
            mc = MicroCluster(data[0], label=cls, lmbda=opt.lmbda)
            for d in data[1:]:
                mc.insert(d, labeled=True)
            mcs_labeled.append(mc)
            counter[cls] += 1

    # self.avg_radius = np.max(np.array([mc.radius for mc in lmcs if mc.n > 1]))
    avg_radius = np.average(np.array([mc.radius for mc in mcs_labeled if mc.n > 1]))
    logging.info(f'average radius: {avg_radius}')
    for mc in mcs_labeled:
        if mc.n <= 1:
            mc.radius = avg_radius    
    return mcs_labeled, counter, model, scaler
    
def ready_data(opt):
    data_path = f"data/eval/{opt.dataset}.npy"
    
    data = np.load(data_path)
    labels = data[:,-2:].astype(np.int64)
    data = data[:,:-2].astype(np.float32)
    
    dataset = DataSet(data, labels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    return dataloader


def classify(dis, label_semi):
    mcs = np.array([[mc.label, mc.re] for mc in lmcs])

    # get topk of dis, class, reliability
    if dis.shape[0] == 1:  # the number of mcs may decrease, smaller than args.k
        topk_idx = np.array([0])
        topk_dis = dis[topk_idx] + 1e-5
    else:
        k = min(opt.k, len(lmcs) - 1)
        topk_idx = np.argpartition(dis, k)[:k]
        topk_dis = dis[topk_idx] + 1e-5
        topk_dis /= np.min(topk_dis)
    topk_cls = mcs[topk_idx, 0]
    topk_res = mcs[topk_idx, 1]

    # predict
    ret_cls = np.zeros(len(classes))
    ret_re = np.zeros(len(classes))
    probabilities = softmax(topk_res / topk_dis)  ############# reliable distance
    for i, cls in enumerate(topk_cls):
        index = classes.index(cls)
        ret_cls[index] += 1
        ret_re[index] += probabilities[i]  # sum of topk's reliability of class index
    label_pred = classes[np.argmax(ret_cls)]
    re_pred = max(ret_re)

    # update the reliability of topk if the true label is knownrt
    if label_semi != -1:
        correct = label_pred == label_semi
        for i, cls in enumerate(topk_cls):
            mc = lmcs[topk_idx[i]]
            mc.update_reliability(probabilities[i], increase=correct)
    return label_pred, re_pred

def cal_distance(x):
    lcs = np.array([mc.get_center() for mc in lmcs])
    ucs = np.array([mc.get_center() for mc in umcs])
    
    if len(ucs) >= 1:
        centers = np.vstack([lcs, ucs])
    else:
        centers = lcs
    
    dis = euclidean_distances(centers, x.reshape((1,-1)))
    return dis.flatten()

def insert_data(data, label_pred, label_semi, re, dis):
    global create_num
    known = False if label_semi == -1 else True

    min_idx = np.argmin(dis)
    if min_idx < len(lmcs):
        nearest_mc = lmcs[min_idx]
    else:
        nearest_mc = umcs[min_idx - len(lmcs)]

    if (dis[min_idx] < nearest_mc.radius) and (re >= opt.minRE):
        if known and (nearest_mc.label == label_semi or nearest_mc.label == -1):
            nearest_mc.insert(data, labeled=known)
            if nearest_mc.label == -1:
                counter[label_semi] += 1
                counter[-1] -= 1
            nearest_mc.label = label_semi
        elif not known and (nearest_mc.label == label_pred or nearest_mc.label == -1):
            nearest_mc.insert(data, labeled=known)
            if nearest_mc.label == -1:
                counter[label_pred] += 1
                counter[-1] -= 1
            nearest_mc.label = label_pred
        else:
            if len(umcs) >= opt.maxUMC:
                drop(unlabeled=True)

            if len(lmcs) >= opt.maxMC:
                drop()

            re = 1 if known else re
            label = label_semi if known else label_pred
            mc = MicroCluster(data, re=re, label=label, radius=avg_radius, lmbda=opt.lmbda)
            lmcs.append(mc)
            # self.mcs.append(mc)

            create_num += 1
            counter[label] += 1
    else:
        if len(umcs) > opt.maxUMC:
            drop(unlabeled=True)

        if len(lmcs) >= opt.maxMC:
            drop()

        mc = MicroCluster(data, label=label_semi, radius=avg_radius, lmbda=opt.lmbda)
        if label_semi == -1:
            umcs.append(mc)
        else:
            lmcs.append(mc)

        create_num += 1
        counter[label_semi] += 1


def drop(unlabeled=False):
    def key(elem):
        return elem.t

    if unlabeled:
        mcs = umcs
    else:
        mcs = lmcs

    mcs.sort(key=key, reverse=False)  # 是否需要通过排序来解决，并且一次只删除一个，基本上每次都会删除
    for mc in mcs[-opt.k:]:
        counter[mc.label] -= 1
        if len(lmcs) == 0:
            lmcs
        mcs.remove(mc)

def decay_time():
        radius = []
        for mcs in [lmcs, umcs]:
            for i, mc in enumerate(mcs):
                re = mc.update()
                if re < opt.minRE:
                    counter[mc.label] -= 1
                    mcs.pop(i)
                else:
                    radius.append(mc.get_radius())

        # 重新计算avg_radius
        if len(radius) > 1:
            avg_radius = np.average(np.array(radius))
            for mcs in [lmcs, umcs]:
                for mc in mcs:
                    if mc.n <= 1:
                        mc.radius = avg_radius

        if counter[-1] >= opt.maxUMC:
            if not (opt.do_propagate and propagate()):
                drop(unlabeled=True)

def propagate():
    # 具有传播效应，与先计算距离再传播不同
    flag = False
    propagation_times += 1
    # print('propagating ', self.counter)

    for i, umc in enumerate(umcs):
        ucenter = umc.get_center().reshape([1, -1])
        dis = cal_distance(ucenter)
        label, re = classify(dis[:len(lmcs)], -1)
        if re > opt.minRE:
            umc.label = label
            umc.re = re
            flag = True
            counter[-1] -= 1
            counter[label] += 1
            umcs.pop(i)
            lmcs.append(umc)
    if flag:
        propagation_success += 1
    return flag

def start(dataloader, model, scaler, opt):
    device = torch.device('cuda' if opt.cuda & torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    start = time.time()
    for i, (data, true_label, semi_label) in enumerate(dataloader):
        if i % 1000 == 0:
            print(f'computing the {i} instance')
        data = data.to(device)
        
        x = model.trunk(data)
        x = model.embedder(x).cpu().detach()[0].numpy()
        
        # compute distance
        dis = cal_distance(x)
        pl, pr = classify(dis, semi_label.item())
        
        insert_data(x, pl, semi_label.item(), pr, dis)

        decay_time()
        
    end = time.time()
    print("total cost time: {cost:.3f}s".format(cost=end-start))


def main():
    global lmcs
    global umcs
    global counter
    
    # initial
    lmcs, cnt, model, scaler = initialization(opt)
    print(cnt)
    dataloader = ready_data(opt)
    
    for cls in classes:
        counter[cls] = 0
    
    # start
    start(dataloader, model, scaler, opt)


if __name__=="__main__":
    main()