from __future__ import print_function, absolute_import
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch

__all__ = ['accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(correct[:k].size()[0]*correct[:k].size()[1]).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return torch.Tensor(res)

def evaluation(X, Y, Kset):
    num = X.shape[0]
    classN = np.max(Y)+1
    kmax = np.max(Kset)
    recallK = np.zeros(len(Kset))
    #compute NMI
    kmeans = KMeans(n_clusters=classN).fit(X)
    nmi = normalized_mutual_info_score(Y, kmeans.labels_, average_method='arithmetic')
    #compute Recall@K
    sim = X.dot(X.T)
    minval = np.min(sim) - 1.
    sim -= np.diag(np.diag(sim))
    sim += np.diag(np.ones(num) * minval)
    indices = np.argsort(-sim, axis=1)[:, : kmax]
    YNN = Y[indices]
    for i in range(0, len(Kset)):
        pos = 0.
        for j in range(0, num):
            if Y[j] in YNN[j, :Kset[i]]:
                pos += 1.
        recallK[i] = pos/num
    return nmi, recallK