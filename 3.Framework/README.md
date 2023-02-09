# 可靠性半监督数据流学习研究
## 研究计划
**StepOne**: 搭建完整个项目流程
- [ ] 数据预处理：训练集和测试集，在此基础上，完成类别不平衡预处理
- [ ] 项目结构梳理确认：各文件夹，文件等内容确定
**Step Two**:完成loss修改
- [ ] 度量学习
- [ ] 对比学习
**Step Three**：完成类别不平衡项目研究
- [ ] 将微簇模型迁移到tensor上
- [ ] 完成类别不平衡预测
**Step Four**：实验验证即可视化工具
**Step Five**：对比试验

## 数据预处理

主要在utils中完成
1. 从参数中读取需要处理的文件
2. 将数据划分为训练集和测试集
针对类别不平衡和类别平衡两种情况，首先确定各样本的数据比例，然后计算出需要筛选多少样本作为初始数据。


```python
!python generate_data.py
```

    original class information: 
        classes  number
    0        0   31250
    1        1   31250
    2        2   31250
    3        3   31250
    imbalanced class information: 
        classes  number
    0        0    3125
    1        1    6732
    2        2   14504
    3        3   31250
    initial data information: 
        classes  number
    0        0     113
    1        1     243
    2        2     522
    3        3    1124
    stream data information: 
        classes  number
    0        0    3012
    1        1    6489
    2        2   13982
    3        3   30126
    masked initial data information:    classes  number
    0        0      40
    1        1      99
    2        2     218
    3        3     445
    4       -1    1200
    masked stream data information:    classes  number
    0        0    1200
    1        1    2606
    2        2    5521
    3        3   12117
    4       -1   32165
    save data to path: ./data/{init,eval}...



```python
# import torch
import numpy as np
import pandas as pd

opt = {'dataset':'4CRE-V1.txt',
      'imb_ratio':100,
       'imb_type':'long',
      'label_ratio':20,
      'num_max':1000,
       'init_size':1000
      }

data_path = f"data/benchmark/{opt['dataset']}"

def compute_class_info(labels):
    """
    return the statistic information about class information
    """
    classes = list(set(labels))
    class_num = len(classes)
    class_num_true_list = []
    for c in classes:
        class_num_true_list.append(len(np.where(labels==c)[0]))
    class_info = np.array([classes, class_num_true_list], dtype=int).transpose()
    class_info = pd.DataFrame(class_info, columns=['classes', 'number'])
    class_info = class_info.sort_values(by='number')   
    return class_info

def compute_imb_class_info(class_info, init_size, imb_ratio, imb_type):
    """
    description: according to the label and imbalance ratio to compute the number of each class in initial set and evaluate set
    
    """
    class_num_list = []
    class_num = class_info.shape[0]
    max_num = class_info.iloc[-1][-1]
    gamma = imb_ratio
    if imb_type == 'long':
        mu = np.power(1/gamma, 1/(class_num - 1))
        for i in range(class_num):
            if i == (class_num - 1):
                class_num_list.append(max_num / gamma)
            else:
                class_num_list.append(max_num * np.power(mu, i))
    elif imb_type == 'step':
        for i in range(class_num):
            if i < int((class_num) / 2):
                class_num_list.append(int(max_num))
            else:
                class_num_list.append(int(max_num / gamma))
                
    list.reverse(class_num_list)
    for i in range(class_num):
        if class_num_list[i] < class_info.iloc[i][-1]:
            class_info.iloc[i][-1] = class_num_list[i]
    return class_info

def sample(labels, class_info, init_size):
    """
    split the data into train and evaluate data,
    """
    init_nums = np.array(class_info.iloc[:,-1], dtype=float) / np.sum(class_info.iloc[:,-1]) * init_size
    init_nums = init_nums.astype(int)
    
    init_idxs = []
    eval_idxs = []
    for i in range(class_info.shape[0]):
        label = class_info.iloc[i][0]
        idxs = np.where(labels == label)[0] # return the index of data
        np.random.shuffle(idxs)
        idxs = idxs[:class_info.iloc[i,1]]
        idxs = np.sort(idxs)
        init_idxs.extend(idxs[:init_nums[i]])
        eval_idxs.extend(idxs[init_nums[i]:])
    return np.sort(init_idxs), np.sort(eval_idxs)

def split_init_data(data_path, init_size, label_ratio, imb_ratio, imb_type='long'):
    data = pd.read_csv(data_path, header=None, dtype=float)

    labels = data.iloc[:,-1]
    labels = np.array(data.iloc[:,-1], dtype=int)
    
    class_info = compute_class_info(labels)
    print('class information before sampling: \n', class_info)
    
    imb_class_info = compute_imb_class_info(class_info, init_size, imb_ratio, imb_type)
    print('class information after sampling: \n', imb_class_info)
    
    init_idxs, eval_idxs = sample(labels, class_info, init_size)
    print(compute_class_info(labels[init_idxs]))
    print(compute_class_info(labels[eval_idxs]))
    
    # random mask
    init_mask = np.random.choice(init_idxs, size=int(init_size * (100.0 - label_ratio) / 100), replace=False)
    eval_mask = np.random.choice(eval_idxs, size=int(len(eval_idxs) * (100.0 - label_ratio) / 100), replace=False)
    print(compute_class_info(labels[init_mask]))
    print(compute_class_info(labels[eval_mask]))
    
    # encode labels
    cols = data.columns
    classes = list(class_info.values[:,0])
    encode_labels = data[cols[-1]].apply(lambda x:classes.index(int(x))).values.reshape((-1,1))
    semi_labels = np.copy(encode_labels)
    data = np.hstack([data.values[:,:-1], encode_labels, semi_labels])
    
    init_data = data[init_idxs]
    eval_data = data[eval_idxs]
    print(compute_class_info(init_data[:,-1]))
    print(compute_class_info(eval_data[:,-1]))
    
    
    file_name = opt['dataset'][:opt['dataset'].rindex('.')]+".npy"
    print(r"save data to path: ./data/{init,eval}/",file_name,sep='')
    np.save(f'./data/init/{file_name}', init_data, allow_pickle=False)
    np.save(f'./data/eval/{file_name}', eval_data, allow_pickle=False)
    
split_init_data(data_path, opt['init_size'], opt['label_ratio'], opt['imb_ratio'], opt['imb_type'])
```

    class information before sampling: 
        classes  number
    0        1   31250
    1        2   31250
    2        3   31250
    3        4   31250
    class information after sampling: 
        classes  number
    0        1     312
    1        2    1450
    2        3    6732
    3        4   31250
       classes  number
    0        1       7
    1        2      36
    2        3     169
    3        4     786
       classes  number
    0        1     305
    1        2    1414
    2        3    6563
    3        4   30464
       classes  number
    0        1       5
    1        2      31
    2        3     134
    3        4     630
       classes  number
    0        1     243
    1        2    1129
    2        3    5254
    3        4   24370
       classes  number
    0        0       7
    1        1      36
    2        2     169
    3        3     786
       classes  number
    0        0     305
    1        1    1414
    2        2    6563
    3        3   30464
    save data to path: ./data/{init,eval}/4CRE-V1.npy


# 半监督表征学习
## 度量学习
度量学习通常是有监督学习


```python
opt = {'dataset':'4CRE-V1',
      'imb_ratio':100,
       'imb_type':'long',
      'label_ratio':20,
      'num_max':1000,
       'init_size':1000,
       'weight_decay':0.0001,
       'lr':0.0001,
       'batch_size':32,
       'num_epochs':4,
       'train_eval_ratio':0.8,
       'start_epoch':0,
       'epochs':50,
       'rate':0.1,
       'cuda':False
      }
```


```python
import time

import joblib
import numpy as np
import torch
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.reducers import ThresholdReducer, AvgNonZeroReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from sklearn.preprocessing import Normalizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config.config as cfg
import utils.eval as eva
from net.base import BaseModel
from utils.data import DataSet
from utils.plot import plot_scatter
```


```python
data = torch.randn(2,5)
data.shape[0]
```




    2




```python
!nvidia-smi
```

    Mon Nov 28 18:14:31 2022       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 515.57       Driver Version: 515.57       CUDA Version: 11.7     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  NVIDIA GeForce ...  Off  | 00000000:03:00.0 Off |                  N/A |
    | 35%   44C    P2    66W / 250W |    922MiB / 11264MiB |      5%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   1  NVIDIA GeForce ...  Off  | 00000000:04:00.0 Off |                  N/A |
    | 35%   38C    P8    29W / 250W |     17MiB / 11264MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   2  NVIDIA GeForce ...  Off  | 00000000:81:00.0 Off |                  N/A |
    | 35%   42C    P8    23W / 250W |     17MiB / 11264MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   3  NVIDIA GeForce ...  Off  | 00000000:82:00.0 Off |                  N/A |
    |  0%   31C    P8    11W / 300W |   4810MiB / 11264MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A     41814      C   python                            905MiB |
    |    3   N/A  N/A     47729      C   ...onda3/envs/131/bin/python     4793MiB |
    +-----------------------------------------------------------------------------+



```python
def main(layer_sizes, opt):
    # load data
    data_path = f"./data/init/{opt.dataset}.npy"

    data = np.load(data_path, allow_pickle=False).astype('float32')
    np.random.shuffle(data)

    # Standardscaler or norminalizer
    X = data[:, :-2]
    Y = data[:, -2:].astype(np.int64)

    scaler = Normalizer().fit(X)
    # scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    plot_scatter(X[:, 0], X[:, 1], np.ones(X.shape[0]), Y[:, 0] * 52)

    # data = torch.from_numpy(data)
    train_num = int(X.shape[0] * opt.train_eval_ratio)
    train_dataset = DataSet(X[:train_num], Y[:train_num])
    eval_dataset = DataSet(X[train_num:], Y[train_num:])
    print(len(train_dataset), len(eval_dataset))

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    # device : cuda or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create model
    models = BaseModel(layer_sizes)

    models.to(device)

    # optimizer
    optimizers = torch.optim.Adam([{"params": models.trunk.parameters(), "lr": opt.lr},
                                   {"params": models.embedder.parameters(), "lr": opt.lr},
                                   {"params": models.classifier.parameters(), "lr": opt.lr}],
                                  weight_decay=opt.weight_decay)

    # classification loss
    classification_loss = torch.nn.CrossEntropyLoss()

    # metric loss & miner
    distance = LpDistance(power=2)
    miner = miners.MultiSimilarityMiner(epsilon=0.1)
#     metric_loss = losses.MultiSimilarityLoss()
    metric_loss = losses.TripletMarginLoss(distance=distance, reducer=AvgNonZeroReducer(), embedding_regularizer=LpRegularizer())
    

    criterions = [classification_loss, metric_loss]

    # tensorboard
    writer = SummaryWriter()
    for epoch in range(opt.start_epoch, opt.epochs):
        print('Training in Epoch[{}]'.format(epoch))
        adjust_learning_rate(optimizers, epoch, opt)

        # train for one epoch
        loss, acc = train(epoch, train_dataloader, models, criterions, miner, optimizers, device, opt)
        writer.add_scalar("loss/train", loss, epoch)
        writer.add_scalar("acc/train", loss, epoch)
    nmi, recall, acc = validate(eval_dataloader, models, device, opt)
    print(
        'Recall@1, 2, 4, 8: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; NMI: {nmi:.3f}; accuracy: {acc:.3f} \n'
            .format(recall=recall, nmi=nmi, acc=acc))
    save(models, scaler, opt)


def train(epoch, train_loader, models, criterions, miner, optimizers, device, opt):
    models.train()

    # log item
    running_loss = 0.0
    running_corrects = 0.0

    since = time.time()
    for i, (inputs, target, _) in enumerate(train_loader):
        inputs = inputs.to(device)
        target = target.to(device)

        trunk = models.trunk(inputs)
        embedding = models.embedder(trunk)
        output = models.classifier(embedding)

        # loss
        classification_loss = criterions[0](output, target)
        pairs = miner(embedding, target)
        metric_loss = criterions[1](embedding, target, pairs)
        loss = classification_loss + 0.5 * metric_loss
        # loss = metric_loss

        # compute gradient and SGD step
        optimizers.zero_grad()
        loss.backward()
        optimizers.step()

        # predict result
        _, preds = torch.max(output.data, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == target.data)

    # compute the average loss and accuracy
    epoch_loss = running_loss / 800
    epoch_acc = float(running_corrects) / 800

    stop = time.time()
    print("Cost time: {time:.3f}s".format(time=stop - since))
    print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc


def adjust_learning_rate(optimizer, epoch, opt):
    # decayed lr by 10 every 20 epochs
    if (epoch + 1) % 20 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= opt.rate


def validate(test_loader, models, device, opt):
    models.eval()

    acc = 0.0
    testdata = torch.Tensor()
    testlabel = torch.LongTensor()
    with torch.no_grad():
        for i, (inputs, target, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            target = target.to(device)

            trunk = models.trunk(inputs)
            embedding = models.embedder(trunk)
            output = models.classifier(embedding)

            # predict result
            _, preds = torch.max(output.data, 1)
            acc += torch.sum(preds == target.data)

            testdata = torch.cat((testdata, output.cpu()), 0)
            testlabel = torch.cat((testlabel, target.cpu()))
    acc = acc / 200.0
    nmi, recall = eva.evaluation(testdata.numpy(), testlabel.numpy(), [1, 2, 4, 8])
    return nmi, recall, acc


def save(models, scaler, opt):
    model_path = f"./model/{opt.dataset}-model.pt"
    scaler_path = f"./model/{opt.dataset}-scaler.pkl"
    torch.save(models, model_path)
    joblib.dump(scaler, scaler_path)


def transform(opt, init=True):
    # load data
    if init:
        data_path = f"./data/init/{opt.dataset}.npy"
        out_path = f"./data/init/trans/{opt.dataset}.npy"
    else:
        data_path = f"./data/eval/{opt.dataset}.npy"
        out_path = f"./data/eval/trans/{opt.dataset}.npy"

    data = np.load(data_path, allow_pickle=False).astype('float32')
    np.random.shuffle(data)

    # Standardscaler or norminalizer
    X = data[:, :-2]
    Y = data[:, -2:].astype(np.int64)

    scaler_path = f"./model/{opt.dataset}-scaler.pkl"
    scaler = joblib.load(scaler_path)
    X = scaler.transform(X)

    # dataset
    dataset = DataSet(X, Y)

    # dataloader
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    # device : cuda or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = f"./model/{opt.dataset}-model.pt"
    models = torch.load(model_path)
    models.to(device)
    models.eval()

    embeddings = []
    with torch.no_grad():
        for i, (inputs, target, mlabel) in enumerate(dataloader):
            inputs = inputs.to(device)
            target = target.to(device)

            trunk = models.trunk(inputs)
            embedding = models.embedder(trunk)

            embeddings.append(
                torch.hstack((embedding.cpu(), torch.unsqueeze(target.cpu(), -1), torch.unsqueeze(mlabel.cpu(), -1))))
    embeddings = torch.vstack(embeddings)
    embeddings = embeddings.numpy()

    np.save(out_path, embeddings, allow_pickle=False)


if __name__ == "__main__":
    opt = cfg.get_options()
    #     transform(opt)
    layer_sizes = [[2, 4], [4, 2], [2, 4]]
    # layer_sizes = [[9,32,64],[64,16],[16,7]]
    main(layer_sizes, opt)
    print('finish')
```

# Evaluation


```python

```

## 对比学习
度量学习通常是在图像上进行，因为图像数据增强方法很多且合理。如何在一般数据上进行数据增广呢？

# MicroCluster


```python
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

```

    n = 1; nl = 0; label = -1; ls = (5,); ss = (5,); t = 0; re = 1; ra = -1.0
     


# Data Stream Classification


```python
import joblib
import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from collections import Counter
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import euclidean_distances
import logging
```


```python
opt = {'dataset':'4CRE-V1',
      'imb_ratio':100,
       'imb_type':'long',
      'label_ratio':20,
      'num_max':1000,
       'init_size':1000,
       'weight_decay':0.0001,
       'lr':0.0001,
       'batch_size':32,
       'num_epochs':4,
       'train_eval_ratio':0.8,
       'start_epoch':0,
       'epochs':50,
       'rate':0.1,
       'init_k_per_class':30,
       'lambda':1e-4,
       'cuda':True,
       'knn':3,
       'minRE':0.8,
       'maxUMC':300,
       'maxMC':1000,
       'k':3
      }
```


```python
import torch.nn.functional as F
```


```python
def load_model(opt):
    model_path = f"./model/{opt['dataset']}-model.pt"
    scaler_path = f"./model/{opt['dataset']}-scaler.pkl"    
    
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
    
    path = f"data/init/{opt['dataset']}.npy"
    init_data = np.load(path).astype(np.float32)
    x = scaler.transform(init_data[:, :-2])
    y = init_data[:,-2:].astype(np.int)
    dataset = DataSet(x, y)
    dataloader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=False, num_workers=2)
    
    x_ebd = torch.Tensor()
    for i, (inputs, tl, sl) in enumerate(dataloader):
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

        if len(data) > opt['init_k_per_class']:  # samples number is smaller than specific parameter
            kmeans = KMeans(n_clusters=opt['init_k_per_class'])
            kmeans.fit(data)
            kmeans_labels = kmeans.labels_
            for _cls in range(opt['init_k_per_class']):
                _data_cls = data[kmeans_labels == _cls]
                if len(_data_cls) == 0:
                    continue
                mc = MicroCluster(_data_cls[0], label=cls, lmbda=opt['lambda'])
                for _d in _data_cls[1:]:
                    mc.insert(_d, labeled=True)
                mcs_labeled.append(mc)
                counter[cls] += 1
        else:
            mc = MicroCluster(data[0], label=cls, lmbda=opt['lambda'])
            for d in data[1:]:
                mc.insert(d, labeled=True)
            mcs_labeled.append(mc)
            counter[cls] += 1

    # self.avg_radius = np.max(np.array([mc.radius for mc in self.mcs_labeled if mc.n > 1]))
    avg_radius = np.average(np.array([mc.radius for mc in mcs_labeled if mc.n > 1]))
    logging.info(f'average radius : {avg_radius}')
    for mc in mcs_labeled:
        if mc.n <= 1:
            mc.radius = avg_radius    
    return mcs_labeled, counter, model, scaler
    
def ready_data(opt):
    data_path = f"data/eval/{opt['dataset']}.npy"
    
    data = np.load(data_path)
    labels = data[:,-2:].astype(np.int)
    data = data[:,:-2].astype(np.float32)
    
    dataset = DataSet(data, labels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    return dataloader
```


```python
from sklearn.metrics.pairwise import euclidean_distances
```


```python
lmcs = []
umcs = []
classes = []
counter = {-1:0}
avg_radius = 0.0
create_num = 0

def classify(dis, label_semi):
    mcs = np.array([[mc.label, mc.re] for mc in lmcs])

    # get topk of dis, class, reliability
    if dis.shape[0] == 1:  # the number of mcs may decrease, smaller than args.k
        topk_idx = np.array([0])
        topk_dis = dis[topk_idx] + 1e-5
    else:
        k = min(opt['k'], len(lmcs) - 1)
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
    
    dis = euclidean_distances(centers, x)
    return dis.flatten()

def insert_data(data, label_pred, label_semi, re, dis):
    global create_num
    known = False if label_semi == -1 else True

    min_idx = np.argmin(dis)
    if min_idx < len(lmcs):
        nearest_mc = lmcs[min_idx]
    else:
        nearest_mc = umcs[min_idx - len(lmcs)]

    if (dis[min_idx] < nearest_mc.radius) and (re >= opt['minRE']):
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
            if len(mcs_unlabeled) >= opt['maxUMC']:
                drop(unlabeled=True)

            if len(lmcs) >= opt['maxMC']:
                drop()

            re = 1 if known else re
            label = label_semi if known else label_pred
            mc = MicroCluster(data, re=re, label=label, radius=avg_radius, lmbda=opt['lambda'])
            lmcs.append(mc)
            # self.mcs.append(mc)

            create_num += 1
            counter[label] += 1
    else:
        if len(umcs) > opt['maxUMC']:
            drop(unlabeled=True)

        if len(lmcs) >= opt['maxMC']:
            drop()

        mc = MicroCluster(data, label=label_semi, radius=avg_radius, lmbda=opt['lambda'])
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
    for mc in mcs[-opt['k']:]:
        counter[mc.label] -= 1
        if len(lmcs) == 0:
            lmcs
        mcs.remove(mc)

def start(dataloader, model, scaler, opt):
    device = torch.device('cuda' if opt['cuda'] & torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    start = time.time()
    for i, (data, true_label, semi_label) in enumerate(dataloader):
        print(f'computing the {i} instance')
        data = data.to(device)
        
        x = model.trunk(data)
        x = model.embedder(x).cpu().detach().numpy()
        
        # compute distance
        dis = cal_distance(x)
        pl, pr = classify(dis, semi_label.item())
        
        insert_data(x, pl, semi_label.item(), pr, dis)
        if i == 1000:
            break
        
    end = time.time()
    print("total cost time: {cost:.3f}s".format(cost=end-start))


def main():
    global lmcs
    global umcs
    global counter
    
    # initial
    lmcs, cnt, model, scaler = initialization(opt)

    dataloader = ready_data(opt)
    
    for cls in classes:
        counter[cls] = 0
    
    # start
    start(dataloader, model, scaler, opt)
main()
```

    computing the 0 instance
    computing the 1 instance


    /home/panliangxu/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:43: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    TypeError: only size-1 arrays can be converted to Python scalars

    
    The above exception was the direct cause of the following exception:


    ValueError                                Traceback (most recent call last)

    <ipython-input-67-916827b7f1d1> in <module>
        163     # start
        164     start(dataloader, model, scaler, opt)
    --> 165 main()
    

    <ipython-input-67-916827b7f1d1> in main()
        162 
        163     # start
    --> 164     start(dataloader, model, scaler, opt)
        165 main()


    <ipython-input-67-916827b7f1d1> in start(dataloader, model, scaler, opt)
        137 
        138         # compute distance
    --> 139         dis = cal_distance(x)
        140         pl, pr = classify(dis, semi_label.item())
        141 


    <ipython-input-67-916827b7f1d1> in cal_distance(x)
         49         centers = lcs
         50 
    ---> 51     dis = euclidean_distances(centers, x)
         52     return dis.flatten()
         53 


    ~/anaconda3/lib/python3.7/site-packages/sklearn/metrics/pairwise.py in euclidean_distances(X, Y, Y_norm_squared, squared, X_norm_squared)
        300            [1.41421356]])
        301     """
    --> 302     X, Y = check_pairwise_arrays(X, Y)
        303 
        304     if X_norm_squared is not None:


    ~/anaconda3/lib/python3.7/site-packages/sklearn/metrics/pairwise.py in check_pairwise_arrays(X, Y, precomputed, dtype, accept_sparse, force_all_finite, copy)
        160             copy=copy,
        161             force_all_finite=force_all_finite,
    --> 162             estimator=estimator,
        163         )
        164         Y = check_array(


    ~/anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)
        744                     array = array.astype(dtype, casting="unsafe", copy=False)
        745                 else:
    --> 746                     array = np.asarray(array, order=order, dtype=dtype)
        747             except ComplexWarning as complex_warning:
        748                 raise ValueError(


    ~/anaconda3/lib/python3.7/site-packages/numpy/core/_asarray.py in asarray(a, dtype, order)
         81 
         82     """
    ---> 83     return array(a, dtype, copy=False, order=order)
         84 
         85 


    ValueError: setting an array element with a sequence.



```python
!pwd
```

    /share/panliangxu/workspace/jupyter/OReSSL/3.Framework



```python
!cat /home/panliangxu/.ssh/id_ed25519.pub
```

    ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFy8plBnmXjhSd7glCRHAlzRbdFwO0JIonLvJNuOkCY6 2805420128@qq.com

