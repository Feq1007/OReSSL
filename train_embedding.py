import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import LpDistance
from sklearn.preprocessing import StandardScaler, Normalizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import  SummaryWriter

import config.config as cfg
from utils.generate_data import compute_class_info
from base.model import BaseModel
from progress.bar import Bar
from utils import AverageMeter
from base.augment import WeakAugment, StrongAugment
from base.data import DataSet, UnlabeledDataSet
from sklearn.metrics import classification_report
from imblearn.metrics import classification_report_imbalanced


def main(opt):
    print(f"========== Start train {opt.dataset} ===============")

    writer = SummaryWriter(f'.runs/train/{opt.dataset}')

    # load data
    data_path = f"./data/init/{opt.dataset}.npy"
    result_path = f'./result/{opt.dataset}.txt'

    data = np.load(data_path, allow_pickle=False).astype('float32')
    np.random.shuffle(data)

    # Standardscaler or norminalizer
    X = data[:, :-2]
    Y = data[:, -2:].astype(np.int64)

    #     scaler = Normalizer().fit(X)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    # data = torch.from_numpy(data)
    train_num = int(X.shape[0] * opt.train_eval_ratio)
    train_dataset = DataSet(X[:train_num], Y[:train_num])
    eval_dataset = DataSet(X[train_num:], Y[train_num:])
    print(
        f'train size:{len(train_dataset)}, evaluate size:{len(eval_dataset)}')

    # dataloader
    train_dataloader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=opt.batch_size * 10, shuffle=True, num_workers=opt.workers)

    # device : cuda or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create model
    _, _, layer_sizes = cfg.get_features_classes(opt)
    models = BaseModel(layer_sizes)

    models.to(device)

    # optimizer
    optimizers = torch.optim.Adam([{"params": models.trunk.parameters(), "lr": opt.lr},
                                   {"params": models.embedder.parameters(),
                                    "lr": opt.lr},
                                   {"params": models.classifier.parameters(), "lr": opt.lr}],
                                  weight_decay=opt.weight_decay)

    # classification loss
    classification_loss = torch.nn.CrossEntropyLoss()

    # metric loss & miner
    distance = LpDistance(power=2)
    miner = miners.MultiSimilarityMiner(epsilon=0.2)

    metric_loss = losses.AngularLoss(alpha=40)
    #     metric_loss = losses.MultiSimilarityLoss()
    #     metric_loss = losses.TripletMarginLoss(margin=0.1, distance=distance, reducer=AvgNonZeroReducer(), embedding_regularizer=LpRegularizer())
    #     metric_loss = losses.SphereFaceLoss(10, 2)
    #     metric_loss = losses.ProxyAnchorLoss(10, 2, margin = 0.1, alpha = 32)

    criterions = [classification_loss, metric_loss]

    acc = 0
    for epoch in range(opt.start_epoch, opt.epochs):
        reset_parameters(models)
        print('\nEpoch: [%d | %d] LR: %f\n' % (epoch + 1, opt.epochs, opt.lr))

        # train for one epoch
        loss, acc = train(epoch, train_dataloader, models,
                            criterions, miner, optimizers, device, opt)
        writer.add_scalar(f"loss/train/{opt.dataset}", loss, epoch)
        writer.add_scalar(f"acc/train/{opt.dataset}", loss, epoch)

        # evaluation
        cur_acc, report = validate(eval_dataloader, models, device, opt)
        
        # save model
        if cur_acc > acc:
            save(models, scaler, opt)
            with open(result_path, 'w') as f:
                f.write(report)

def train(epoch, train_loader, model, criterions, miner, optimizers, device, opt):
    losses = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    bar = Bar('Training', max=opt.val_iteration)

    # log item
    running_loss = 0.0
    running_corrects = 0.0
    total_input_size = 0.0

    model.train()
    since = time.time()
    for batch_idx in range(opt.val_iteration):
        try:
            inputs, target = next(train_loader)
        except:
            train_loader = iter(train_loader)
            inputs, target = next(train_loader)

        total_input_size += inputs.size(0)

        inputs = inputs.to(device)
        target = target.to(device)

        # update statistic info
        data_time.update(time.time() - since)

        embedding = model(inputs)
        output = model.classify(embedding)

        # loss
        classification_loss = criterions[0](output, target)
        pairs = miner(embedding, target)
        metric_loss = criterions[1](embedding, target, pairs)
        loss = classification_loss + 0.5 * metric_loss
        #         loss = metric_loss
        #         loss = classification_loss

        # compute gradient and SGD step
        optimizers.zero_grad()
        loss.backward()
        optimizers.step()

        # predict result
        running_loss += loss.item() * inputs.size(0)

        _, preds = torch.max(output.data, 1)
        running_corrects += torch.sum(preds == target.data)

        # update statistic information
        batch_time.update(time.time() - since)
        losses.update(loss.item(), inputs.size(0))
        since = time.time()

        # plot progress
        bar.suffix = 'Epoch: {epoch:3d} | ({batch}/{size}) | Data: {data:.3f}s | Batch: {bt:.3f}s  | Loss: {loss:.4f} | Acc: {acc:.3f} '.format(
                         batch=batch_idx + 1,
                         size=opt.val_iteration,
                         epoch=epoch,
                         data=data_time.avg,
                         bt=batch_time.avg,
                         loss=losses.avg,
                         acc=running_corrects / total_input_size,
                     )
        bar.next()
    bar.finish()

    # compute the average loss and accuracy
    epoch_loss = running_loss / total_input_size
    epoch_acc = float(running_corrects) / total_input_size
    return epoch_loss, epoch_acc


def reset_parameters(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.01, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def adjust_learning_rate(optimizer, epoch, opt):
    if (epoch + 1) % opt.adjust_epoch == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= opt.rate


def save(models, scaler, opt):
    model_path = f"./model/{opt.dataset}-model.pt"
    scaler_path = f"./model/{opt.dataset}-scaler.pkl"
    torch.save(models, model_path)
    joblib.dump(scaler, scaler_path)


def validate(test_loader, models, device, opt):
    models.to(device)
    models.eval()

    acc = 0.0
    total = 0.0
    label_true = []
    label_predict = []
    with torch.no_grad():
        for i, (inputs, target, _) in enumerate(test_loader):
            total += inputs.shape[0]
            inputs = inputs.to(device)
            target = target.to(device)

            trunk = models.trunk(inputs)
            embedding = models.embedder(trunk)
            output = models.classifier(embedding)

            # predict result
            _, preds = torch.max(output.data, 1)
            acc += torch.sum(preds == target.data)

            label_true.append(target.cpu().detach())
            label_predict.append(preds.cpu().detach())
    acc = acc / total
    label_true = torch.vstack(label_true).view(-1, 1).numpy()
    label_predict = torch.vstack(label_predict).view(-1,1).numpy()
    report = classification_report_imbalanced(label_true, label_predict, digits=4, output_dict=False, zero_division=1)
    return acc, report


def stream_test(opt):
    model_path = f"./model/{opt.dataset}-model.pt"
    model = torch.load(model_path)
    model.eval()

    scaler_path = f"./model/{opt.dataset}-scaler.pkl"
    scaler = joblib.load(scaler_path)

    # transform x from original space to embedding space
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get the embeded initial data
    path = f"data/eval/{opt.dataset}.npy"
    data = np.load(path).astype(np.float32)
    x = scaler.transform(data[:, :-2])
    y = data[:, -2:].astype(np.int64)
    dataset = DataSet(x, y)
    dataloader = DataLoader(
        dataset, batch_size=opt.batch_size * 10, shuffle=False, num_workers=2)

    nmi, recall, acc = validate(dataloader, model, device, opt)
    print(
        'Stream: Recall@1, 2, 4, 8: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; NMI: {nmi:.3f}; accuracy: {acc:.3f} \n'
        .format(recall=recall, nmi=nmi, acc=acc))


def transform(opt, init=True):
    # load data
    if init:
        data_path = f"./data/init/{opt.dataset}.npy"
        out_path = f"./data/init/trans/{opt.dataset}.csv"
    else:
        data_path = f"./data/eval/{opt.dataset}.npy"
        out_path = f"./data/eval/trans/{opt.dataset}.csv"

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
    dataloader = DataLoader(
        dataset, batch_size=opt.batch_size * 10, shuffle=True, num_workers=4)

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

    df = pd.DataFrame(embeddings)
    header = []
    for i in range(header.shape[0] - 1):
        header.append(f"f{i}")
    header.append("class")
    df.to_csv(out_path, index=False, header=header)


if __name__ == "__main__":
    dir = 'data/benchmark/realworld'
    datasets = ['spam', 'gas', 'covtypeNorm']

    opt = cfg.get_options()
    for dataset in datasets:
        opt.dataset = dataset
        opt.datatype = 'realworld'
        main(opt)
        transform(opt, init=False)
