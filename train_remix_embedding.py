import time
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import LpDistance
from sklearn.preprocessing import StandardScaler, Normalizer
from torch.utils.data import DataLoader

import config.config as cfg
from utils.generate_data import compute_class_info
from base.model import BaseModel
from utils.misc import AverageMeter
from progress.bar import Bar
from base.augment import WeakAugment, StrongAugment
from base.data import DataSet, UnlabeledDataSet
from sklearn.metrics import classification_report

# ABC Train
def main_abc(args):
    print("==> creating dataloader")
    labeled_trainloader, unlabeled_trainloader, eval_dataloader, class_info, scaler = get_data(args)
    
    # prepare training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==> creating model")
    _, class_num, layer = cfg.get_features_classes(args)
    
    model = BaseModel(layer)
    model.to(device)
    params = list(model.parameters())

    ema_model = BaseModel(layer)
    ema_model.to(device)
    for param in list(ema_model.parameters()):
        param.detach_()

    # metric learning
    distance = LpDistance(power=2)
    miner = miners.MultiSimilarityMiner(epsilon=0.2)

    metric_loss = losses.AngularLoss(alpha=40)
    #     metric_loss = losses.MultiSimilarityLoss()
    #     metric_loss = losses.TripletMarginLoss(margin=0.1, distance=distance, reducer=AvgNonZeroReducer(), embedding_regularizer=LpRegularizer())
    #     metric_loss = losses.SphereFaceLoss(10, 2)
    #     metric_loss = losses.ProxyAnchorLoss(10, 2, margin = 0.1, alpha = 32)
    
    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params, lr=args.lr)

    ema_optimizer = WeightEMA(model, ema_model, lr=args.lr, alpha=args.ema_decay)

    # train
    target_disb = get_target_distribution(class_info, class_num, args.imb_ratio, args.imb_type)
    ir2 = torch.min(target_disb) / target_disb

    print(f"==> true distribution: {target_disb}, ir2: {ir2}")
    
    test_accs = []
    acc = 0
    emp_distb_u = int(0)
    for epoch in range(args.start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f\n' % (epoch + 1, args.epochs, args.lr))
        train_loss, train_loss_x, train_loss_u, abcloss = train_imbalance(labeled_trainloader,
                                                                                       unlabeled_trainloader,
                                                                                       model, optimizer,
                                                                                       ema_optimizer, train_criterion,
                                                                                       epoch, ir2,
                                                                                       emp_distb_u, target_disb,
                                                                                       class_num, device, miner, metric_loss, args)
        test_loss, test_acc, testclassacc, rep = validate_imbalance(eval_dataloader, ema_model, criterion, mode='Test Stats ', class_num=class_num, device=device)
        if rep['accuracy'] > acc:
            save(model, scaler, opt)
            acc = rep['accuracy']      
        print("each class accuracy test", testclassacc, testclassacc.mean())


def train_imbalance(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, ir2,
                    emp_distb_u, target_disb, class_num, device, miner, metric_loss, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    losses_r = AverageMeter()
    losses_e = AverageMeter()
    losses_abc = AverageMeter()

    ws = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x, _ = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, _ = next(labeled_train_iter)

        try:
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = next(unlabeled_train_iter)

        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x2 = torch.zeros(batch_size, class_num).scatter_(1, targets_x.view(-1, 1), 1)
        
        inputs_x, targets_x, targets_x2 = inputs_x.to(device), targets_x.to(device), targets_x2.to(device)
        inputs_u, inputs_u2, inputs_u3 = inputs_u.to(device), inputs_u2.to(device), inputs_u3.to(device)

        # Generate the pseudo labels
        with torch.no_grad():
            q1 = model(inputs_u)
            outputs_u = model.classify(q1)
            p = torch.softmax(outputs_u, dim=1).detach()
            
            # Tracking the empirical distribution on the unlabeled samples (ReMixMatch)
            real_batch_idx = batch_idx + epoch * args.val_iteration
            if type(emp_distb_u)==int:
                emp_distb_u = p.mean(0, keepdim=True)
            elif real_batch_idx == 0:
                emp_distb_u = p.mean(0, keepdim=True)
            elif real_batch_idx // 128 == 0:
                emp_distb_u = torch.cat([emp_distb_u, p.mean(0, keepdim=True)], 0)
            else:
                emp_distb_u = emp_distb_u[-127:]
                emp_distb_u = torch.cat([emp_distb_u, p.mean(0, keepdim=True)], 0)

            # Distribution alignment
            if args.align:
                pa = p * (target_disb.to(device) + 1e-6) / (emp_distb_u.mean(0).to(device) + 1e-6)
                p = pa / pa.sum(dim=1, keepdim=True)

            # Temperature scailing
            pt = p ** (1 / args.T)
            targets_u2 = (pt / pt.sum(dim=1, keepdim=True)).detach()            
            
        q = model(inputs_x)
        q2 = model(inputs_u2)
        q3 = model(inputs_u3)
        
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2, inputs_u3], dim=0)
        all_targets = torch.cat([targets_x2, targets_u2, targets_u2, targets_u2], dim=0)
        
        l = np.random.beta(args.mix_alpha, args.mix_alpha)
        l = max(l, 1-l)
        idx = torch.randperm(all_inputs.size(0))
        
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b
        
        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)
        
        logits = [model.classify(model(mixed_input[0]))]
        for input in mixed_input[1:]:
            logits.append(model.classify(model(input)))
            
        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)
        
        #abc
        ir22 = 1 - (epoch / args.epochs) * (1 - ir2)
        maskforbalance = torch.bernoulli(torch.sum(targets_x2 * ir2.clone().to(device),dim=1).detach())
#         maskforbalance = torch.bernoulli(torch.sum(targets_x2 * torch.tensor(ir2).cuda(0), dim=1).detach())
        
        embd = model.abc_embedding(q)
        embd_u = model.abc_embedder(q1)
        embd_u2 = model.abc_embedder(q2)
        embd_u3 = model.abc_embedder(q3)

        logit = model.abc_classify(embd)
        logitu = model.abc_classify(embd_u)
        logitu2 = model.abc_classify(embd_u2)
        logitu3 = model.abc_classify(embd_u3)

        logits = nn.functional.softmax(logit, dim=-1)
        logitsu1 = nn.functional.softmax(logitu, dim=-1)
        max_p2, label_u = torch.max(logitsu1, dim=1)
        select_mask2 = max_p2.ge(0.95)
        label_u = torch.zeros(batch_size, class_num).scatter_(1, label_u.cpu().view(-1, 1), 1)
        
        maskforbalanceu= torch.bernoulli(torch.sum(label_u.to(device) * ir22.to(device), dim=1).detach())

        logitsu2 = nn.functional.softmax(logitu2, dim=-1)
        logitsu3 = nn.functional.softmax(logitu3, dim=-1)

        abcloss = -torch.mean(maskforbalance * torch.sum(torch.log(logits) * targets_x2.to(device), dim=1))
        abcloss1 = -torch.mean(
            select_mask2 * maskforbalanceu * torch.sum(torch.log(logitsu2) * logitsu1.to(device).detach(), dim=1))
        abcloss2 = -torch.mean(
            select_mask2 * maskforbalanceu * torch.sum(torch.log(logitsu3) * logitsu1.to(device).detach(), dim=1))
        
        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:],
                              epoch + batch_idx / args.val_iteration, args)
        
        totalabcloss = abcloss + abcloss1 + abcloss2
        
        outputs_u2 = model.abc_classify(model.abc_embedder(q2))
        Le = -1 * torch.mean(torch.sum(F.log_softmax(outputs_u2, dim=1) * targets_u2, dim=1))
        
        loss = Lx + w * Lu + args.w_ent * Le * linear_rampup(epoch+batch_idx/args.val_iteration, args.epochs) + totalabcloss
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        losses_abc.update(abcloss.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ' \
                     'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | ' \
                     ' Loss_m: {loss_m:.4f}'.format(
            batch=batch_idx + 1,
            size=args.val_iteration,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            loss=losses.avg,
            loss_x=losses_x.avg,
            loss_u=losses_u.avg,
            loss_m=losses_abc.avg,
        )
        bar.next()
    bar.finish()
    
    return (losses.avg, losses_x.avg, losses_u.avg, losses_abc.avg)

def validate_imbalance(val_loader, model, criterion, mode, class_num, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    
    # switch to evaluate mode
    model.eval()

    accperclass = np.zeros((class_num))

    end = time.time()
    bar = Bar(f'{mode}', max=len(val_loader))

    label_true = []
    label_predict = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(val_loader):
            targets = targets.long()
            
            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # compute output
            targetsonehot = torch.zeros(inputs.size()[0], class_num).scatter_(1, targets.cpu().view(-1, 1).long(), 1)
            q = model(inputs)
            outputs2 = model.abc_classify(model.abc_embedder(q))

            unbiasedscore = nn.functional.softmax(outputs2, dim=-1)

            unbiased = torch.argmax(unbiasedscore, dim=1)
            
            label_true.append(targets.cpu().detach())
            label_predict.append(unbiased.cpu().detach())
            
            outputs2onehot = torch.zeros(inputs.size()[0], class_num).scatter_(1, unbiased.cpu().view(-1, 1).long(), 1)
            loss = criterion(outputs2, targets)
            accperclass = accperclass + torch.sum(targetsonehot * outputs2onehot, dim=0).cpu().detach().numpy().astype(
                np.int64)

            # measure accuracy and record loss
            losses.update(loss.item(), inputs.size(0))
            # top1.update(prec1.item(), inputs.size(0))
#             top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                         'Loss: {loss:.4f}'.format(
                batch=batch_idx + 1,
                size=len(val_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                # top1=top1.avg,
#                 top5=top5.avg,
            )
            bar.next()
        bar.finish()
    label_true = torch.vstack(label_true).view(-1,1).numpy()
    label_predict = torch.vstack(label_predict).view(-1,1).numpy()
    print(classification_report(label_true, label_predict, digits=2, zero_division=1))
    rep = classification_report(label_true, label_predict, digits=4, output_dict=True, zero_division=1)
    return (losses.avg, top1.avg, accperclass, rep)

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, args):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1))

        return Lx, Lu, args.lambda_u * linear_rampup(epoch, args.epochs)


class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.2 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.float()
            param = param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            param.mul_(1 - self.wd)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]



def make_proxy(X, Y, opt):
    # X, Y‘, fake2true:从子标签到真实标签的映射
    true_label = Y[:,0]
    semi_label = Y[:, -1]
    classes = set(true_label)
    sub2true = {}

    new_label = 0
    new_data = []

    for cls in classes:
        index = (true_label == cls)
        data = X[index]

        model = DBSCAN()
        model.fit(data)

        sub_labels = model.labels_
        sub_classes = set(sub_labels)

        for sub_cls in sub_classes:
            sub_idx = (sub_labels == sub_cls)

            sub_true_label = np.expand_dims(true_label[index][sub_idx], axis=1)
            sub_semi_label = np.expand_dims(semi_label[index][sub_idx], axis=1)
            sub_label = np.ones(sub_true_label.shape) * new_label

            sub_data = np.hstack([data[sub_idx], sub_true_label, sub_label, sub_semi_label])
            new_data.append(sub_data)
            sub2true[new_label] = cls
            new_label += 1
    new_data = np.vstack(new_data)
    return new_data[:,:-3], new_data[:,-3:], sub2true

def get_data(args):
    # prepare data
    data_path = f"./data/init/{args.dataset}.npy"

    data = np.load(data_path, allow_pickle=False).astype('float32')
    np.random.shuffle(data)

    # split data
    train_num = int(data.shape[0] * args.train_eval_ratio)
    train_data = data[:train_num]
    eval_data = data[train_num:]

    train_x = train_data[:, :-2]
    train_y = train_data[:, -2:].astype(np.int64)
    
    eval_x = eval_data[:, :-2]
    eval_y = eval_data[:, -2:]

#     train_x, train_y = make_proxy(train_x, train_y, args)
    
    # Standardizer or Normalizer
    scaler = Normalizer().fit(train_x)
    # scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    eval_x = scaler.transform(eval_x)

    labeled_train_x = train_x[train_y[:, -1] != -1]
    labeled_train_y = train_y[train_y[:, -1] != -1]

    unlabeled_train_x = train_x[train_y[:, -1] == -1]
    unlabeled_train_y = train_y[train_y[:, -1] == -1]
    
    # dataset
    labeled_train_dataset = DataSet(labeled_train_x, labeled_train_y)
    
    weak = WeakAugment(train_data)
    strong = StrongAugment(train_data)
    unlabeled_train_dataset = UnlabeledDataSet(unlabeled_train_x, unlabeled_train_y, weak, strong)
    
    eval_dataset = DataSet(eval_x, eval_y)
    print(
        f'labeled train size:{len(labeled_train_dataset)}, unlabeled train size: {len(unlabeled_train_dataset)}, evaluate size:{len(eval_dataset)}')

    # dataloader
    labeled_trainloader = DataLoader(labeled_train_dataset, batch_size=opt.batch_size, shuffle=True,
                                     num_workers=opt.workers, drop_last=True)
    unlabeled_trainloader = DataLoader(unlabeled_train_dataset, batch_size=opt.batch_size, shuffle=True,
                                       num_workers=opt.workers,drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=opt.batch_size * 10, shuffle=True, num_workers=opt.workers)
    
    # class statistic information
    class_info = compute_class_info(train_y[:, -1])
    
    return labeled_trainloader, unlabeled_trainloader, eval_dataloader, class_info, scaler


def save(model, scaler, opt):
    model_path = f"./model/{opt.dataset}-model.pt"
    scaler_path = f"./model/{opt.dataset}-scaler.pkl"
    torch.save(model, model_path)
    joblib.dump(scaler, scaler_path)

    
def get_target_distribution(class_info, class_num, ir, imb_type):
    # assume the unlabeled data are more than the labeled one
    disb = make_imb_data(1, class_num, ir, imb_type)
    disb = np.array(disb)
    disb = disb / np.sum(disb)

    class_info = class_info.sort_values(by='number', ascending=False)

    target_disb = np.zeros(class_num)
    print(disb, class_info, target_disb)
    for i, idx in enumerate(class_info['classes']):
        if idx == -1:
            continue
        target_disb[idx] = disb[i - 1]
    
    
    
    zeros = len(target_disb[target_disb == 0])
    if zeros != 0:
        avg = (1 - np.sum(target_disb)) / zeros
        target_disb[target_disb == 0] = avg
    return torch.tensor(target_disb)


def make_imb_data(max_num, class_num, gamma, imb):
    class_num_list = []
    if imb == 'long':
        mu = np.power(1 / gamma, 1 / (class_num - 1))
        for i in range(class_num):
            if i == (class_num - 1):
                class_num_list.append(max_num / gamma)
            else:
                class_num_list.append(max_num * np.power(mu, i))
    if imb == 'step':
        for i in range(class_num):
            if i < int((class_num) / 2):
                class_num_list.append(max_num)
        else:
            class_num_list.append(max_num / gamma)
    return class_num_list

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
    dataloader = DataLoader(dataset, batch_size=opt.batch_size * 10, shuffle=True, num_workers=4)

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
    main_abc(opt)
    transform(opt)
    print('finish')
#     test(opt)
