import logging
import time

import torch
import numpy as np
from torch.utils.data import DataLoader

from pytorch_metric_learning import losses, miners

from sklearn.preprocessing import StandardScaler, Normalizer

from net.base import BaseModel
from utils.data import DataSet

import joblib
import utils.eval as eva
import config.config as cfg

opt = cfg.get_options()

from torch.utils.tensorboard import SummaryWriter

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
        loss = classification_loss + metric_loss
        
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
    print("Cost time: {time:.3f}s".format(time=stop-since))
    print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc
    
def adjust_learning_rate(optimizer, epoch, opt):
    # decayed lr by 10 every 20 epochs
    if (epoch+1)%20 == 0:
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
    nmi, recall = eva.evaluation(testdata.numpy(), testlabel.numpy(),[1,2,4,8])
    return nmi, recall, acc

def save(models, scaler, opt):
    model_path = f"./model/{opt.dataset}-model.pt"
    scaler_path = f"./model/{opt.dataset}-scaler.pkl"    
    torch.save(models, model_path)
    joblib.dump(scaler, scaler_path)

def main():
    # device : cuda or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # create model
    layer_sizes = [[2,4,8],[8,4],[4,4]]
    models = BaseModel(layer_sizes)
    
    models.to(device)
        
    # optimizer
    optimizers = torch.optim.Adam([{"params":models.trunk.parameters(),"lr":opt.lr},
                                  {"params":models.embedder.parameters(),"lr":opt.lr},
                                  {"params":models.classifier.parameters(),"lr":opt.lr}], weight_decay=opt.weight_decay)
    
    # classification loss
    classification_loss = torch.nn.CrossEntropyLoss()
    
    # metric loss & miner 
    metric_loss = losses.TripletMarginLoss(margin=0.1)
    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    
    criterions = [classification_loss, metric_loss]
    
    # load data
    data_path = f"./data/init/{opt.dataset}.npy"
    
    data = np.load(data_path, allow_pickle=False).astype('float32')
    np.random.shuffle(data)

    # Standardscaler or norminalizer
    X = data[:,:-2]
    Y = data[:,-2:].astype(np.int64)

    scaler = Normalizer().fit(X)
    X = scaler.transform(X)

    # data = torch.from_numpy(data)
    train_num = int(X.shape[0] * opt.train_eval_ratio)
    train_dataset = DataSet(X[:train_num], Y[:train_num])
    eval_dataset = DataSet(X[train_num:], Y[train_num:])
    print(len(train_dataset), len(eval_dataset))
    
    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    eval_dataloader = DataLoader(eval_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2,)
    
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
    print('Recall@1, 2, 4, 8: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; NMI: {nmi:.3f}; accuracy: {acc:.3f} \n'
                  .format(recall=recall, nmi=nmi, acc=acc))  
    save(models, scaler, opt)

if __name__=="__main__":
    main()
    print('finish')