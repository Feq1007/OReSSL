import random
import argparse
import numpy as np
import pandas as pd
from river import metrics
from river.preprocessing import StandardScaler
from river import stream
from river.imblearn import RandomSampler, RandomOverSampler, RandomUnderSampler
from imbslearn.mc_smote import SMOTESampler
from base.restream import ReStream
import datetime
import config.config as cfg 
from imblearn.metrics import classification_report_imbalanced
from torch.utils.tensorboard import SummaryWriter

def split_init(data: np.array, init_size=1000):
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values  # start from 0 and increase one by one
    classes = list(set(Y))
    Y = np.apply_along_axis(lambda x: classes.index(x),
                            1, Y.reshape((-1, 1))).flatten()
    classes = list(set(Y))

    x_init, y_init = [], []
    for y_temp in classes:
        x = X[Y == y_temp]
        y = Y[Y == y_temp]
        x_init.extend(x[:int(init_size / len(classes))])
        y_init.extend(y[:int(init_size / len(classes))])
    x_init, y_init = np.vstack(x_init), np.vstack(y_init)

    # features name
    feature_names = []
    for i in range(x_init.shape[1]):
        feature_names.append(f"f{i}")
    return x_init, y_init.reshape(-1), X, Y.reshape(-1), feature_names

def accept(label, distribution):
    if random.random() < distribution[label]:
        return True
    return False

def sampler_model(sample:str, classifier, ir=10):
    if sample == "random":
        return RandomSampler(classifier=classifier)
    elif sample == "over":
        return RandomOverSampler(classifier=classifier)
    elif sample == "under":
        return RandomUnderSampler(classifier=classifier)
    elif sample == "smote":
        return SMOTESampler(classifier=classifier, ir=ir)
    else:
        return classifier

if __name__ == "__main__":
    args = cfg.get_options(argparse.ArgumentParser())

    # Prepare for training
    samplers = ['no', 'smote']
    for sampler in samplers:
        modelname = 'ReStream'
        logpath = f'.runs/{datetime.datetime.now().strftime("%d-%H-%M")}-{args.dataset}-{modelname}-{sampler}'
        resultpath = f'result/imb-{sampler}-{args.imb_ratio}-{args.label_ratio}.txt'
        writer = SummaryWriter(logpath)

        # get ready data
        init_dataset, eval_dataset, true_init_size = None, None, 0
        if args.datatype == "realworld":
            datapath = f'data/benchmark/realworld/{args.dataset}/{args.dataset}.csv'
            data = pd.read_csv(datapath)

            # Split init and eval data
            init_x, init_y, eval_x, eval_y, feature_names = split_init(
                data=data, init_size=args.init_size)

            init_dataset = stream.iter_array(
                init_x, init_y, feature_names=feature_names, target_names='class')

            eval_dataset = stream.iter_array(
                eval_x, eval_y, feature_names=feature_names, target_names="class")
            
            true_init_size = init_x.shape[0]
        else:
            dataset = cfg.get_artificial_dataset(args.dataset)
            init_dataset = dataset.take(args.init_size)
            eval_dataset = dataset.take(args.eval_size)
            true_init_size = args.init_size

        # Prepare for training
        scaler = StandardScaler()
        parameters = cfg.get_proper_parameters(dataname=args.dataset)
        model = ReStream(n_samples_init=true_init_size, knn=parameters['knn'],
                         epsilon=parameters['epsilon'], re_threshold=parameters['re_threshold'])
        if args.imblearn:
            model = sampler_model(sampler, model)

        acc = metrics.Accuracy()
        kappa = metrics.CohenKappa()

        # Initialization
        for x, y in init_dataset:
            model.learn_one(x, y)

        # Evaluation
        y_true = []
        y_hat = []
        distribution = cfg.get_imb_distribution(args.dataset)
        for i, (x, y) in enumerate(eval_dataset):
            # make imbalance
            if not accept(y, distribution):
                continue

            # make semi-supervised
            y_semi = y if random.random() < args.label_ratio else -1

            # predict and learn
            x = scaler.learn_one(x).transform_one(x)
            y_pred = model.predict_proba_one(x)
            model.learn_one(x, y_semi, y_pred=y_pred)
            y_pred = max(y_pred, key=y_pred.get)

            # update metrics
            acc.update(y, y_pred)
            kappa.update(y, y_pred)
            y_true.append(y)
            y_hat.append(y_pred)

            writer.add_scalar(f"Acc/imb/{args.dataset}", acc.get(), i)
            writer.add_scalar(f"Kappa/imb/{args.dataset}", kappa.get(), i)

            if (i + 1) % 1000 == 0:
                print(model.describe())
                print(f"Acc:{acc.get():.2f}, kappa:{kappa.get():.2f}")
        writer.close()

        # Save result
        report = classification_report_imbalanced(y_true, y_hat, zero_division=True)
        with open(resultpath, 'w') as f:
            f.write(report)