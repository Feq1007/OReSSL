import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

import config.config as cfg


def generate_fake_label(data, class_num):
    x = data[:, :-1]
    y = data[:, -1].astype(np.int32)

    res = []
    classes = set(y)
    for i, cls in enumerate(classes):
        index = (y == cls)
        _data = x[index]
        if len(_data) > class_num:
            kmeans = KMeans(n_clusters=class_num)
            kmeans.fit(_data)
            kmeans_labels = kmeans.labels_
            for _cls in range(class_num):
                _data_cls = _data[kmeans_labels == _cls]
                if len(_data_cls) == 0:
                    continue
                fake_cls = np.ones(len(_data_cls)) * (i * class_num + _cls)
                fake_cls = fake_cls.reshape((-1, 1))
                new_data = np.hstack([_data_cls, fake_cls])
                res.append(new_data)
        else:
            for j, d in enumerate(_data):
                res.append(np.hstack([d, np.array([j])]))
    res = np.vstack(res)
    res = pd.DataFrame(res)
    return res


def compute_class_info(labels):
    """
    return the statistic information about class information
    """
    classes = list(set(labels))
    class_num_true_list = []
    for c in classes:
        class_num_true_list.append(len(np.where(labels == c)[0]))
    class_info = np.array([classes, class_num_true_list], dtype=int).transpose()
    class_info = pd.DataFrame(class_info, columns=['classes', 'number'])
    class_info = class_info.sort_values(by='number')
    return class_info


def compute_imb_class_info(class_info, imb_ratio, imb_type):
    """
    description: according to the label and imbalance ratio to compute the number of each class in initial set and evaluate set
    """
    class_num_list = []
    class_num = class_info.shape[0]
    max_num = class_info.iloc[-1][-1]
    gamma = imb_ratio
    if imb_type == 'long':
        mu = np.power(1 / gamma, 1 / (class_num - 1))
        for i in range(class_num):
            if i == (class_num - 1):
                class_num_list.append(int(max_num / gamma))
            else:
                class_num_list.append(int(max_num * np.power(mu, i)))
    elif imb_type == 'step':
        for i in range(class_num):
            if i < int((class_num) / 2):
                class_num_list.append(int(max_num))
            else:
                class_num_list.append(int(max_num / gamma))

    list.reverse(class_num_list)
    for i in range(class_num):
        if class_num_list[i] < class_info.iloc[i, -1]:
            class_info.iloc[i, -1] = class_num_list[i]
    return class_info


def sample(labels, class_info, init_size):
    """
    split the data into train and evaluate data
    """
    init_nums = np.array(class_info.iloc[:, -1], dtype=float) / np.sum(
        class_info.iloc[:, -1]) * init_size + 1  # plus one: make sure every class are included
    init_nums = init_nums.astype(int)

    init_idxs = []
    eval_idxs = []
    for i in range(class_info.shape[0]):
        label = class_info.iloc[i][0]
        idxs = np.where(labels == label)[0]  # return the index of data
        np.random.shuffle(idxs)
        idxs = idxs[:class_info.iloc[i, 1]]
        idxs = np.sort(idxs)
        init_idxs.extend(idxs[:init_nums[i]])
        eval_idxs.extend(idxs[init_nums[i]:])
    return np.sort(init_idxs), np.sort(eval_idxs)


def split_init_data(data_path, init_size, label_ratio, imb_ratio, imb_type='long'):
    data = pd.read_csv(data_path, header=None, dtype=float)
    # data['synclasses'] = data[data.columns[-1]].apply(lambda x: 3 * x + random.randint(0, 3))
    # data = generate_fake_label(data.values, 3)

    # encode label
    labels = np.array(data.iloc[:, -1], dtype=int)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)

    class_info = compute_class_info(labels)
    print('original class information: \n', class_info)

    imb_class_info = compute_imb_class_info(class_info, imb_ratio, imb_type)
    print('imbalanced class information: \n', imb_class_info)

    init_idxs, eval_idxs = sample(labels, imb_class_info, init_size)
    print('initial data information: \n', compute_class_info(labels[init_idxs]))
    print('stream data information: \n', compute_class_info(labels[eval_idxs]))

    # random mask
    init_mask = np.random.choice(init_idxs, size=int(init_size * (100.0 - label_ratio) / 100), replace=False)
    eval_mask = np.random.choice(eval_idxs, size=int(len(eval_idxs) * (100.0 - label_ratio) / 100), replace=False)

    # encode labels
    semi_labels = np.copy(labels)
    semi_labels[init_mask] = -1  # some minority class maybe set to -1, pay attention to that
    semi_labels[eval_mask] = -1
    data = np.hstack([data.values[:, :-1], labels.reshape((-1, 1)), semi_labels.reshape((-1, 1))])

    init_data = data[init_idxs]
    eval_data = data[eval_idxs]
    print('masked initial data information:', compute_class_info(init_data[:, -1]))
    print('masked stream data information:', compute_class_info(eval_data[:, -1]))

    print(r"save data to path: ./data/{init,eval}...")
    np.save(f'./data/init/{opt.dataset}.npy', init_data, allow_pickle=False)
    np.save(f'./data/eval/{opt.dataset}', eval_data, allow_pickle=False)


if __name__ == "__main__":
    opt = cfg.get_options()
    data_path = f"data/benchmark/{opt.datatype}/{opt.dataset}.csv"
    split_init_data(data_path, opt.init_size, opt.label_ratio, opt.imb_ratio, opt.imb_type)
