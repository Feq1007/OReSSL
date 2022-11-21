import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

markers = list(MarkerStyle().markers.keys())[2:]

def plot_scaler(x, y, names, titles, xlabel, ylabel):
    """
    data is 3 dimensional: [axes, names, x, y]
    """
    scaler_num = len(names)
    axs_num = len(x)

    fig, axs = plt.subplots(1, axs_num, figsize=(6,3.5))
    axs[0].set_ylabel(ylabel)
    for i in range(axs_num):
        xi = x[i]
        for j in range(scaler_num):
            yj = y[i,j]
            axs[i].plot(xi, yj, label=names[j], marker=markers[j])
        axs[i].set_xlabel(xlabel)
        axs[i].set_title(titles[i])
        axs[i].legend(loc='best')
    plt.plot()
    plt.show()

def plot_scatter(x, y, sizes, colors, vmin=0, vmax=100):
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=sizes, c=colors, vmin=vmin, vmax=vmax)
    # ax.set(xlim=(),xticks=np.arange(),
    #        ylim=(),yticks=np.arange())
    plt.show()


def eval_plot(X, Y):
    X_std = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X_std)

    tsne_data = np.vstack((X_tsne.T, Y)).T

    df_tsne = pd.DataFrame(tsne_data, columns=['Dim1', 'Dim2', 'class'])
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2')

    plt.show()


if __name__=="__main__":
    # x = np.array([range(0, 120, 20) for i in range(2)])
    # y = np.array([[[90.87,89.39,87.26,84.28, 81.27, 76.94],[88.87,85.39,80.26,74.28, 68.27, 60.94],[30.87,28.39,23.26,16.28, 11.27, 9.94]],
    #      [[95.87,94.39,92.26,89.28, 86.27, 82.94],[94.87,93.39,90.26,88.28, 83.27, 76.94],[83.87,82.39,79.26,75.28, 73.27, 60.94]]])
    # print(y)
    # names = ["IR-SSL","ReSSL","SPASC"]
    # titles = ["CR4", "FG2C2D"]
    # xlabel = 'imbalance ratio'
    # ylabel = 'accuracy'
    # plot_scaler(x, y, names, titles, xlabel, ylabel)
    data = np.load('../data/init/4CRE-V2.npy')
    X = data[:,:-2]
    Y = data[:,-2].astype(np.int64)
    print(Y)
    # eval_plot(X, Y)
    plot_scatter(X[:,0],X[:,1],np.ones(len(X)), Y)
