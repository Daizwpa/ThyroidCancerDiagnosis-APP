
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd


def Get_barPlot(dataset: pd.DataFrame, column_name):
    counts = dataset[column_name].value_counts()
    courses = [str(v) for v in list(counts.index)]
    values = counts.tolist()

    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    fig.title = f'Bar plot of {column_name}'
    ax.bar(courses, values)
    addlabels(courses, values, ax)
    ax.set(xlabel="Vaules", ylabel="Counts")

    return fig, ax


def addlabels(x, y, ax):
    for i in range(len(x)):
        ax.text(i, y[i], y[i],
                horizontalalignment="center")


def Get_HistogramPlot(dataset: pd.DataFrame, column_name):
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    fig.title = f'Histogram of {column_name}'
    ax.hist(dataset[column_name], bins=20)
    return fig, ax


def Get_Density_Destributation(dataset, column_name):
    plt.style.use('ggplot')
    data = dataset[column_name]
    density = gaussian_kde(data)
    x_axs = np.linspace(data.min(), data.max(), 100)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    fig, ax = plt.subplots()
    ax.plot(x_axs, density(x_axs))
    fig.title = f'Desnity of {column_name}'

    return fig, ax


def Get_Box_Plot(dataset, column_name):
    plt.style.use('ggplot')
    data = dataset[column_name]
    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 7))

    plt.boxplot(data)
    # plot.boxplot(data)
    return fig
