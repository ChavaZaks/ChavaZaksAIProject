import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def data(file_name, column, bool):
    with open(file_name, "r") as filestream:
        line = filestream.read().split()
        for i in range(0, len(line)):
            line[i] = line[i].split(",")
    matrix = np.array(line, dtype=str)
    colors = matrix[0:, 5]
    if bool != True:
        matrix = np.delete(matrix, column, 1)
        return matrix, colors
    return matrix


def hot_vector(mushrooms):
    matrix = pd.DataFrame()
    size = np.shape(mushrooms)
    for i in range(size[1]):
        temp = pd.get_dummies(mushrooms[0:, i])
        matrix = pd.concat([matrix, temp], axis=1)
    return matrix


def colors_data(fifth_column):
    list = px.colors.qualitative.Plotly.copy()
    list.remove(list[9])
    line = np.unique(fifth_column)
    line = line.tolist()
    colors = []
    for value in fifth_column:
        where_value = line.index(value)
        color = list[where_value]
        colors.append(color)
    colors = np.array(colors)
    return colors


def make_into_colors(list_of_colors):
    list = px.colors.qualitative.Plotly.copy()
    list.remove(list[9])
    tupels = []
    for i in range(len(list)):
        tupels.append((i, list[i]))
    colors = []
    for i in list_of_colors:
        tuple = tupels[i]
        colors.append(tuple[1])
    return colors


def plot_clusters(plot_columns, plot_true_columns, models):
    fig = make_subplots(rows=2, cols=2, start_cell="top-left",
                        subplot_titles=("Real Data", "K-means", "Agglomerative Clustering", "Spectral Clustering"))

    fig.add_trace(go.Scatter(x=plot_true_columns[:, 1],
                             y=plot_true_columns[:, 0], mode="markers", marker=dict(size=5, color=models[0])), row=1,
                  col=1)
    fig.update_traces(marker=dict(size=5, line=dict(width=1,
                                                    color='DarkSlateGrey')), selector=dict(mode='markers'))
    fig.add_trace(go.Scatter(x=plot_columns[:, 1],
                             y=plot_columns[:, 0], mode="markers", marker=dict(size=5, color=models[1])), row=1,
                  col=2)
    fig.update_traces(marker=dict(size=5, line=dict(width=1,
                                                    color='DarkSlateGrey')), selector=dict(mode='markers'))
    fig.add_trace(go.Scatter(x=plot_columns[:, 1],
                             y=plot_columns[:, 0], mode="markers", marker=dict(size=5, color=models[2])), row=2,
                  col=1)
    fig.update_traces(marker=dict(size=5, line=dict(width=1,
                                                    color='DarkSlateGrey')), selector=dict(mode='markers'))
    fig.add_trace(go.Scatter(x=plot_columns[:, 1],
                             y=plot_columns[:, 0], mode="markers", marker=dict(size=5, color=models[3])), row=2,
                  col=2)
    fig.update_traces(marker=dict(size=5, line=dict(width=1,
                                                    color='DarkSlateGrey')), selector=dict(mode='markers'))
    # Update xaxis properties
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=2)
    fig.update_xaxes(title_text="x", row=2, col=1)
    fig.update_xaxes(title_text="x", row=2, col=2)

    # Update yaxis properties
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=2)
    fig.update_yaxes(title_text="y", row=2, col=1)
    fig.update_yaxes(title_text="y", row=2, col=2)
    fig.update_layout(title_text="Data and Clustered data", height=700, showlegend=False)
    fig.show()
    return


def silhouettes_scores(plot_columns, models):
    print("silhouette score for K-menas")
    print(silhouette_score(plot_columns, labels=models[1]))
    print("silhouette score for AgglomerativeClustering")
    print(silhouette_score(plot_columns, labels=models[2]))
    print("silhouette score for SpectralClustering")
    print(silhouette_score(plot_columns, labels=models[3]))


def preparing_data():
    mushrooms, colors = data("mushrooms_data.txt", 5, False)
    true_mushrooms = data("mushrooms_data.txt", 5, True)
    colors = colors_data(colors)
    one_hot_array = hot_vector(mushrooms)
    one_hot_array_true = hot_vector(true_mushrooms)
    pca = PCA(2)
    plot_columns = pca.fit_transform(one_hot_array)
    plot_true_columns = pca.fit_transform(one_hot_array_true)
    models = []
    models.append(colors)
    kmeans = KMeans(n_clusters=9, init='k-means++', n_init=10, verbose=0)
    kmeans_clusters = kmeans.fit_predict(plot_columns)
    clusters_kme = make_into_colors(kmeans_clusters)
    models.append(clusters_kme)

    algoclu = AgglomerativeClustering(n_clusters=9)
    algoclu_kmodes = algoclu.fit_predict(plot_columns)
    clusters_algoclu = make_into_colors(algoclu_kmodes)
    models.append(clusters_algoclu)

    spectral_clustering = SpectralClustering(n_clusters=9, affinity='rbf', assign_labels='discretize')
    clusters_spectal = spectral_clustering.fit_predict(plot_columns)
    clusters_sc = make_into_colors(clusters_spectal)
    models.append(clusters_sc)
    return plot_columns, plot_true_columns, models


def main():
    val = input("To see silhouette_scores please type 1, to see the graphs type 2: ")
    print()
    plot_columns, plot_true_columns, models = preparing_data()
    if val == "1":
        silhouettes_scores(plot_columns, models)
    if val == "2":
        plot_clusters(plot_columns, plot_true_columns, models)


if __name__ == "__main__":
    main()
