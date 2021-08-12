import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import prince
from sklearn.metrics import accuracy_score
import sys
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import FastICA
from sklearn import manifold
import umap.umap_ as umap


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


def accuracies(models):
    for model in models:
        print("accuracy for ", model[0], "is:", accuracy_score(model[2], model[1]))


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
    fig.update_layout(title_text="MCA: Data and Clustered data", height=700, showlegend=False)
    fig.show()
    return


def silhouettes_scores(plot_columns, models):
    print("For K-menas: ", silhouette_score(plot_columns, labels=models[1]))
    print("For AgglomerativeClustering: ", silhouette_score(plot_columns, labels=models[2]))
    print("For SpectralClustering: ", silhouette_score(plot_columns, labels=models[3]))


def preparing_data_clustering(dr):
    mushrooms, colors = data("mushrooms_data.txt", 5, False)
    true_mushrooms = data("mushrooms_data.txt", 5, True)
    colors = colors_data(colors)
    one_hot_array = hot_vector(mushrooms)
    one_hot_array_true = hot_vector(true_mushrooms)
    plot_columns = None
    plot_true_columns = None
    if dr == "pca":
        pca = PCA(2)
        plot_columns = pca.fit_transform(one_hot_array)
        plot_true_columns = pca.fit_transform(one_hot_array_true)
    if dr == "mca":
        mca = prince.MCA(2)
        plot_columns = mca.fit_transform(one_hot_array)
        plot_columns = plot_columns.to_numpy()
        plot_true_columns = mca.fit_transform(one_hot_array_true)
        plot_true_columns = plot_true_columns.to_numpy()
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


def preparing_data_supervised_learning():
    mushrooms, colors = data("mushrooms_data.txt", 5, False)
    data_set = hot_vector(mushrooms)
    colors = colors_data(colors)
    return data_set, colors


def prepare_models(data_set, colors, dr_sl):
    if dr_sl == "svm":
        umap_data = umap.UMAP().fit_transform(data_set)
        X_train, X_validation, Y_train, Y_validation = train_test_split(umap_data, colors, test_size=0.20,
                                                                        random_state=2)
        model = SVC(kernel='rbf', gamma=10 ** -2, C=10 ** -1 * 18)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        print("accuracy for Kernel SVM is:", accuracy_score(predictions, Y_validation))
    if dr_sl == "dt":
        ica = FastICA(algorithm='deflation', fun='cube').fit_transform(data_set)
        X_train, X_validation, Y_train, Y_validation = train_test_split(ica, colors, test_size=0.20,
                                                                        random_state=2)
        model = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=10 ** -2 * 5)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        print("accuracy for Decision Tree is:", accuracy_score(predictions, Y_validation))
    if dr_sl == "ae":
        iso = manifold.Isomap(n_neighbors=6, n_components=101, eigen_solver='dense').fit_transform(data_set)
        X_train, X_validation, Y_train, Y_validation = train_test_split(iso, colors, test_size=0.20,
                                                                        random_state=2)
        model = MLPClassifier(activation='relu', solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(6, 2), random_state=1,
                              max_iter=200)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        print("accuracy for neuron network:", accuracy_score(predictions, Y_validation))


def main():
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
    val = input("To see dimensionality reduction for clustering type 1,"
                " for reduction for Supervised learning type 2: ")
    if val == "1":
        val_cluster = input("To see pca write 1 and to see MCA type 2: ")
        val_is = input("To see silhouette scores write 1 and to see the graphs type 2: ")
        print()
        if val_cluster == "1":
            plot_columns, plot_true_columns, models = preparing_data_clustering("pca")
            if val_is == "1":
                silhouettes_scores(plot_columns, models)
            if val_is == "2":
                plot_clusters(plot_columns, plot_true_columns, models)
        if val_cluster == "2":
            plot_columns, plot_true_columns, models = preparing_data_clustering("mca")
            if val_is == "1":
                silhouettes_scores(plot_columns, models)
            if val_is == "2":
                plot_clusters(plot_columns, plot_true_columns, models)
    if val == "2":
        val_learning = input("For Kernel SVM type 1, for Decision Tree type 2 and for Neural-Network type 3 ")
        print()
        data_set, colors = preparing_data_supervised_learning()
        if val_learning == "1":
            prepare_models(data_set, colors, "svm")
        if val_learning == "2":
            prepare_models(data_set, colors, "dt")
        if val_learning == "3":
            prepare_models(data_set, colors, "ae")


if __name__ == "__main__":
    main()
