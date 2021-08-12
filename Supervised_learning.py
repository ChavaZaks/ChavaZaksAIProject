import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import sys


def hot_vector(mushrooms):
    matrix = pd.DataFrame()
    size = np.shape(mushrooms)
    for i in range(size[1]):
        temp = pd.get_dummies(mushrooms[0:, i])
        matrix = pd.concat([matrix, temp], axis=1)
    return matrix


def data(file_name, column):
    with open(file_name, "r") as filestream:
        line = filestream.read().split()
        for i in range(0, len(line)):
            line[i] = line[i].split(",")
    matrix = np.array(line, dtype=str)
    colors = matrix[0:, 5]
    matrix = np.delete(matrix, column, 1)
    return matrix, colors


def colors_data(fifth_column):
    list = px.colors.qualitative.Plotly.copy()
    list.remove(list[9])
    #    list = [i for i in range(10)]
    line = np.unique(fifth_column)
    line = line.tolist()
    colors = []
    for value in fifth_column:
        where_value = line.index(value)
        color = list[where_value]
        colors.append(color)
    colors = np.array(colors)
    return colors


def prepare_models(data_set, colors):
    X_train, X_validation, Y_train, Y_validation = train_test_split(data_set, colors, test_size=0.20, random_state=2)
    models = []
    model = SVC(kernel='rbf', gamma=10 ** -2, C=10 ** -1 * 18)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    models.append(("Kernel SVM", predictions, Y_validation))
    model = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=10 ** -2 * 5)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    models.append(("Decision Tree", predictions, Y_validation))
    model = MLPClassifier(activation='relu', solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(6, 2), random_state=1,
                          max_iter=200)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    models.append(("neuron network", predictions, Y_validation))
    return models


def accuracies(models):
    for model in models:
        print("accuracy for ", model[0], "is:", accuracy_score(model[2], model[1]))


def confusion_classification(model):
    print("confusion matrix\n ", confusion_matrix(model[2], model[1]))
    print("classification_report\n ", classification_report(model[2], model[1]))


def preparing_data():
    mushrooms, colors = data("mushrooms_data.txt", 5)
    data_set = hot_vector(mushrooms)
    colors = colors_data(colors)
    return data_set, colors


def main():
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
    val = input("To see the accuracies please type 1 and to see confusion matrix and classification_report of"
                " a model type 2: ")
    data_set, colors = preparing_data()
    models = prepare_models(data_set, colors)
    if val == "1":
        accuracies(models)
    if val == "2":
        val_t = input("For Kernel SVM type 1, for Decision Tree type 2 and for Neural-Network type 3 ")
        if val_t == "1":
            confusion_classification(models[0])
        if val_t == "2":
            confusion_classification(models[1])
        if val_t == "3":
            confusion_classification(models[2])


if __name__ == "__main__":
    main()
