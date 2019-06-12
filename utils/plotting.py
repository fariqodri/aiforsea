import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import seaborn as sns


def plot_difference_line(x_axis, y_axises, legend):
    for y in y_axises:
        plt.plot(x_axis, y)
    plt.legend(legend)


def plot_silhouette_scores_of_clusters(data, cluster_range, figsize):
    try_models = [KMeans(n_clusters=i, random_state=1, tol=1e-8)
                  for i in cluster_range]
    scores = []
    plt.figure(figsize=figsize)

    for m in try_models:
        m.fit(data)
        scores.append(silhouette_score(
            data, m.labels_))

    plt.plot(cluster_range, scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.grid(True)

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig
