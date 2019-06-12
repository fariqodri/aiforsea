from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def get_variance_of_window(dataframe, column, window_size):
    return dataframe[column].rolling(window_size).var()


def get_mean_of_window(dataframe, columns, window_size):
    return dataframe[columns].rolling(window_size).mean()


def count_events(groups, columns, window_size, n_clusters, k_means):
    features = {'bookingID': [], **{i: [] for i in range(n_clusters)}}
    for name, group in groups:
        window_var = get_variance_of_window(
            group, columns, window_size).dropna()
        ss = StandardScaler()
        cluster_data = ss.fit_transform(window_var)
        events = k_means.predict(cluster_data)

        labels, counts = np.unique(events, return_counts=True)
        features = add_all_num_of_events_to_dict(
            features, name, labels, counts)
    return pd.DataFrame.from_dict(features)


def add_all_num_of_events_to_dict(features, group_name, labels, counts):
    features_dict = {'bookingID': group_name, **dict(zip(labels, counts))}
    for k in features:
        if k not in features_dict:
            features_dict[k] = 0
        features[k].append(features_dict[k])
    return features


def count_speeding(groups, window_size, speed_limit):
    features = {'bookingID': [], 'num_of_speeding': []}
    for name, group in groups:
        speed_window_mean = get_mean_of_window(
            group, ["Speed"], window_size).dropna()
        features['bookingID'].append(name)
        features['num_of_speeding'].append(len(
            speed_window_mean.loc[speed_window_mean['Speed'] > speed_limit, :]))
    return pd.DataFrame.from_dict(features)
