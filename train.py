import warnings
warnings.filterwarnings("ignore")

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from joblib import dump

from utils.load import load_features_group, load_label_df
from config.constants import *
from utils.preprocessing import kalman_filter
from utils.driving_event import DrivingEventDetection
from utils.dangerous_driving import DangerousDrivingDetection


dtypes_dict = {'bookingID': str}


def get_longest_trip(group):
    longest_trip_id = group.size().sort_values(
        ascending=False).head(1).index.values[0]
    driving_event_features = group.get_group(longest_trip_id)
    return driving_event_features


def train_k_means(features_group):
    longest_trip = get_longest_trip(features_group)
    driving_event = DrivingEventDetection(longest_trip)
    driving_event.select_features(["acceleration_y", "gyro_z", "second"])

    print(" " * 3, "Preprocessing (Kalman Filter)")
    driving_event.pre_process(
        "acceleration_y", lambda x: kalman_filter(x, "acceleration_y"))
    driving_event.pre_process("gyro_z", lambda x: kalman_filter(x, "gyro_z"))
    print(" " * 3, "Preprocessing Done")

    clustering_features = driving_event.load_clustering_features(
        ["acceleration_y", "gyro_z"]).dropna()

    driving_event.set_data(clustering_features)

    driving_event.scale_data(StandardScaler())
    km = KMeans(n_clusters=N_CLUSTERS, random_state=1, tol=1e-8)
    km = driving_event.train(km)
    return km


def upsample(x, y, ratio):
    sm = SMOTE(random_state=42, ratio=ratio)
    x, y = sm.fit_sample(x, y)
    return x, y


def train_random_forest(features_group, label_df, k_means_model):
    danger_driving = DangerousDrivingDetection(features_group)

    print(" " * 3, "Counting Events and Speeding per Trip")
    classification_features = danger_driving.load_classification_features(
        k_means_model)
    print(" " * 3, "Done")

    danger_driving.set_data(classification_features)
    danger_driving.merge_label_to_data(label_df, on="bookingID")
    danger_driving.drop_column("bookingID")

    x, y = danger_driving.split_feature_and_label("label")
    x, y = upsample(x, y, 1.0)
    mms = MinMaxScaler()
    x = mms.fit_transform(x)

    rf = RandomForestClassifier(max_depth=2, n_estimators=100, random_state=42)
    rf.fit(x, y)
    return rf


if __name__ == "__main__":
    label = load_label_df(LABEL_PATH)
    features = load_features_group(FEATURES_PATHS)

    print()
    print("Training K-Means Model...")
    km = train_k_means(features)

    print()
    print("Training Random Forest Classifier Model...")
    rf = train_random_forest(features, label, km)

    dump(km, 'models/k_means.joblib')
    dump(rf, 'models/random_forest_classifier.joblib')
    print()
    print("K-Means and Random Forest Classifier models has been saved at folder models")
