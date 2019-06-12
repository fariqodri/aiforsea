from joblib import dump, load
import sys
import pandas as pd
from utils.dangerous_driving import DangerousDrivingDetection
from utils.extraction import count_events, count_speeding

WINDOW_SIZE = 3
N_CLUSTERS = 3
URBAN_SPEED_LIMIT_IN_MS = 13.8889


def load_random_forest_classifier():
    print("Load Random Forest Classifier model from joblib file\n")
    rf = load('models/dangerous_driving_detection_model.joblib')
    return rf


def load_k_means_model():
    print("Load K-Means model from joblib file\n")
    km = load('models/driving_event_model.joblib')
    return km


def load_classification_features(dangerous_driving, k_means):
    print("Loading number of events detected\n")
    classification_features = dangerous_driving.get_learning_features(
        lambda x: count_events(x, columns=["acceleration_y", "gyro_z"], window_size=WINDOW_SIZE, n_clusters=N_CLUSTERS, k_means=k_means))

    print("Loading number of speeding detected\n")
    num_of_speeding = dangerous_driving.get_learning_features(lambda x: count_speeding(
        x, window_size=WINDOW_SIZE, speed_limit=URBAN_SPEED_LIMIT_IN_MS))

    classification_features = classification_features.merge(
        num_of_speeding, on="bookingID")

    return classification_features


def predict(feature_file):
    df = pd.read_csv(feature_file, dtype={'bookingID': str})
    rf = load_random_forest_classifier()
    km = load_k_means_model()
    features = df.sort_values("second").groupby("bookingID")

    dangerous_driving = DangerousDrivingDetection(features)

    print("Loading features\n")
    classification_features = load_classification_features(
        dangerous_driving, k_means=km)

    dangerous_driving.set_data(classification_features)
    bookingIds = dangerous_driving.data['bookingID']
    dangerous_driving.drop_column("bookingID")

    print("Predicting\n")
    y_pred = rf.predict(dangerous_driving.data)
    res = {'bookingID': bookingIds.values, 'is_dangerous': y_pred}
    return pd.DataFrame.from_dict(res)


if __name__ == "__main__":
    feature_file = sys.argv[1]
    print()
    print(predict(feature_file))
