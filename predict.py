from joblib import dump, load
import sys
import pandas as pd
from utils.dangerous_driving import DangerousDrivingDetection
from utils.load import load_features_group, load_model
from config.constants import *


def predict(feature_files):
    rf = load_model('models/random_forest_classifier.joblib')
    km = load_model('models/k_means.joblib')
    features = load_features_group(feature_files)

    dangerous_driving = DangerousDrivingDetection(features)

    classification_features = dangerous_driving.load_classification_features(
        k_means=km)

    dangerous_driving.set_data(classification_features)
    bookingIds = dangerous_driving.data['bookingID']
    dangerous_driving.drop_column("bookingID")

    y_pred = rf.predict(dangerous_driving.data)
    res = {'bookingID': bookingIds.values, 'prediction': y_pred}
    return pd.DataFrame.from_dict(res)


if __name__ == "__main__":
    feature_files = sys.argv[1:]
    print()
    result = predict(feature_files)
    print(result.to_string(index=False))
    print()
    print("The prediction result is also saved in this script directory as result.csv")
    result.to_csv("result.csv", index=False)