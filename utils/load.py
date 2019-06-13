import pandas as pd
from joblib import load


dtypes_dict = {'bookingID': str}


def load_features_group(features_paths):
    features = pd.concat([pd.read_csv(f, dtype=dtypes_dict)
                          for f in features_paths]).sort_values("second").groupby("bookingID")
    return features


def load_label_df(label_path):
    label = pd.read_csv(label_path, dtype={
                        **dtypes_dict}).drop_duplicates(subset="bookingID")
    return label


def load_model(model_path):
    model = load(model_path)
    return model
