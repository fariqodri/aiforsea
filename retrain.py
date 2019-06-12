from utils.dangerous_driving import DangerousDrivingDetection
from utils.driving_event import DrivingEventDetection
from utils.preprocessing import kalman_filter
from utils.extraction import count_events, count_speeding, get_variance_of_window

import pandas as pd


N_CLUSTERS = 3
LABEL_PATH = "./safety/labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv"
FEATURES_PATHS = [
    "./safety/features/part-0000{}-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv".format(i) for i in range(10)]
WINDOW_SIZE = 3
URBAN_SPEED_LIMIT_IN_MS = 13.8889

dtypes_dict = {'bookingID': str}
label = pd.read_csv(LABEL_PATH, dtype={
                    **dtypes_dict, 'label': 'category'}).drop_duplicates(subset="bookingID")
features = pd.concat([pd.read_csv(f, dtype=dtypes_dict)
                      for f in FEATURES_PATHS]).sort_values("second").groupby("bookingID")
