from .detection import Detection
from .extraction import count_events, count_speeding
from config.constants import *


class DangerousDrivingDetection(Detection):
    def split_feature_and_label(self, label_column):
        x = self.data.drop(label_column, axis=1)
        y = self.data[label_column]
        return x, y

    def merge_label_to_data(self, label_df, on):
        self.data = self.data.merge(label_df, on=on)

    def load_classification_features(self, k_means):
        classification_features = self.get_learning_features(
            lambda x: count_events(x, columns=["acceleration_y", "gyro_z"], window_size=WINDOW_SIZE, n_clusters=N_CLUSTERS, k_means=k_means))

        num_of_speeding = self.get_learning_features(lambda x: count_speeding(
            x, window_size=10, speed_limit=URBAN_SPEED_LIMIT_IN_MS))

        classification_features = classification_features.merge(
            num_of_speeding, on="bookingID")

        return classification_features
