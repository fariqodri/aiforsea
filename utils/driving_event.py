from .detection import Detection
from .preprocessing import kalman_filter
from .extraction import get_variance_of_window
from config.constants import WINDOW_SIZE


class DrivingEventDetection(Detection):
    def train(self, model):
        model.fit(self.data)
        return model

    def load_clustering_features(self, columns):
        cluster_data = self.get_learning_features(lambda x: get_variance_of_window(x, columns,
                                                                                   WINDOW_SIZE))
        return cluster_data

    def select_features(self, selected_cols):
        self.set_data(self.data.loc[:, selected_cols])
