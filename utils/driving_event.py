from .detection import Detection
from .preprocessing import kalman_filter


class DrivingEventDetection(Detection):
    def train(self, model):
        model.fit(self.data)
        return model
