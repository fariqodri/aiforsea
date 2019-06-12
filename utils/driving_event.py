from .detection import Detection


class DrivingEventDetection(Detection):
    def train(self, model):
        model.fit(self.data)
        return model
