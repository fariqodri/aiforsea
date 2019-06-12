from .detection import Detection


class DangerousDrivingDetection(Detection):
    def split_feature_and_label(self, label_column):
        x = self.data.drop(label_column, axis=1)
        y = self.data[label_column]
        return x, y

    def merge_label_to_data(self, label_df, on):
        self.data = self.data.merge(label_df, on=on)
