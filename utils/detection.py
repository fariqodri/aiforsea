class Detection:
    def __init__(self, dataframe):
        self.data = dataframe

    def set_data(self, new_data):
        self.data = new_data

    def pre_process(self, column, callback):
        self.data[column] = callback(self.data)

    def get_learning_features(self, callback):
        feature_values = callback(self.data)
        return feature_values

    def dump_data(self, filename):
        self.data.to_csv(r'{}'.format(filename), index=False, header=True)

    def scale_data(self, scaler):
        self.data = scaler.fit_transform(self.data)

    def drop_column(self, column_name):
        self.data = self.data.drop(column_name, axis=1)
