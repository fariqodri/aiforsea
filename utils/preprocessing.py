from pykalman import KalmanFilter


def kalman_filter(dataframe, column):
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    measurement = dataframe[column].values
    mean, variance = kf.em(measurement).filter(measurement)
    return mean
