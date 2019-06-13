N_CLUSTERS = 3
LABEL_PATH = "./safety/labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv"
FEATURES_PATHS = [
    "./safety/features/part-0000{}-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv".format(i) for i in range(10)]
WINDOW_SIZE = 3
URBAN_SPEED_LIMIT_IN_MS = 13.8889