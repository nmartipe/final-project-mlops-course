import os


def get_absolute_path(file_path):
    src_directory = os.path.dirname(__file__)
    src_directory = os.path.abspath(src_directory)
    return os.path.join(src_directory, file_path)

class Constants:
    DATA_PATH = get_absolute_path('./data/census.csv')
    MODEL_PATH = get_absolute_path('./model/model.pkl')
    ENCODER_PATH = get_absolute_path('./data/encoder.pkl')
    COL_PATH = get_absolute_path('./data/encoded_columns.pkl')
    METRICS_PATH = get_absolute_path('./model/metrics.txt')
    
    CAT_FEATURES = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]