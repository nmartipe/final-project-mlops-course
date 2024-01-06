import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import pandas as pd
import joblib
from src.constants import Constants

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def load_clean_data(file_path):
    logger.info("Loading data...")
    df = pd.read_csv(file_path)
    logger.info("Cleaning white spaces...")
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def process_data(df, categorical_features=[], label=None, inference = False):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    df : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    inference : bool
        Indicator if training mode or inference mode.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    """
    if inference == False:
        logger.info("We are not in inference process")
        if label is not None:
            y = df[label]
            X = df.drop([label], axis=1)
        else:
            y = np.array([])
        logger.info("Preprocessing categorical info...")
        X_categorical = X[categorical_features]

        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()

        logger.info("Applying OneHotEncoding to categorical features...")
        X_categorical_encoded = encoder.fit_transform(X_categorical)
        X_categorical_encoded_columns = encoder.get_feature_names_out(categorical_features)
        X_categorical_encoded_df = pd.DataFrame(X_categorical_encoded, columns=X_categorical_encoded_columns)

        logger.info("Applying LabelBinarizer to labels...")
        y = lb.fit_transform(y.values).ravel()

        X_continuous = X.drop(categorical_features, axis=1)
        X = pd.concat([X_continuous, X_categorical_encoded_df], axis=1)
        
        return X, y, encoder
    else:
        logger.info("We are in inference process")
        df = pd.DataFrame([df])
        logger.info("Loading encoder...")
        encoder = joblib.load(Constants.ENCODER_PATH)
        encoded_columns = joblib.load(Constants.COL_PATH).to_numpy()
        logger.info("Preprocessing inference data...")
        X_categorical = df[categorical_features]
        X_categorical_encoded = encoder.fit_transform(X_categorical)
        X_categorical_encoded_columns = encoder.get_feature_names_out(categorical_features)
        X_categorical_encoded_df = pd.DataFrame(X_categorical_encoded, columns=X_categorical_encoded_columns)
        X_continuous = df.drop(categorical_features, axis=1)
        X = pd.concat([X_continuous, X_categorical_encoded_df], axis=1)

        missing_columns = np.setdiff1d(encoded_columns, X.columns.to_numpy())
        X = pd.concat([X, pd.DataFrame(0, index=X.index, columns=missing_columns)], axis=1)
        X = X[encoded_columns]

        return X, None, None