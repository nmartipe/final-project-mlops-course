import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import pandas as pd


def load_clean_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def process_data(df, categorical_features=[], label=None):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = df[label]
        X = df.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features]

    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    lb = LabelBinarizer()

    # Apply OneHotEncoding to categorical features
    X_categorical_encoded = encoder.fit_transform(X_categorical)
    X_categorical_encoded_columns = encoder.get_feature_names_out(categorical_features)
    X_categorical_encoded_df = pd.DataFrame(X_categorical_encoded, columns=X_categorical_encoded_columns)

    # Apply LabelBinarizer to labels
    y = lb.fit_transform(y.values).ravel()

    X_continuous = X.drop(categorical_features, axis=1)
    X = pd.concat([X_continuous, X_categorical_encoded_df], axis=1)
    
    return X, y, encoder
