# Script to train machine learning model.

from utils.data import load_clean_data, process_data
from sklearn.model_selection import train_test_split


file_path = '../data/census.csv'
df = load_clean_data(file_path)

train, test = train_test_split(df, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, 
                                    label="salary", training=False, encoder=encoder, lb=lb)

 
# Train and save a model.
