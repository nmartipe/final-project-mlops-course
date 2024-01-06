from utils.data import load_clean_data, process_data
from utils.model import train_model, save_model_and_encoder, compute_model_metrics_by_slice, load_model
from sklearn.model_selection import train_test_split
import json
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

data_path = '../data/census.csv'
model_path = '../model/model.pkl'
encoder_path = '../model/encoder.pkl'
col_path = '../model/encoded_columns.pkl'

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

logger.info("Loading data...")
df = load_clean_data(data_path)
logger.info("Starting preprocessing process...")
X, y, encoder = process_data(df, categorical_features=cat_features, label="salary", inference = False)
logger.info("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info("Training model...")

model = train_model(X_train, y_train)
logger.info("Saving model, encoder and columns...")
save_model_and_encoder(model, model_path, encoder, encoder_path, X_train.columns, col_path)
logger.info("Getting model metrics...")
metrics = compute_model_metrics_by_slice(X_test, y_test, model, cat_features)
logger.info("Saving model metrics...")
with open('../model/metrics.txt', 'w') as f:
    metrics_str = json.dumps(metrics, indent=2)
    f.write(metrics_str)