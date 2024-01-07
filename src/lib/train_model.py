from src.lib.utils.data import load_clean_data, process_data
from src.lib.utils.model import train_model, save_model_and_encoder, compute_model_metrics_by_slice
from sklearn.model_selection import train_test_split
import json
from src.constants import Constants

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

if __name__ == "__main__":
    logger.info("Loading data...")
    df = load_clean_data(Constants.DATA_PATH)
    logger.info("Starting preprocessing process...")
    X, y, encoder = process_data(df, categorical_features=Constants.CAT_FEATURES, label="salary", inference = False)
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info("Training model...")
    model = train_model(X_train, y_train)
    logger.info("Saving model, encoder and columns...")
    save_model_and_encoder(model, Constants.MODEL_PATH, encoder, Constants.ENCODER_PATH, X_train.columns, Constants.COL_PATH)
    logger.info("Getting model metrics...")
    metrics = compute_model_metrics_by_slice(X_test, y_test, model, Constants.CAT_FEATURES)
    logger.info(metrics)
    logger.info("Saving model metrics in model/slice_output.txt...")
    with open(Constants.METRICS_PATH, 'w') as f:
        metrics_str = json.dumps(metrics, indent=2)
        f.write(metrics_str)