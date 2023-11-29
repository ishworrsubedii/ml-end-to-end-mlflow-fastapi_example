from src.mlflow_implementation.api.api_base import logger_sys
from configparser import ConfigParser
from mlflow.sklearn import load_model


class ExampleService:
    def __init__(self):
        """
        Initializes ExampleService with model and scaler paths from config.ini file.
        """
        try:
            configure = ConfigParser()
            configure.read('config/config.ini')
            model_path = configure.get("example.model_info", "model_path")
            scaler_path = configure.get("example.model_info", "scaler_path")

            self.scaler = load_model(scaler_path)
            self.model = load_model(model_path)
            logger_sys.info("ExampleService initialized successfully.")
        except Exception as e:
            logger_sys.exception("Error initializing ExampleService: %s", str(e))
            raise

    def main_prediction(self, data):
        """
        Makes predictions on input data using the loaded model from mlflow.

        :param data: Input data for prediction
        :return: Scaled features and model predictions
        """
        try:
            scaled_features = self.scaler.transform(data)
            prediction = self.model.predict(scaled_features)
            logger_sys.info("Prediction made successfully.")
            return scaled_features, prediction
        except Exception as e:
            logger_sys.exception("Error making prediction: %s", str(e))
            return None, None
