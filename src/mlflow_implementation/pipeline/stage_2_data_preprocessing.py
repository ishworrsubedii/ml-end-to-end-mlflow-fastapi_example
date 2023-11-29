from src.mlflow_implementation.config.configuration import ConfigFileManagement
from src.mlflow_implementation.components.c_2_data_preprocessing import DataPreprocess
from src.mlflow_implementation import logger


class DataPreprocessPipeline:
    def __init__(self):
        pass

    def main(self, X, y):
        """
        Preprocesses the data.

        :param X: Input features.
        :param y: Target labels.
        :return: Processed training and testing data along with the scaler.
        """
        config_manager = ConfigFileManagement()
        data_preprocess_param = config_manager.get_data_preprocessing_config()

        data_preprocessor = DataPreprocess(param=data_preprocess_param)

        X_train, X_test, y_train, y_test = data_preprocessor.split_data(X, y)
        X_train_scaled, X_test_scaled, scaler = data_preprocessor.scale_features(X_train, X_test)

        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler
