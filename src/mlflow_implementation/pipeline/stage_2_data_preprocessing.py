from src.mlflow_implementation.config.configuration import ConfigFileManagement
from src.mlflow_implementation.components.c_2_data_preprocessing import DataPreprocess
from src.mlflow_implementation import logger


class DataPreprocessPipeline:
    def __init__(self):
        pass

    def main(self, X, y):
        """

        :param X:
        :param y:
        :return:

        """
        param = ConfigFileManagement()
        data_preprocess_param = param.get_data_preprocessing_config()
        data_preprocess = DataPreprocess(param=data_preprocess_param)
        X_train, X_test, y_train, y_test = data_preprocess.split_data(X, y)
        X_train_scaled, X_test_scaled, scaler = data_preprocess.scale_features(X_train,X_test)

        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler
