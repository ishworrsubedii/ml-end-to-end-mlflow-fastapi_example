from src.mlflow_implementation.config.configuration import ConfigFileManagement
from src.mlflow_implementation.components.c_5_model_training import ModelTraining
from src.mlflow_implementation import logger


class ModelTrainPipeline:
    def __init__(self):
        pass

    def main(self, X_train_scaled, y_train):
        """
        Trains a Random Forest model using the provided scaled training data.

        :param X_train_scaled: Scaled feature data for training
        :param y_train: Target labels for training
        :return: Trained Random Forest model
        """
        try:
            param = ConfigFileManagement()
            model_training_param = param.get_model_training_config()
            model_train = ModelTraining(param=model_training_param)
            model = model_train.train_random_forest(X_train_scaled, y_train)
            return model
        except Exception as e:
            logger.exception(f"Exception occurred during model training: {e}")
            raise e
