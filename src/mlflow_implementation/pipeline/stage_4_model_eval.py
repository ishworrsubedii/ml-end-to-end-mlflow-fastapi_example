from src.mlflow_implementation.config.configuration import ConfigFileManagement
from src.mlflow_implementation.components.c_4_model_eval import ModelEvaluation
from src.mlflow_implementation import logger


class ModelEvalPipeline:
    def __init__(self):
        pass

    def main(self, y_test, model, X_test_scaled):
        """
        Main function to execute the model evaluation pipeline.

        :param y_test: y_test dataset.
        :param model: Model object.
        :param X_test_scaled: Scaled values of X_test.
        :return: Accuracy score or None if an exception occurs.
        """
        try:
            param = ConfigFileManagement()
            model_training_param = param.get_model_eval_config()
            model_eval = ModelEvaluation(param=model_training_param)
            accuracy = model_eval.accuracy(y_test, model, X_test_scaled)
            logger.info("Model evaluation pipeline executed successfully.")
            return accuracy
        except Exception as e:
            logger.exception("An error occurred in the model evaluation pipeline: %s", str(e))
            return None
