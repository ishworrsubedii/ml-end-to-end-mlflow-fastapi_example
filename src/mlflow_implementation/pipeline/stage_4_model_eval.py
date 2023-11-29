from src.mlflow_implementation.config.configuration import ConfigFileManagement
from src.mlflow_implementation.components.c_4_model_eval import ModelEvaluation
from src.mlflow_implementation import logger


class ModelEvalPipeline:
    def __init__(self):
        pass

    def main(self, y_test, model, X_test_scaled):
        """
        Execute the model evaluation pipeline.

        :param y_test: Test labels.
        :param model: Trained model object.
        :param X_test_scaled: Scaled test features.
        :return: Accuracy score or None if an exception occurs.
        """
        try:
            config_manager = ConfigFileManagement()
            model_eval_params = config_manager.get_model_eval_config()

            model_evaluator = ModelEvaluation(param=model_eval_params)
            accuracy = model_evaluator.accuracy(y_test, model, X_test_scaled)

            logger.info("Model evaluation pipeline executed successfully.")
            return accuracy
        except Exception as e:
            logger.exception("An error occurred in the model evaluation pipeline: %s", str(e))
            return None
