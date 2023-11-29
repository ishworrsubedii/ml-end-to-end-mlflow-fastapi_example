from src.mlflow_implementation import logger
from sklearn.metrics import accuracy_score
from src.mlflow_implementation.entity.config_entity import ModelEvalConfig

class ModelEvaluation:
    def __init__(self, param: ModelEvalConfig):
        self.param = param

    def accuracy(self, y_test, model, X_test_scaled):
        """
        Calculate the accuracy of the model using X_test and y_test.

        Args:
        - y_test: Target values for the test dataset.
        - model: Trained model object.
        - X_test_scaled: Scaled values of the test dataset.

        Returns:
        - Accuracy score if calculated successfully otherwise throw None.
        """
        try:
            accuracy_score_ = accuracy_score(y_test, model.predict(X_test_scaled))
            logger.info("Accuracy calculated successfully.")
            return accuracy_score_
        except Exception as e:
            logger.exception("An error occurred during accuracy calculation: %s", str(e))
            return None
