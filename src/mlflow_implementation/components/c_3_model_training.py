from sklearn.ensemble import RandomForestClassifier
from src.mlflow_implementation.entity.config_entity import ModelTrainingConfig


class ModelTraining:
    def __init__(self, param: ModelTrainingConfig):
        """
        Initialize the ModelTraining class.

        :param : ModelTrainingConfig object containing model training parameters like n_estimators and so on.
        """
        self.param = param

    def train_random_forest(self, X_train_scaled, y_train):
        """
        Train a Random Forest classifier.

        :param X_train_scaled: Scaled training features.
        :param y_train: Training target values.
        :return: Trained Random Forest classifier model.
        """
        model = RandomForestClassifier(n_estimators=self.param.n_estimators, random_state=self.param.random_state)

        model.fit(X_train_scaled, y_train)

        return model
