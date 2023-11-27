from sklearn.ensemble import RandomForestClassifier
from src.mlflow_implementation.entity.config_entity import ModelTrainingConfig


class ModelTraining:
    def __init__(self, param: ModelTrainingConfig):
        """

        :param param:
        """
        self.param = param

    def train_random_forest(self, X_train_scaled, y_train):
        """

        :param X_train_scaled:
        :param y_train:
        :return:
        """

        model = RandomForestClassifier(n_estimators=self.param.n_estimators, random_state=self.param.random_state)
        model.fit(X_train_scaled, y_train)
        return model
