from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.mlflow_implementation.entity.config_entity import DataPreprocessingConfig


class DataPreprocess:
    def __init__(self, param: DataPreprocessingConfig):
        self.param = param

    def split_data(self, X, y):
        """
        :param X: Input Features
        :param y:Target columns
        :return:
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.param.test_size,
                                                            random_state=self.param.random_state)
        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train, X_test):
        """
        :param X_train:  training dataset
        :param X_test: test dataset
        :return:
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, scaler
