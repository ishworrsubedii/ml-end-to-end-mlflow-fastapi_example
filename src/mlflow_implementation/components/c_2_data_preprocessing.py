from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.mlflow_implementation.entity.config_entity import DataPreprocessingConfig


class DataPreprocess:
    def __init__(self, param: DataPreprocessingConfig):
        self.param = param

    def split_data(self, X, y):
        """
        Split the data into training and testing sets.

        Args:
        - X: Input Features
        - y: Target columns

        Returns:
        - X_train: Training features
        - X_test: Testing features
        - y_train: Training target
        - y_test: Testing target
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.param.test_size,
                                                            random_state=self.param.random_state)
        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train, X_test):
        """
        Scale the features using StandardScaler.

        Args:
        - X_train: Training dataset
        - X_test: Test dataset

        Returns:
        - X_train_scaled: Scaled training features
        - X_test_scaled: Scaled testing features
        - scaler: StandardScaler object fitted on the training data
        """
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, scaler
