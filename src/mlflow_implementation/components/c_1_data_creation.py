import os
from src.mlflow_implementation import logger
from sklearn.datasets import make_classification
from src.mlflow_implementation.entity.config_entity import DataCreationConfig


class CreateSyntheticDataset:
    def __init__(self, param: DataCreationConfig):
        self.param = param

    def create_synthetic_dataset(self):
        """
        For Creating dataset using make classification library
        we can make as much as data as we want

        :return:
        """
        try:
            X, y = make_classification(n_samples=self.param.n_samples, n_features=self.param.n_features,
                                       n_classes=self.param.n_classes, random_state=self.param.random_state)

            logger.info(f"Dataset Created with provided info: Samples={self.param.n_samples}, "
                        f"Features={self.param.n_features}, Classes={self.param.n_classes}, "
                        f"Random State={self.param.random_state}")

        except Exception as e:
            logger.exception(f"An error occurred: {str(e)}")

        return X, y
