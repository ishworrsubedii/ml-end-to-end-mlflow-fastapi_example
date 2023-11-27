from src.mlflow_implementation.config.configuration import ConfigFileManagement
from src.mlflow_implementation.components.c_1_data_creation import CreateSyntheticDataset
from src.mlflow_implementation import logger


class DataCreationPipeline:
    def __init__(self):
        pass

    def main(self):
        """

        :return:
        """
        param = ConfigFileManagement()
        data_generation_param = param.get_data_creation_config()
        data_creation = CreateSyntheticDataset(param=data_generation_param)
        X, y = data_creation.create_synthetic_dataset()

        return X, y
