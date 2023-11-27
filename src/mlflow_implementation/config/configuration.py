from src.mlflow_implementation.constant import *
from src.mlflow_implementation.utils.utils import read_yaml
from src.mlflow_implementation.entity.config_entity import (DataPreprocessingConfig, DataCreationConfig,
                                                            ModelEvalConfig, ModelTrainingConfig)


class ConfigFileManagement:
    def __init__(self, param_filepath=PARAMS_FILE_PATH,
                 config_filepath=CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.param = read_yaml(param_filepath)

    def get_data_creation_config(self) -> DataCreationConfig:
        param = self.param.data_creation
        data_creation_config = DataCreationConfig(
            n_samples=param.n_samples,
            n_features=param.n_features,
            n_classes=param.n_classes,
            random_state=param.random_state

        )
        return data_creation_config

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        param = self.param.train_test_split
        data_preprocessing_config = DataPreprocessingConfig(
            test_size=param.test_size,
            random_state=param.random_state

        )
        return data_preprocessing_config

    def get_model_training_config(self) -> ModelTrainingConfig:
        param = self.param.model_train
        model_training_config = ModelTrainingConfig(
            n_estimators=param.n_estimator,
            random_state=param.random_state

        )
        return model_training_config

    def get_model_eval_config(self) -> ModelEvalConfig:
        pass
