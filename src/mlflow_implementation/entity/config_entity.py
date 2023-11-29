from dataclasses import dataclass


@dataclass(frozen=True)
class DataCreationConfig:
    n_samples: int
    n_features: int
    n_classes: int
    random_state: int


@dataclass(frozen=True)
class DataPreprocessingConfig:
    test_size: float
    random_state: int


@dataclass(frozen=True)
class ModelTrainingConfig:
    n_estimators: int
    random_state: int


@dataclass(frozen=True)
class ModelEvalConfig:
    pass
