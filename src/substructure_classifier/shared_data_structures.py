from dataclasses import dataclass

@dataclass
class LiveMetrics:
    batch_number: list = None
    sample_number: list = None
    running_loss: dict = None  # key: tuple or list name+dataset+dataset_config, value: loss array
    accuracy: dict = None  # key: tuple or list name+dataset+dataset_config, value: accuracy array
    f1_score: dict = None  # key: tuple or list name+dataset+dataset_config, value: f1_score array
    precision: dict = None  # key: tuple or list name+dataset+dataset_config, value: precision array
    recall: dict = None  # key: tuple or list name+dataset+dataset_config, value: recall array
    confusion_matrix: dict = None  # key: tuple or list name+dataset+dataset_config, value: confusion matrix array
    learning_rate: list = None
    """
    ___________
    |True Positives     |   False Negatives | [0, 0] [0, 1]
    |False Positives    |   True Negatives  | [1, 0] [1, 1]
    """

@dataclass
class TrainingCompletedMetrics:
    accuracy: dict = None  # key: catalog name, value: accuracy
    f1_score: dict = None  # key: catalog name, value: f1_score
    precision: dict = None  # key: catalog name, value: precision
    recall: dict = None  # key: catalog name, value: recall
