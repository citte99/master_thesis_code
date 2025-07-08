import os
import json
from config import TRAINED_CLASSIFIERS_DIR
from config import CATALOGS_DIR
from catalog_manager import CatalogManager
from deep_learning.registry import DATASET_REGISTRY, MODEL_REGISTRY



def _load_stage_config_dict(classifier_name, stage_id):
    stage_config_path = os.path.join(TRAINED_CLASSIFIERS_DIR, classifier_name, "stages", stage_id, "stage_config.json")
    try:
        with open(stage_config_path, "r") as f:
            stage_config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Stage config file not found for stage ID: {stage_id}")
    return stage_config

def _get_model_parmas_path(classifier_name, stage_id):
    model_params_path = os.path.join(TRAINED_CLASSIFIERS_DIR, classifier_name, "stages", stage_id, "trained_params.pth")
    return model_params_path

#================================================SANITY CHECKS========================================================


def _validate_stage_config(stage_config):
    #outsourcing the validation: curret validating checks:
    # - training_catalog: must be provided, exist
    # - samples_used_for_training: must be > 0 and < than the number of samples in the training catalog
    # - if the stage id is set, then also is_trained should be set to True.
    # - the validation catalogs must exist.
    # - the samples used for validation (one number for all catalogs) must be > 0 and < than the number of samples in the validation catalog
    # - the dataset class must be provided and exist
    # - if the is_trained is set to False, then the early_stopping cannot be defined.
    # - the optimizer must be in a list of allowed optimizers
    # - the loss function must be in a list of allowed loss functions
    # - the validation metrics must be in a list of allowed validation metrics, both for live and training completed.
    # - check that the dataset exists.
    optimizers_registry= ["Adam", "sgd", "rmsprop"]

    #the losses are hard coded in make_loss_function of stage, so watch out for that.
    loss_functions_registry = ["Binary_cross_entropy", "mse"]
    validation_metrics_registry = ["accuracy", "f1_score", "precision", "recall"]


    #the exsitence of the catalogs is checked by loading them in the catalog manager
    training_catalog = CatalogManager(stage_config["training_catalog"])
    if stage_config["samples_used_for_training"] <= 0:
        raise ValueError("Samples used for training must be > 0")
    
    if stage_config["samples_used_for_training"] > training_catalog.len():
        raise ValueError("Samples used for training must be < than the number of samples in the training catalog")
    
    if stage_config["is_trained"] and stage_config["stage_id"] is None:
        raise ValueError("If the stage is trained, the stage id must be set")
    
    if not stage_config.get("is_trained") and "early_stopping" in stage_config:
        raise ValueError("If the stage is not trained, the early stopping cannot be defined")
    
    if "active_val_cats_live" in stage_config:
        for catalog, dataset, dataset_config in stage_config["active_val_cats_live"]:
            catalog = CatalogManager(catalog)
            if stage_config["samples_used_for_validation"] > catalog.len():
                raise ValueError("Samples used for validation must be < than the number of samples in the validation catalog")
            # check that the related datsetclass exists
            try:
                dataset = DATASET_REGISTRY.get(dataset)
            except KeyError:
                raise ValueError(f"Dataset class {dataset} not found in registry")
            
    if stage_config["samples_used_for_validation"] <= 0:
        raise ValueError("Samples used for validation must be > 0")
    
    
        
    try:
        dataset = DATASET_REGISTRY.get(stage_config["dataset_class_str"])
    except KeyError:
        raise ValueError(f"Dataset class {stage_config['dataset_class_str']} not found in registry")
    
    if stage_config["is_trained"] is False and "early_stopping" in stage_config:
        raise ValueError("If the stage is not trained, the early stopping cannot be defined")
    
    if stage_config["optimizer"] not in optimizers_registry:
        raise ValueError(f"Optimizer {stage_config['optimizer']} not found in registry. Check the util of classifier at validate_stage_config")
    if stage_config["loss_function"] not in loss_functions_registry:
        raise ValueError(f"Loss function {stage_config['loss_function']} not found in registry. Check the util of classifier at validate_stage_config")
    if stage_config["validation_metrics"] is not None:
        for metric in stage_config["validation_metrics"]:
            if metric not in validation_metrics_registry:
                raise ValueError(f"Validation metric {metric} not found in registry. Check the util of classifier at validate_stage_config")


def _check_ready_to_train(classifier_prop, stage_prop):
    if stage_prop.stage_id is None:
            raise ValueError("Stage id is not set. Cannot write the config. This is to ensure that the id the hash of all the necessary properties. Set id should be used only right before training.")
    #check device compatibilities
    print("Implement the device compatibility check in _check_ready_to_train, util.py")





#=========================================CLASSIFIER CONFIGURATION========================================================
def _validate_classifier_config(config):
    # self.classifier_properties = ClassifierProperties(
    #     classifier_name=self.config["classifier_name"],
    #     NN_model=self.config["NN_model"],
    #     NN_config=self.config["NN_config"],
    #     active_stage=self.config["active_stage"],
    #     active_validation_catalogs=self.config["active_validation_catalogs"]
    # )
    # check that all the keys are present in the config dict
    required_keys = ["classifier_name", "NN_model", "NN_config", "active_stage_id", "active_val_cats_live"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Key {key} not found in config dict")
    #check that the classifier model is in the registry
    try:
        MODEL_REGISTRY.get(config["NN_model"])
    except KeyError:
        raise ValueError(f"Classifier model {config['NN_model']} not found in registry")

    return