from deep_learning import MODEL_REGISTRY, DATASET_REGISTRY, custom_dataloader
from .util import _load_stage_config_dict, _validate_stage_config, _check_ready_to_train, _get_model_parmas_path
from dataclasses import dataclass
#from substructure_classifier.substructure_classifier_development import SubstructureClassifier
from torch import nn
import json
import torch
import os
from tqdm.auto import tqdm
import copy
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import hashlib
from .shared_data_structures import LiveMetrics, TrainingCompletedMetrics
from matplotlib import pyplot as plt
from torch_lr_finder import LRFinder          # optional package
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from collections import deque
import wandb



@dataclass
class ClassifierProperties:
    classifier_name: str = None
    NN_model_str: str = None
    NN_config: dict = None
    active_val_cats_live: list = None
    active_stage_id: str = None


@dataclass
class StageProperties:
    stage_id: str = None
    parent_stage_id: str = None
    training_catalog: str = None
    validation_like_train_catalog: str = None
    dataset_class_str: str = None
    NN_datatype: str = None
    dataset_config: dict = None
    samples_used_for_training: int = 0
    samples_used_for_validation: int = 0
    epochs: int = 0
    batch_size: int = 0
    learning_rate: float = 0.0
    optimizer: str = None
    loss_function: str = None
    active_val_cats_live: list = None
    validation_metrics: list = None
    jump_batch_val: int = 0
    validation_catalogs_training_completed: list = None
    validation_metrics_training_completed: list = None
    early_stopping: int = 0
    is_trained: bool = False




class Stage:
    config={
            #commented out are assigned in the constructor
            #"stage_id":None,
            #"parent_stage_id":None,
            "training_catalog":"SIS_10e10_sub_train",
            "validation_like_train_catalog":"SIS_10e10_sub_test",
            "dataset_class_str":"NoNoiseDataset",
            "dataset_config":{},
            "NN_datatype":"float32",
            "samples_used_for_training":1,
            "samples_used_for_validation":1,
            "epochs":1,
            #"early_stopping":0,
            "batch_size":1,
            "learning_rate":0.001,
            "optimizer":"Adam",
            "loss_function":"Binary_cross_entropy",
            #"active_val_cats_live":[("catalog_name1", "dataset_name1"), ... ],
            "validation_metrics":["accuracy", "f1_score", "precision", "recall"],
            "jump_batch_val":1,
            "validation_catalogs_training_completed":"",
            "validation_metrics_training_completed":["accuracy", "f1_score", "precision", "recall"],
            #"is_trained":False,
        }
    
    @staticmethod
    def get_example_config(return_config=True):
        if return_config:
            return Stage.config.copy()

    def __init__(self, classifier_instance, config=None, stage_id=None, device='cuda'):
        # Modes: Interactive, Config as input, Stage id as input
        self.Classifier=classifier_instance
        print("doneStage1")
        self.classifier_name=classifier_instance.get_classifier_name()
        self.device = device
        print("doneStage2")


        self.classifier_properties = ClassifierProperties()
        self.stage_properties = StageProperties()
        print("doneStage3")

        if config is not None and stage_id is None:
            self.config_dict = config
            self.config_dict["is_trained"] = False
        elif stage_id is not None:
            self.stage_id = stage_id
            self.config_dict=_load_stage_config_dict(classifier_name=self.classifier_name, stage_id=self.stage_id)
        else:
            raise ValueError("Either config or stage_id must be provided")
        
        

        print("doneStage4")

        self._load_Classifier_config()
        print("doneStage5")

        
        self._load_stage_config()
        print("doneStage6")

        self.stage_params_trained_loaded=False
        print("doneStage7")




    def _load_Classifier_config(self):
        classifier_config= self.Classifier.get_config_dict()
                
        self.classifier_properties.classifier_name = classifier_config["classifier_name"]
        self.classifier_properties.NN_model_str = classifier_config["NN_model"]
        # check that the model is in the registry
        try:
            model = MODEL_REGISTRY.get(classifier_config["NN_model"])
        except KeyError:
            raise ValueError(f"Model {classifier_config['NN_model']} not found in registry")
        
        self.classifier_properties.NN_config = classifier_config["NN_config"]
        self.classifier_properties.active_val_cats_live = classifier_config["active_val_cats_live"]
        self.classifier_properties.active_stage_id = classifier_config["active_stage_id"]

    def _load_stage_config(self):
        
        print("load_stage1")
        """
        Load and validate the stage configuration from disk (if a stage_id is set),
        then populate StageProperties accordingly.
        """
        # If we already have a stage_id, reload config from the latest JSON on disk
        if getattr(self.stage_properties, 'stage_id', None) is not None:
            fresh_config = _load_stage_config_dict(
                classifier_name=self.classifier_name,
                stage_id=self.stage_properties.stage_id
            )
            # Replace in-memory dict with the fresh one
            self.config_dict = fresh_config
        print("load_stage2")

        # Validate the resulting config_dict
        
        #super slow for some reason....
        #_validate_stage_config(self.config_dict)
        
        print("load_stage2.1")

        # Parent stage logic: prefer explicit, otherwise pull from classifier
        self.stage_properties.parent_stage_id = self.config_dict.get("parent_stage_id")
        if self.stage_properties.parent_stage_id is None:
            self._load_Classifier_config()
            print("load_stage2.2")

            self.stage_properties.parent_stage_id = (
                self.classifier_properties.active_stage_id
                if self.classifier_properties.active_stage_id is not None
                else None
            )
        print("load_stage3")

        # If stage_id was provided in config, use it and honor is_trained flag
        if self.config_dict.get("stage_id") is not None:
            self.stage_properties.stage_id = self.config_dict["stage_id"]
            self.stage_properties.is_trained = self.config_dict.get("is_trained", False)
        print("load_stage4")

        # Populate the rest of StageProperties from config_dict
        self.stage_properties.training_catalog = self.config_dict["training_catalog"]
        self.stage_properties.validation_like_train_catalog = self.config_dict["validation_like_train_catalog"]
        self.stage_properties.samples_used_for_training = self.config_dict.get("samples_used_for_training", 0)
        self.stage_properties.samples_used_for_validation = self.config_dict.get("samples_used_for_validation", 0)
        self.stage_properties.epochs = self.config_dict.get("epochs", 0)
        self.stage_properties.early_stopping = self.config_dict.get("early_stopping")
        self.stage_properties.batch_size = self.config_dict.get("batch_size", 0)
        self.stage_properties.learning_rate = self.config_dict.get("learning_rate", 0.0)
        self.stage_properties.optimizer = self.config_dict.get("optimizer")
        self.stage_properties.loss_function = self.config_dict.get("loss_function")
        self.stage_properties.dataset_class_str = self.config_dict.get("dataset_class_str")
        self.stage_properties.dataset_config = self.config_dict.get("dataset_config", {})
        self.stage_properties.NN_datatype = self.config_dict.get("NN_datatype")
        self.stage_properties.jump_batch_val = self.config_dict.get("jump_batch_val", 0)

        # Always use classifier's live validation catalogs
        self.stage_properties.active_val_cats_live = (
            self.classifier_properties.active_val_cats_live
        )
        print("load_stage5")

        # Validation-completed settings
        self.stage_properties.validation_metrics = self.config_dict.get("validation_metrics", [])
        self.stage_properties.validation_catalogs_training_completed = self.config_dict.get(
            "validation_catalogs_training_completed", []
        )
        print("load_stage6")
        
        self.stage_properties.validation_metrics_training_completed = self.config_dict.get(
            "validation_metrics_training_completed", []
        )

    def set_stage_id(self):
        #setting the stage id is possible only before the stage is trained.
        if self.stage_properties.stage_id is not None:
            raise ValueError("Stage id is already set. Cannot set it again.")
        if self.stage_properties.is_trained:
            raise ValueError("Stage is already trained. Cannot set the stage id.")
        #this needs to be set at the end of the load stage config.
        #stage_id: this could be already set if we loaded an existing stage. otherwise, we generate a new one.
        # we hash the properties fixed at training time. So everything unless:
        # - early_stopping
        # - validation_catalogs_training_completed
        # - validation_metrics_training_completed
        # - the current stage id


        hashable_properties = self.stage_properties.__dict__.copy()
        hashable_properties.pop("early_stopping")
        hashable_properties.pop("validation_catalogs_training_completed")
        hashable_properties.pop("validation_metrics_training_completed")
        hashable_properties.pop("stage_id")
        # Serialize the hashable properties dictionary into a JSON string
        hashable_properties_str = json.dumps(hashable_properties, sort_keys=True)
        # Use SHA256 to create a unique hash
        self.stage_properties.stage_id = self.stage_id = hashlib.sha256(hashable_properties_str.encode()).hexdigest()

    def _load_dataloader(self, catalog_name, dataset_class_str, dataset_config=None, val=False):
        dataset_config = dataset_config
        dataset_config_temp=dataset_config.copy()
        dataset_config_temp["catalog_name"] = catalog_name
        if val:
            dataset_config_temp["samples_used"] = self.stage_properties.samples_used_for_validation
        else:
            dataset_config_temp["samples_used"] = self.stage_properties.samples_used_for_training
        dataset = DATASET_REGISTRY.get(dataset_class_str)(**dataset_config_temp)
        if val:
            return custom_dataloader(dataset=dataset, batch_size=self.stage_properties.batch_size, shuffle=False)
        else:
            return custom_dataloader(dataset=dataset, batch_size=self.stage_properties.batch_size, shuffle=True)


    def _load_model(self, load_parent=False):
        if self.stage_params_trained_loaded==True:
            return self.stocked_loaded_model

        
        #load the model from the registry
        model_str = self.classifier_properties.NN_model_str
        model_config = self.classifier_properties.NN_config.copy()
        # check that the model is in the registry
        try:
            NN_model_class = MODEL_REGISTRY.get(model_str)
        except KeyError:
            raise ValueError(f"Model {model_str} not found in registry")
        model = NN_model_class(**model_config)
        
        model.to(self.device)
        model = torch.compile(
            model,
            backend="inductor",
            fullgraph=False,      # or drop this if you don't need full-graph
            dynamic=False        # or False if your input size is fixed
        )

        
        
#         if load_parent:
#             if self.stage_properties.parent_stage_id is None:
#                 # initialize the model with random parameters
#                 # not loading any weights, the model is initialized with random parameters
#                 print("Model initialized with default random parameters")
#             else:
#                 parms_path= _get_model_parmas_path(classifier_name=self.classifier_name, stage_id=self.stage_properties.parent_stage_id)
#                 model.load_state_dict(torch.load(parms_path))
#                 print(f"parameters loaded from  {parms_path}")
        if load_parent:
            if self.stage_properties.parent_stage_id is None:
                print("Model initialized with kaiming_init")
                

                def kaiming_init(m):
                    """
                    Applies Kaiming init to weights and zero to biases:
                      - Conv2d: kaiming_normal_ (fan_out) for ReLU
                      - Linear: kaiming_uniform_ (fan_in) for ReLU
                    """
                    if isinstance(m, nn.Conv2d):
                        # He normal is a good default for conv + ReLU
                        nn.init.kaiming_normal_(m.weight, 
                                                mode='fan_out', 
                                                nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

                    elif isinstance(m, nn.Linear):
                        # He uniform also works nicely for linear + ReLU
                        nn.init.kaiming_uniform_(m.weight, 
                                                 mode='fan_in', 
                                                 nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

                # Example: assume `model` is your nn.Module
                model.apply(kaiming_init)
#                 print("Model initialized with ImageNet parameters")

#                 import torchvision.models as models
#                 import torch.nn as nn

#                 # 1. Get the official weights
#                 reference = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#                 state_dict = reference.state_dict()

#                 # 2. Adapt conv1 from (64, 3, 7, 7) ➜ (64, 1, 7, 7)
#                 c1 = state_dict['conv1.weight']          # (64,3,7,7)
#                 state_dict['conv1.weight'] = c1.mean(dim=1, keepdim=True)

#                 # 3. Drop the classifier
#                 state_dict.pop('fc.weight')
#                 state_dict.pop('fc.bias')

                # 4. Build *your* network and load what fits
                #missing, unexpected = model.load_state_dict(state_dict, strict=False)
            else:
                parms_path = _get_model_parmas_path(
                    classifier_name=self.classifier_name,
                    stage_id=self.stage_properties.parent_stage_id
                )
                print(f"[DEBUG] parent_stage_id = {self.stage_properties.parent_stage_id!r}")
                print(f"[DEBUG] loading checkpoint from {parms_path!r}")
                if not os.path.exists(parms_path):
                    raise FileNotFoundError(f"Checkpoint not found at {parms_path}")
                # show one weight‐vector norm before load
                wname, wparam = next(model.named_parameters())
                print(f"[DEBUG] before load: ‖{wname}‖ = {wparam.norm().item():.6f}")
                sd = torch.load(parms_path, map_location=self.device)
                model.load_state_dict(sd)
                print(f"[DEBUG] after load:  ‖{wname}‖ = {model.state_dict()[wname].norm().item():.6f}")
        else:
            print("Loading paramters of the current stage, not the parent. This mode is intended for using an already trained stage for some evaluations")
            parms_path= _get_model_parmas_path(classifier_name=self.classifier_name, stage_id=self.stage_id)
            model.load_state_dict(torch.load(parms_path))
            self.stocked_loaded_model=model
            self.stage_params_trained_loaded=True
            
        return model
    
    def _make_loss_function(self, loss_function):
        if loss_function =="Binary_cross_entropy":
            #return nn.BCEWithLogitsLoss()
            return nn.CrossEntropyLoss()  # For multi-class classification

        else:
            raise ValueError(f"Loss function {loss_function} currently not supported. If you add a new one, check the training and eval code.")
        
    def _make_optimizer(self, model, optimizer):
        if optimizer == "Adam":
            #print("authomatic learning rate, disregarding suggestion in the config")
            #return torch.optim.Adam(model.parameters(), lr=self.stage_properties.learning_rate)
            LR_FINDER_START_LR  = 1e-7
            #return torch.optim.Adam(model.parameters(), lr=LR_FINDER_START_LR)
            from torch.optim import AdamW

            # optimizer = AdamW(
            #     model.parameters(),
            #     lr=LR_FINDER_START_LR,
            #     weight_decay=1e-4      # typical values: 1e-3 → 1e-5
            # )
            optimizer = AdamW(
                model.parameters(),
                lr=self.stage_properties.learning_rate,
                weight_decay=1e-4      # typical values: 1e-3 → 1e-5
            )
            return optimizer
        
        else:
            raise ValueError(f"Optimizer {optimizer} currently not supported. If you add a new one, check the training and eval code.")
        
    def get_current_settings(self):
        print("Classifier properties:")
        print(json.dumps(self.classifier_properties.__dict__, indent=4))
        print("Stage properties:")
        print(json.dumps(self.stage_properties.__dict__, indent=4))
        return
    
    def get_stage_summary_small(self):
        # catalog_name, dataset_class, samples_used
        catalog_name = self.stage_properties.training_catalog
        dataset_class = self.stage_properties.dataset_class_str
        samples_used = self.stage_properties.samples_used_for_training
        summary = {
            "catalog_name": catalog_name,
            "dataset_class": dataset_class,
            "samples_used": samples_used
        }
        return summary


#==============================================METRICS========================================================
    def get_live_validation_catalogs(self):
        return self.classifier_properties.active_val_cats_live
    
    def add_live_validation_catalog(self, catalog_name):
        self.classifier_properties.active_val_cats_live.append(catalog_name)
    
    def remove_live_validation_catalog(self, catalog_name):
        self.classifier_properties.active_val_cats_live.remove(catalog_name)

    def get_val_cat_training_completed(self):
        return self.stage_properties.validation_catalogs_training_completed
    
    def add_val_cat_training_completed(self, catalog_name):
        self.stage_properties.validation_catalogs_training_completed.append(catalog_name)

    def remove_val_cat_training_completed(self, catalog_name):
        self.stage_properties.validation_catalogs_training_completed.remove(catalog_name)


    def get_completed_validation_metrics(self):
        return self.stage_properties.validation_metrics_training_completed
    
    def add_completed_validation_metric(self, metric_name):
        self.stage_properties.validation_metrics_training_completed.append(metric_name)

    def remove_completed_validation_metric(self, metric_name):
        self.stage_properties.validation_metrics_training_completed.remove(metric_name)

    def _initialize_running_metrics(self):
        # Create empty lists for batch number
        batch_numbers = []
        sample_number = []
        # Create empty dictionaries for each metric
        running_loss = {}
        accuracy = {}
        f1_score = {}
        precision = {}
        recall = {}
        confusion_matrix = {}
        learning_rate=[]
        
        # Get tracked catalogs
        tracked_catalogs = []
        
        # Extract catalog names from active_val_cats_live
        for cat_name, dataset_class, dataset_config in self.stage_properties.active_val_cats_live:
            tracked_catalogs.append(cat_name)
        
        print(f"Tracked catalogs: {tracked_catalogs}")
        # Initialize the dictionaries for metrics
        for catalog in tracked_catalogs:
            running_loss[catalog] = []
            accuracy[catalog] = []
            f1_score[catalog] = []
            precision[catalog] = []
            recall[catalog] = []
            confusion_matrix[catalog] = []
        
        # Initialize LiveMetrics
        self.live_metrics = LiveMetrics(
            batch_number=batch_numbers,
            sample_number=sample_number,
            running_loss=running_loss,
            accuracy=accuracy,
            f1_score=f1_score,
            precision=precision,
            recall=recall,
            confusion_matrix=confusion_matrix,
            learning_rate=learning_rate
        )

    def compute_validation_metrics_offline(self):
        # if not loaded, load the model
        # load the model
        if self.stage_properties.is_trained is False:
            raise ValueError("Stage is not trained. Cannot compute validation metrics.")
        
        model = self._load_model()
        model.eval()  # Set model to evaluation mode
        
        # Create a dictionary to store metrics for each validation catalog
        training_completed_metrics = TrainingCompletedMetrics(
            accuracy={},
            f1_score={},
            precision={},
            recall={}
        )
        
        # Iterate over all validation catalogs for training completed
        for catalog in self.stage_properties.validation_catalogs_training_completed:
            # Load the validation dataloader
            val_dataloader = self._load_dataloader(
                catalog_name=catalog, 
                dataset_class_str=self.stage_properties.dataset_class_str, 
                val=True
            )
            
            # Set up variables to store predictions and labels
            all_predictions = []
            all_labels = []
            
            # Determine if it's binary classification (check first batch)
            is_binary = False
            for inputs, labels in val_dataloader:
                outputs = model(inputs)
                is_binary = (outputs.shape[1] == 1)
                break
            
            # Use the model to make predictions
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    outputs = model(inputs)
                    
                    # For binary classification
                    if is_binary:
                        predictions = (torch.sigmoid(outputs) > 0.5).float()
                    else:  # Multi-class classification
                        predictions = torch.argmax(outputs, dim=1)
                    
                    # Convert to numpy and store
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Convert lists to numpy arrays
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            
            # Calculate metrics
            if is_binary:  # Binary classification
                average_method = 'binary'
            else:  # Multi-class classification
                average_method = 'weighted'
            
            # Calculate accuracy
            training_completed_metrics.accuracy[catalog] = (all_predictions == all_labels).mean()
            
            # Calculate F1 score
            training_completed_metrics.f1_score[catalog] = f1_score(
                all_labels, all_predictions, average=average_method
            )
            
            # Calculate precision
            training_completed_metrics.precision[catalog] = precision_score(
                all_labels, all_predictions, average=average_method, zero_division=0
            )
            
            # Calculate recall
            training_completed_metrics.recall[catalog] = recall_score(
                all_labels, all_predictions, average=average_method, zero_division=0
            )
        
        # Save the metrics
        metrics_path = os.path.join(
            self.Classifier.classifier_path, 
            "stages", 
            str(self.stage_properties.stage_id), 
            "training_completed_metrics.json"
        )
        
        with open(metrics_path, "w") as f:
            json.dump(training_completed_metrics.__dict__, f, indent=4)
        
        print(f"Training completed metrics saved to {metrics_path}")
        
        return training_completed_metrics

    def write_stage_config(self):
        # check that all the necessary properties are set in the self.stage_properties
        # and write the config to a json file.
        # check that the stage id is set
        _check_ready_to_train(classifier_prop=self.classifier_properties, stage_prop=self.stage_properties)

        # Ensure the necessary directories exist
        stage_path = os.path.join(self.Classifier.classifier_path, "stages", str(self.stage_properties.stage_id))
        os.makedirs(stage_path, exist_ok=True)

        stage_config_path = os.path.join(stage_path, "stage_config.json")

        # Check if the file already exists
        if os.path.exists(stage_config_path):
            if self.stage_properties.is_trained:
                allowed_to_change_keys = ["is_trained","validation_catalogs_training_completed", "validation_metrics_training_completed", "learning_rate"]
                # Check if the keys that are changed are allowed to be changed                
                previous_config = _load_stage_config_dict(classifier_name=self.classifier_name, stage_id=self.stage_properties.stage_id)
                for key in previous_config:
                    #check if previous value is different from the current value
                    if key not in allowed_to_change_keys and previous_config[key] != self.stage_properties.__dict__[key]:
                        raise ValueError(f"Key {key} cannot be changed after training. Previous value: {previous_config[key]}, current value: {self.stage_properties.__dict__[key]}")
    
        with open(stage_config_path, "w") as f:
            json.dump(self.stage_properties.__dict__, f, indent=4)


    def train(self, train_ready=False, early_stopping=False):
        if self.stage_properties.is_trained:
            raise ValueError("Stage is already trained. Cannot train again.")

        if not train_ready:
            self.get_current_settings()
            raise ValueError(
                "This is a safety feature. Check that the stage is ready to be trained. "
                "If it is, set train_ready to True."
            )

        # Reload classifier properties (to pick up previous active stage)
        self.Classifier.get_config_dict()
        self._load_Classifier_config()
        # Load stage properties (parent_stage_id from classifier_properties.active_stage_id)
        self._load_stage_config()

        # Prepare configurations
        training_catalog = self.stage_properties.training_catalog
        dataset_class_str = self.stage_properties.dataset_class_str
        train_dataset_config = self.stage_properties.dataset_config.copy()

        # The training and validation catalogs are added to the tracked catalog in order to store the evolution of the metrics on them. 
        trak_cfg = copy.deepcopy(train_dataset_config)
        
        trak_cfg["samples_used"] = self.stage_properties.samples_used_for_validation
        self.Classifier.add_active_val_cat_live(training_catalog, dataset_class_str, trak_cfg)
        
        self.Classifier.add_active_val_cat_live(
            self.stage_properties.validation_like_train_catalog,
            dataset_class_str,
            trak_cfg
        )

        # Reload classifier and stage configs after updating live cats: is this necessary?
        self.Classifier.get_config_dict()
        self._load_Classifier_config()
        self._load_stage_config()

        # Generate stage_id: the stage id is set once all the properties of the stage are set. Two exactly equal stages(mistake) share the same id, throwing an error. and persist stage config
        self.set_stage_id()
        self.write_stage_config()

        # Build train dataloader and model (load_parent uses parent_stage_id set earlier)
        train_loader = self._load_dataloader(
            catalog_name=training_catalog,
            dataset_class_str=dataset_class_str,
            dataset_config=train_dataset_config,
            val=False
        )
        
        model = self._load_model(load_parent=True)
        model = model.type(getattr(torch, self.stage_properties.NN_datatype)).to(self.device)

        # Prepare validation loaders
        val_loaders, val_cats = [], []
        for cat, ds_cls, ds_cfg in self.stage_properties.active_val_cats_live:
            loader = self._load_dataloader(
                catalog_name=cat,
                dataset_class_str=ds_cls,
                dataset_config=ds_cfg,
                val=True
            )
            val_loaders.append(loader)
            val_cats.append(cat)

        criterion = self._make_loss_function(self.stage_properties.loss_function)
        
#         base_opt=  self._make_optimizer(model=model, optimizer=self.stage_properties.optimizer)

#         optimizer = self._make_optimizer(model=model, optimizer=self.stage_properties.optimizer)
        
#         #=======================================LEARNING RATE SEARCH LOGIC================================
        
#         #=================================================================================================
        
        
        
#         LR_FINDER_END_LR    = 10
#         LR_FINDER_ITERS     = 20        # mini-batches
#         DIV_FACTOR          = 25.0         # base_lr  = max_lr / DIV_FACTOR
#         FINAL_DIV_FACTOR    = 1e4          # final_lr = base_lr / FINAL_DIV_FACTOR
#         PCT_START           = 0.2          # % of steps for upward LR ramp
        
#         #base_opt = optim.SGD(model.parameters(), lr=LR_FINDER_START_LR, momentum=0.9, weight_decay=5e-4)
#         lr_finder = LRFinder(model, optimizer, criterion, device=self.device)

#         print("▶️  Running LR-Finder …")
#         print(min(f"Min {LR_FINDER_ITERS, len(train_loader)}"))
#         print(LR_FINDER_ITERS)
#         print(len(train_loader))
        
#         lr_finder.range_test(train_loader,
#                              end_lr=LR_FINDER_END_LR,
#                              num_iter=min(LR_FINDER_ITERS, len(train_loader)),
#                              step_mode="exp")
        
        
        
        
#         fig, ax = plt.subplots()
#         ax.plot(lr_finder.history["lr"], lr_finder.history["loss"])
#         ax.set_xscale("log"); ax.set_xlabel("learning-rate"); ax.set_ylabel("loss"); ax.set_title("LR-Finder curve")
        
#         # Heuristic choice: LR just before the loss minimum
#         losses = torch.tensor(lr_finder.history["loss"])
#         lrs    = torch.tensor(lr_finder.history["lr"])
        
        
# #         def pick_lr_steepest_descent(lrs: torch.Tensor, losses: torch.Tensor, smooth: int = 5) -> float:
# #             """Return LR at the point where the *negative* slope |dLoss/d logLR| is largest.

# #             A simple rolling‑median is applied to damp noise.
# #             """
# #             # smooth losses with median filter on CPU for simplicity
# #             losses_cpu = losses.cpu()
# #             smoothed = losses_cpu.clone()
# #             for i in range(len(losses_cpu)):
# #                 start = max(0, i - smooth // 2)
# #                 end   = min(len(losses_cpu), i + smooth // 2 + 1)
# #                 smoothed[i] = losses_cpu[start:end].median()

# #             log_lrs = torch.log10(lrs.cpu())
# #             # numerical derivative w.r.t. log LR
# #             grads = torch.diff(smoothed) / torch.diff(log_lrs)
# #             # steepest descent = most negative slope (min value of grads)
# #             steep_idx = int(torch.argmin(grads))
# #             return float(lrs[steep_idx])
        
# #         lr_best = pick_lr_steepest_descent(lrs, losses)

#         best_i = int(torch.argmin(losses))
#         lr_peak = float(lrs[best_i])
#         print(f"📈  Suggested max_lr (lr_peak) ≈ {lr_peak:.2e}")
#         lr_best=lr_peak
        
#         print(f"📈  Suggested max_lr (lr_peak) ≈ {lr_best:.2e}")
              
#         lr_finder.reset()      # restores initial weights

        
        
#         base_lr = lr_best / DIV_FACTOR
        
#NOTE: very very very bad way of programming.

        #user_set_learning_rate=self.stage_properties.learning_rate
        #self.stage_properties.learning_rate=base_lr
        
        
        
        optimizer = self._make_optimizer(model=model, optimizer=self.stage_properties.optimizer)

        print("hardcoding lr_best")
#         scheduler = OneCycleLR(optimizer,
#                                max_lr=user_set_learning_rate,#lr_best/1.,
#                                epochs=self.stage_properties.epochs,
#                                steps_per_epoch=len(train_loader),
#                                pct_start=PCT_START,
#                                anneal_strategy="cos",
#                                div_factor=DIV_FACTOR,
#                                final_div_factor=FINAL_DIV_FACTOR)
        
        
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',        # or 'max' if you’re monitoring e.g. accuracy
            factor=1/10.,        # multiply LR by 0.2 when triggered
            patience=15,       # wait 10 epochs with no improvement
            verbose=True,      # prints a message when LR is reduced
            threshold=1e-3,    # “improvement” threshold (optional)
            threshold_mode='rel'  # relative change (optional)
        )
        #=======================================LIVE EVALUATIONS LOGIC====================================
        
        #=================================================================================================
        lr_history = []

                
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="francescocitterio99-max-planck-society",
            # Set the wandb project where this run will be logged.
            project=self.classifier_name,
            # Track hyperparameters and run metadata.
            config={
                "no_config": "no_config"
            },
        )
        wandb.define_metric("epoch")                                      # or "step" if you prefer
        for cat in val_cats:                                              # one overlayable metric per cat
            for m in ["loss", "accuracy", "f1", "precision", "recall"]:
                wandb.define_metric(f"{m}/{cat}", step_metric="samples_seen")  
                
        def evaluate_model():
            model.eval()
            with torch.no_grad():
                losses = {}
                for idx, loader in enumerate(val_loaders):
                    cat = val_cats[idx]
                    total_loss = correct = count = 0
                    preds, labs = [], []
                    
                    for inputs, labels in loader:
                        
                        if not torch.isfinite(inputs).all():
                            print(f"🚨 Non-finite in loader {cat}, validation catalogs; skipping this batch.")
                            continue  # skip forward/backward/update on this batch
                        #assert torch.isfinite(inputs).all(), "🚨 Inf or NaN in inputs!"

                        inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                        out = model(inputs)
                        batch_loss = criterion(out, labels)
                        total_loss += batch_loss.item() * inputs.size(0)
                        pr = torch.argmax(out, dim=1)
                        
                        correct += (pr == labels).sum().item()
                        count += labels.size(0)
                        preds.extend(pr.cpu().numpy())
                        labs.extend(labels.cpu().numpy())
                        
                    avg_loss = total_loss / count
                    accuracy = correct / count
                    arr_p, arr_l = np.array(preds), np.array(labs)

                    avg_method = 'binary' 
                    
                    #store also in local var?
                    losses[cat] = avg_loss
                    
                    
                    avg_method = "binary"
                    f1        = f1_score(arr_l, arr_p, average=avg_method)
                    precision = precision_score(arr_l, arr_p, average=avg_method, zero_division=0)
                    recall    = recall_score(arr_l, arr_p, average=avg_method, zero_division=0)

                    # keep your in-memory trackers untouched
                    self.live_metrics.running_loss [cat].append(avg_loss)
                    self.live_metrics.accuracy     [cat].append(accuracy)
                    self.live_metrics.f1_score     [cat].append(f1)
                    self.live_metrics.precision    [cat].append(precision)
                    self.live_metrics.recall       [cat].append(recall)
                    self.live_metrics.confusion_matrix[cat].append(
                        confusion_matrix(arr_l, arr_p, labels=[0, 1]).tolist()
                    )
                    # ---- W&B logging ----------------------------------------------------
                    wandb.log(
                        {
                            f"loss/{cat}":      avg_loss,
                            f"accuracy/{cat}":  accuracy,
                            f"f1/{cat}":        f1,
                            f"precision/{cat}": precision,
                            f"recall/{cat}":    recall,

                            # confusion matrix as an interactive plot
                            f"conf_mat/{cat}": wandb.plot.confusion_matrix(
                                y_true   = arr_l,
                                preds    = arr_p,
                                class_names = ["0", "1"],
                                title    = f"{cat} confusion"
                            ),
                        }, 
                        step=self.samples_seen
                    )
                    
                return losses
                
        
        
        # Initialize metrics and early stopping
        self._initialize_running_metrics()
        best_loss = float('inf')
        patience_counter_lr = 0
        patience = 30 if early_stopping else float('inf')
        best_model_state = None
        model_path = _get_model_parmas_path(
            classifier_name=self.classifier_name,
            stage_id=self.stage_properties.stage_id
        )
        
        # Training loop
        early_stop = False
        for epoch in range(self.stage_properties.epochs):
            if early_stop:
                break
                
            model.train()
            batch_bar=tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{self.stage_properties.epochs}", unit="batch")
            self.loss_window = getattr(self, "loss_window", deque(maxlen=200))  # ≈200 most recent batches
            self.gradient_array= getattr(self, "gradient_array", [])
                                        
            #batch index, images, labels
            for bidx, (imgs, lbls) in enumerate(batch_bar):
              
                    
                    
                #imgs, lbls = imgs.to(self.device), lbls.to(self.device).long()
                
                
                
                
                if not torch.isfinite(imgs).all():
                    print(f"🚨 Non-finite inputs in batch {bidx}; skipping this batch.")
                    continue  # skip forward/backward/update on this batch
                    
                    
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, lbls)
                
                # if not torch.isfinite(loss):
                #     print("🚨 loss went bad:", loss)
                #     import torch.autograd as ag; ag.detect_anomaly()
                
                # first_img   = imgs[0].detach().cpu()       # shape: (C, H, W)
                # first_label = lbls[0].item()

                self.samples_seen = bidx * self.stage_properties.batch_size + epoch * len(train_loader.dataset)
                #print(f"samples seen={self.samples_seen}")
                # if you’ve normalized your inputs, undo it here:
                # mean = torch.tensor(self.stage_properties.mean)[:, None, None]
                # std  = torch.tensor(self.stage_properties.std)[:, None, None]
                # first_img = (first_img * std + mean).clamp(0, 1)

                # convert from torch.Tensor to HWC NumPy
                #first_img_np = first_img.permute(1, 2, 0).numpy()

                #log it to W&B
#                 wandb.log({
#                     "batch_first_image": wandb.Image(
#                         first_img_np,
#                         caption=f"Label: {first_label}"
#                     )
#                 }, step=self.samples_seen)
                
                
                loss.backward()
                
                total_norm = torch.nn.utils.clip_grad_norm_(          # max_norm=∞ ⇒ no clipping
                    model.parameters(),
                    max_norm=float("20")
                )
                

                self.gradient_array.append(total_norm.cpu().numpy())
                
                self.live_metrics.running_loss[training_catalog].append(loss.item())
                self.loss_window.append(loss.item())
                
    

                
                
                
                
                
                #===============The follwing periodic checks sobstitute those ususally done at end epoch, like with infinite data==========
                if bidx % self.stage_properties.jump_batch_val == 0:
                    self.live_metrics.batch_number.append(bidx)
                    
                    run.log({"samples_seen": self.samples_seen}, step=self.samples_seen)

                    self.live_metrics.sample_number.append(self.samples_seen)
                    val_losses = evaluate_model()
                    #re-set to train after evaluation
                    model.train()
                    vloss = val_losses[self.stage_properties.validation_like_train_catalog]
                    #calc training loss of last batches. Statistical meaning depends on the size of the batch...
                    train_loss_avg = sum(self.loss_window) / len(self.loss_window)
                    
                    run.log({"Validation loss": vloss}, step=self.samples_seen)
                    print(f"Training_loss= {train_loss_avg}, Validation_loss={vloss}")
                    
                    
                    
                    if vloss < best_loss:
                        best_loss, patience_counter = vloss, 0
                        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                        print(f"New best validation loss: {best_loss:.6f}")
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping at epoch {epoch}, batch {bidx}")
                            self.stage_properties.early_stopping = self.samples_seen
                            self.write_stage_config()
                            early_stop = True
                            break
                    scheduler.step(vloss)
                            
                optimizer.step()
                #scheduler.step()
                lr_history.append(scheduler.get_last_lr()[0])
                

                current_lr = optimizer.param_groups[0]['lr']
                
                run.log({"total_norm": total_norm, "loss": loss, "learning_rate": current_lr},  step=self.samples_seen)

                
                
                self.live_metrics.learning_rate.append(current_lr)


            batch_bar.close()
                    
        # Plot One-Cycle LR schedule
        plt.figure(); plt.plot(lr_history)
        plt.yscale("log"); plt.xlabel("training step"); plt.ylabel("learning-rate"); plt.title("One-Cycle LR schedule")
        plt.show()     
        
        # Plot the history of the gradient
        plt.figure(); plt.plot(self.gradient_array)
        plt.yscale("log"); plt.xlabel("training step"); plt.ylabel("gradient norm"); plt.title("Gradient norm tracking")
        plt.show()     
              
              
              
        self.live_metrics.sample_number.append(self.samples_seen)
        # Final evaluation and save parameters
        final_losses = evaluate_model()
        run.log({"total_norm": total_norm, "loss": loss, "learning_rate": current_lr}, step=self.samples_seen)

        fl = final_losses[self.stage_properties.validation_like_train_catalog]
        if fl < best_loss:
            best_model_state = {k: v.clone() for k,v in model.state_dict().items()}
            print(f"Final model is best with loss {fl:.6f}")
            
        state_dict_to_save = best_model_state or model.state_dict()
        torch.save(state_dict_to_save, model_path)
        print(f"Model parameters saved to {model_path}")
        # save the moments of the optimizer to carry on to the new stage.
        torch.save(optimizer.state_dict(), model_path.replace(".pt", "_optim.pt"))

        

        # Finalize
        run.finish()

        self.stage_properties.is_trained = True
        self.Classifier.set_active_stage_id(self.stage_properties.stage_id)
        self.write_stage_config()
        live_metrics_path = os.path.join(
            self.Classifier.classifier_path,
            "stages", str(self.stage_properties.stage_id),
            "live_metrics.json"
        )
        with open(live_metrics_path, "w") as f:
            json.dump(self.live_metrics.__dict__, f, indent=4)
        print("Live metrics saved.")

        return self.compute_validation_metrics_offline()
    
    
    
    
    
    def get_model(self, compiled=False):
        #load the model from the registry
        model_str = self.classifier_properties.NN_model_str
        model_config = self.classifier_properties.NN_config.copy()
        # check that the model is in the registry
        try:
            NN_model_class = MODEL_REGISTRY.get(model_str)
        except KeyError:
            raise ValueError(f"Model {model_str} not found in registry")
            
        model = NN_model_class(**model_config)
        params_path= _get_model_parmas_path(classifier_name=self.classifier_name, stage_id=self.stage_id)

        
        if compiled==True:
            
            model = torch.compile(
                model,
                backend="inductor",
                fullgraph=False,      # or drop this if you don't need full-graph
                dynamic=False        # or False if your input size is fixed
            )
            
            model.load_state_dict(torch.load(params_path))        

            model.to(self.device)

            return model
        if compiled==False:
            
            state_dict = torch.load(params_path, map_location=self.device)
            
            # strip off the “_orig_mod.” prefix
            clean_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

            model.load_state_dict(clean_sd)
            model.to(self.device)

            return model
            
            
    
    def predict(self, images_batch):

        model = self._load_model(load_parent=False)
        model.eval()

        inputs=images_batch.to(self.device)
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)

        outputs= model(inputs)

        return outputs