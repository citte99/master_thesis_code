"""
    The classifier is represented by:

    what is common bewteen all training stages:
    - the model
         - the architecture and its configuration

    - active metrics

    the stages
    -status is the tree


    - all about data ingested
         -the catalog of data for training,
         -the catalog of data for validation,
         -the dataloaders used
            - the dataloader settings


    - parameters initialization
        -initializations or based on previous stage

    - training parameters
         - learning rate
         - batch size
         - number of epochs
         - optimizer
         - loss function
         - early stopping
        
         
         - all about validation
         - jump batch
            - checkpointing
            - metrics
    

    - to solve the online-offline mess, we store the checkpoints 




"""


"""
Notes on code corrections:

- Uses class logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) what is assigned to __name__? 



- Uses abc, ABC and abstracmethod

- Uses dataclass, asdict
- uses typing Dict, list, optional, tuple, union, any

-uses enum import Enom



===================================================Current state===================================
classifier properties dataclass

    get_example_config
    _initialize classifier
    _set_active_stage_it
    _update classifier properties
    _get_config_dict
    _get_classifier_name
    _set_active_val_cats
    _add_active_val_cat_live
    _graph management -> question: how to manage this?

    _get live stage history -> kill and go full weights and biases. Problems with online logging

    _all the plotting stuff is completely useless



SubClassifier does:
    -classify images. 
    

we need to be able to efficiently manage classifiers, training stages, parameters, store complete intializaitons
for reproducibility, etc.



ClassifierManager:
    -get_classifier()->  SubClassifier



"""
from abc import ABC, abstractmethod
from enum import Enum

class NN_model(ABC):
    pass



class SubClassifier:
    """
    Base would just be the nn model with the loaded parameters
    It whould be nice to have available some summaries about this particular classifier, 
    like what it was trained on, or its performance on particular test sets. 

    """
    def __init__(self)
        self.NN_model: NN_model = None
        self.NN_parameters = None
        




# I will follow the pattern of separating the config managment from the class itself.
# the config manager has attributes read and save, and is dataclass. while the class itself has
# method cls update.

class TrainingRun:
    """

    """

    


    def get_sub_classifier(self, stage_id)-> SubClassifier:
        #could be last, or one in particular. 
        pass


class TrainingStage:
    pass