import os
import numpy as np

from config import TRAINED_CLASSIFIERS_DIR
from dataclasses import dataclass
from .util import _validate_classifier_config
from .directed_graph import DirectedGraph
import json
import shutil
from .shared_data_structures import LiveMetrics, TrainingCompletedMetrics
from .training_stage_development import Stage


@dataclass
class ClassifierProperties:
    classifier_name: str
    NN_model: str   
    NN_config: dict
    active_stage_id: str
    active_val_cats_live: list[tuple] #list of tuples (catalog, dataset_class)
    



class SubstructureClassifier:
    example_classifier_config={
        "classifier_name": "test_classifier2",
        "NN_model": "ResNet50",
        "NN_config": {},
        "active_val_cats_live": [
            [
                "SIS_10e9_sub_test",
                "NoNoiseDataset",
                {
                    "grid_width_arcsec": 6.0,
                    "grid_pixel_side": 100
                }
            ],
            [
                "SIS_10e8_sub_test",
                "NoNoiseDataset",
                {
                    "grid_width_arcsec": 6.0,
                    "grid_pixel_side": 100
                }
            ]
        ],
        #"active_stage_id": null, #initialized by the classifier
    }
    @staticmethod
    def get_example_classifier_config():
        return SubstructureClassifier.example_classifier_config
    
    def __init__(self, classifier_name, config_dict=None):
        self.classifier_path = os.path.join(TRAINED_CLASSIFIERS_DIR, classifier_name)

        if os.path.exists(self.classifier_path) and config_dict is None:
            with open(os.path.join(self.classifier_path, "classifier_config.json"), "r") as f:
                self.config = json.load(f)
        elif config_dict is not None:
            # Check if the classifier directory already exists
            if os.path.exists(self.classifier_path):
                raise FileExistsError(f"Classifier {classifier_name} already exists. Provide a different name or delete the existing classifier.")
            self.config = config_dict
            self.config["active_stage_id"] = None
            # create the classifier directory
            os.makedirs(self.classifier_path, exist_ok=False)
            # create the stages directory
            os.makedirs(os.path.join(self.classifier_path, "stages"), exist_ok=False)
            # dump the config dict to the classifier config file
            with open(os.path.join(self.classifier_path, "classifier_config.json"), "w") as f:
                json.dump(self.config, f, indent=4)
        else:
            raise FileNotFoundError(f"Classifier {classifier_name} does not exist. Provide a config dict to create it.")

        try:
            self._initialize_classifier()
        except Exception as e:
            print(f"An error occurred: {e}")
            print("The classifier config is missing some keys or has other issues. Please check the config file.")
            if config_dict is not None:
                print("The provided config dict may also be invalid.")
                #delete the classifier directory
                shutil.rmtree(self.classifier_path)
                print("Classifier directory deleted.")
            raise

    def _initialize_classifier(self):
        _validate_classifier_config(self.config)
        self.classifier_properties = ClassifierProperties(
            classifier_name=self.config["classifier_name"],
            NN_model=self.config["NN_model"],
            NN_config=self.config["NN_config"],
            active_stage_id=self.config["active_stage_id"],
            active_val_cats_live=self.config["active_val_cats_live"]
        )
        self._update_graph()

    def set_active_stage_id(self, active_stage_id):
        self.classifier_properties.active_stage_id = active_stage_id
        self._update_classifier_properties()

    def _update_classifier_properties(self):
        # dump the new dict to the classifier config file
        new_config = self.classifier_properties.__dict__
        # remove the previous config file
        if os.path.exists(os.path.join(self.classifier_path, "classifier_config.json")):
            os.remove(os.path.join(self.classifier_path, "classifier_config.json"))
        
        with open(os.path.join(self.classifier_path, "classifier_config.json"), "w") as f:
            json.dump(new_config, f, indent=4)
            print("should have updated the classifier config file")

    def get_config_dict(self):
        """Reload the classifier_config.json from disk before returning it."""
        config_path = os.path.join(self.classifier_path, "classifier_config.json")
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # (Re–initialize in–memory properties if you want to keep them in sync)
        self._initialize_classifier()

        return self.config
    
    def get_classifier_name(self):
        return self.classifier_properties.classifier_name

    def set_active_val_cats(self, active_val_cats):
        self.active_val_cats = active_val_cats
        self.classifier_properties.set_active_val_cats(active_val_cats)
        self._update_classifier_properties()

    def add_active_val_cat_live(self, catalog, dataset_class, dataset_config):
        """
        Add a new active validation catalog to the classifier.
        Ensure that the catalog is unique.
        """
        new_entry_full = [catalog, dataset_class, dataset_config]
        new_entry = catalog
        print("forcing only one entry per catalog in add activ val cat live, substructure classifier dev")
        already_active_val_cats = [cat[0] for cat in self.classifier_properties.active_val_cats_live]
        #if new_entry in self.classifier_properties.active_val_cats_live:
        if new_entry in already_active_val_cats:

            print(f"Catalog {catalog} already exists in active validation catalogs.")
        else:
            self.classifier_properties.active_val_cats_live.append(new_entry_full)
        self._update_classifier_properties()

    def _update_graph(self):
        if hasattr(self, 'graph'):
            self.graph.update_graph()
        else:
            # If graph does not exist, create it
            self._build_graph()


    def _build_graph(self):
       stage_paths= os.path.join(self.classifier_path, "stages")
       self.graph= DirectedGraph(stages_path= stage_paths)

        
        


#===============================================================VISUALIZATION===================================================================
    def _load_history_data(self):
        self._update_graph()
        stage_ids = self.graph.get_nodes_list()
        
        # Filter out None values and ensure all stage_ids are strings
        stage_ids = [str(stage_id) for stage_id in stage_ids if stage_id is not None]
        
        self.live_metrics = {stage_id: LiveMetrics() for stage_id in stage_ids}
        self.offline_metrics = {stage_id: TrainingCompletedMetrics() for stage_id in stage_ids}
        
        for stage_id in stage_ids:
            # Open the json of live metrics
            stage_path = os.path.join(self.classifier_path, "stages", stage_id)
            live_metrics_path = os.path.join(stage_path, "live_metrics.json")
            
            if os.path.exists(live_metrics_path):
                with open(live_metrics_path, "r") as f:
                    live_metrics_data = json.load(f)
                # Convert to LiveMetrics
                self.live_metrics[stage_id] = LiveMetrics(**live_metrics_data)
            else:
                print(f"Live metrics not found for stage {stage_id}")
                self.live_metrics[stage_id] = None
                
            # Open the json of offline metrics
            offline_metrics_path = os.path.join(stage_path, "training_completed_metrics.json")
            if os.path.exists(offline_metrics_path):
                with open(offline_metrics_path, "r") as f:
                    offline_metrics_data = json.load(f)
                # Convert to TrainingCompletedMetrics
                self.offline_metrics[stage_id] = TrainingCompletedMetrics(**offline_metrics_data)
            else:
                print(f"Offline metrics not found for stage {stage_id}")
                self.offline_metrics[stage_id] = None
        
    def _get_live_stage_history(self, stage_id=None, metric=None):
        """
        X_axis: need to compose the x axis for different training sessions.
        Each training session recorded live metrics at a particular batch interval.
        Each training session may have had different number of epochs. 
        Since we have potentially infinite data, we should never have more than one epoch...
        But still.
        x_axis should be the epoch.number_of_samples_seen
        they should be concatenated, and displayed in the x axis disregarding the numerical order.
        """
        if stage_id is None:
            stage_id = self.classifier_properties.active_stage_id
        # get the genaology of the stage
        self._update_graph()
        stage_genalogy=self.graph.find_route_to_origin(node_id=stage_id)
        
        self._load_history_data()
        x_axis= []
        merged_data_dict= {}

        active_catalogs_names=[]

        for i in range(len(self.classifier_properties.active_val_cats_live)):
            active_catalogs_names.append(self.classifier_properties.active_val_cats_live[i][0])
            merged_data_dict[active_catalogs_names[i]]= []

        # merged datas dictionary: [dict_name]:[y_data]
        for i, stage_id in enumerate(stage_genalogy):
            x_axis_new_chunk= self.live_metrics[stage_id].sample_number
            x_axis+= [f"{i}.{x}" for x in x_axis_new_chunk]

            for catalog in active_catalogs_names:
                # make the x axis in notation i.sample_number
                # Not all the stages have data for all the catalogs. Those should be filled with placeholders: -0.2
                if catalog in getattr(self.live_metrics[stage_id], metric):
                    merged_data_dict[catalog] += getattr(self.live_metrics[stage_id], metric)[catalog]
                else:
                    # Append placeholder values with the same length as x_axis_new_chunk
                    placeholder_length = len(x_axis_new_chunk)
                    merged_data_dict[catalog] += [-0.2] * placeholder_length
        
        beginning_of_stage_indexes = [i for i, value in enumerate(x_axis) if value.endswith(".0")]

        #self._plot_concatenated_data(x_axis, merged_data_dict)
        return x_axis, merged_data_dict, stage_genalogy, beginning_of_stage_indexes
   
    def _plot_concatenated_data(self):
        """
        x_axis: list of strings
        merged_data_dict: dict of lists
        """
        print("Right now considering only selection by catalog and not by dataset in plotting in substructure_classifier_development.py")
        all_traked_catalogs= [cat[0] for cat in self.classifier_properties.active_val_cats_live]

        from matplotlib import pyplot as plt
        #running loss plot for the trainining catalogs: various lenghts
        print("running loss plot for the training catalogs")
        for catalog in all_traked_catalogs:
            x_axis, merged_dict, _, beginning_of_stage_indexes=self._get_live_stage_history( metric="running_loss")
            number_of_samples_seen= len(merged_dict[catalog])
            if number_of_samples_seen==0:
                print(f"Catalog {catalog} was not used for training.")
                continue
            x_axis_running_loss=np.arange(0, len(merged_dict[catalog]))
            plt.figure(figsize=(20, 5))                   
            plt.plot(x_axis_running_loss, merged_dict[catalog])
            plt.title(f"Live running loss for {catalog}")
            plt.xlabel("batch number")
            plt.ylabel("Running loss")
            plt.xticks(rotation=90)  # Orient the x-axis ticks vertically
            plt.show()

        print(f"hard coded shown metrics at _plot_concatedated_data in substructure_classifier_development.py")

        all_metrics = ["accuracy", "f1_score", "precision", "recall"]


        stages_initialized=[]
        stage_genealogy=self.graph.find_route_to_origin(node_id=self.classifier_properties.active_stage_id)
        for stage_id in stage_genealogy:
            stages_initialized.append( Stage(stage_id=stage_id, classifier_instance=self))


        for metric in all_metrics:
            plt.figure(figsize=(30, 10))
            already_plotted_labels=False
            for catalog in all_traked_catalogs:
                x_axis, merged_dict, stage_genealogy, beginning_of_stage_indexes=self._get_live_stage_history( metric=metric)
                plt.plot(x_axis, merged_dict[catalog], label=catalog)
                y_max = plt.ylim()[1]
                y_min = plt.ylim()[0]
                y_range = y_max - y_min
                if not already_plotted_labels:
                    for i, pos in enumerate(beginning_of_stage_indexes):
                            stage= stages_initialized[i]
                            plt.axvline(x=pos, color='green', linestyle='--', alpha=0.7)
                            stage_info = stage.get_stage_summary_small()
                            info_text = f"Stage id begin {stage_genealogy[i][:5]}\n"
                            info_text += "\n".join([f"{k}: {v}" for k, v in stage_info.items()])
                
                            # Position text to the right of the line
                            # Adjust x and y offset as needed
                            x_offset = 0.5  # Horizontal distance from the line
                            y_pos = y_min-0.5  # Vertical position (adjust as needed)
                    
                            plt.text(pos + x_offset, y_pos, info_text,
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'),
                            verticalalignment='top')
                            # You could also add a label if desired
                            stage_num = x_axis[pos].split('.')[0]
                            plt.text(pos, plt.ylim()[1]*0.95, f"Stage {stage_num}", 
                                    rotation=90, verticalalignment='top')
                    already_plotted_labels=True
            plt.axhline(y=0.5, color='red', linestyle='--')
            plt.title(f"Live {metric} for all tracked catalogs")
            plt.xlabel("Samples seen in training")
            plt.ylabel(metric)
            plt.legend()
            plt.ylim(0.0,1.0)
            plt.xticks(rotation=90)  # Orient the x-axis ticks vertically
            plt.show()

   
#========================STAGE MANAGEMENT AND NEW STAGES==========================