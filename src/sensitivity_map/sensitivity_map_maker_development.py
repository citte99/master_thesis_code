import torch
import copy
import json
from matplotlib import pyplot as plt
import inspect

from shared_utils import _grid_lens
from substructure_classifier.training_stage_development import Stage
from deep_learning import DATASET_REGISTRY, custom_dataloader



class SensitivityMapMakerDevelopment:
    def __init__(self,
                system_dict,
                dataset_str,
                dataset_config,
                Stage_obj,
                substructure_to_test_dict=None,
                eval_grid_width_arcsec=None,
                eval_grid_pixels_side=None,
                eval_mode=None,
                sub_detec_criterion=None,
                sub_detect_criterion_threshold=None,
                criterion_low_bound=None,
                criterion_high_bound=None,
        ):
        self._sanitize_system_dict(system_dict)
        self._sanitize_substructure_dict(substructure_to_test_dict)

        self._set_sub_detection_criterion(sub_detec_criterion)

        self._load_Stage(Stage_obj)

        self._set_target_grid(eval_grid_width_arcsec, eval_grid_pixels_side)
        self._load_dataloaders(dataset_str, dataset_config)
        self._set_eval_mode(eval_mode)

        print("Could add a better check for the substructure detection criterion.")
        self.sub_detection_criterion_threshold= sub_detect_criterion_threshold
        self.criterion_low_bound= criterion_low_bound
        self.criterion_high_bound= criterion_high_bound


#=================Input initialization==============================
    def _set_target_grid(self, eval_grid_width_arcsec, eval_grid_pixels_side):
        # Regardless of the particular points over which we evaluate the sensitivity,
        # one of the final result will be a grid of points over the image plane.

        self.target_grid_width_arcsec = eval_grid_width_arcsec
        self.target_grid_pixels_side = eval_grid_pixels_side
        self.target_grid= _grid_lens(eval_grid_width_arcsec, eval_grid_pixels_side, device="cpu")
       


    def _load_Stage(self, Stage_obj):
        # check if stage obj is a Stage object
        if isinstance(Stage_obj, Stage):
            self.Stage_obj = Stage_obj
        else:
            raise ValueError(f"Stage_obj is not a Stage object. It is {type(Stage_obj)}")
        
        # check that the stage is trained
        if not self.Stage_obj.stage_properties.is_trained:
            raise ValueError("Stage_obj is not trained. Please train the stage before using it.")

    def _sanitize_system_dict(self, system_dict):
        self.no_substructure_system = copy.deepcopy(system_dict)
        # Filter out any substructures
        mass_components=self.no_substructure_system["lens_model"]["mass_components"] 
        sanitized_components = []
        for comp in mass_components:
            if not comp.get("is_substructure", False):
                sanitized_components.append(comp)
        self.no_substructure_system["lens_model"]["mass_components"] = sanitized_components
        

    def _sanitize_substructure_dict(self, substructure_to_test_dict):
        #print("No check on the substructure dictionary yet")
        self.substructure_to_test_dict = copy.deepcopy(substructure_to_test_dict)

    def _set_sub_detection_criterion(self, sub_detec_criterion):
        params = self.substructure_to_test_dict.get("params", {})      # {} avoids KeyError if "params" is missing
        if sub_detec_criterion in params:
            self.sub_detection_criterion = sub_detec_criterion
        else:
            raise ValueError(f"Substructure detection criterion {sub_detec_criterion} not found in substructure parameters.")
        
    def _load_dataloaders(self, dataset_str, dataset_config):
        dataset_config_temp= copy.deepcopy(dataset_config)

        # testing a random catalog, then the dataloader is dropped.
        dataset_config_temp["catalog_dict"] = self._sys_dict_to_cat_dict(self.no_substructure_system)
        dataset = DATASET_REGISTRY.get(dataset_str)(**dataset_config_temp)
        my_dataloader = custom_dataloader(dataset=dataset, batch_size=1, shuffle=False)
        
        self.dataset_str = dataset_str
        self.dataset_config = dataset_config

        self.dataset_no_noise = DATASET_REGISTRY.get("NoNoiseDataset")(
            grid_width_arcsec=self.target_grid_width_arcsec,
            grid_pixel_side=self.target_grid_pixels_side,
            device="cpu",
            catalog_dict=self._sys_dict_to_cat_dict(self.no_substructure_system),
        )


        self.dataloader_no_noise = custom_dataloader(dataset=self.dataset_no_noise, batch_size=1, shuffle=False)
        

        # Load no substructure noisy image
        self.dataset_eval = DATASET_REGISTRY.get(self.dataset_str)(catalog_dict=self._sys_dict_to_cat_dict(self.no_substructure_system), **self.dataset_config)
        self.dataloader_eval = custom_dataloader(dataset=self.dataset_eval, batch_size=1, shuffle=False)

        

    def _set_eval_mode(self, eval_mode):
        modes_available= SensitivityMapMakerDevelopment._get_available_eval_modes()
        if eval_mode not in modes_available:
            raise ValueError(f"Eval mode {eval_mode} not available. Available modes are: {modes_available}")
        self.eval_mode= eval_mode



#====================================VISUALIZATIONS================================================

    def _visualize_clean_and_dirty_image(self):
        # Load no substructure no noise image, (which size of images?)
        images_batch = next(iter(self.dataloader_no_noise))[0]
        no_noise_image = images_batch[0, 0, :, :].cpu().numpy()

        # Load no substructure noisy image
        images_batch = next(iter(self.dataloader_eval))[0]
        noisy_image = images_batch[0, 0, :, :].cpu().numpy()

        # Plot the images side by side
        _, axes = plt.subplots(1, 2, figsize=(12, 6))
        im0 = axes[0].imshow(no_noise_image, cmap="viridis")
        axes[0].set_title("No substructure, no noise")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(noisy_image, cmap="viridis")
        axes[1].set_title("No substructure, noisy")
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.show()

    def _plot_sensitivity_map(self, values_on_grid, grid):
        # Plot the sensitivity map
        plt.imshow(values_on_grid, cmap="viridis")
        plt.colorbar()
        plt.title("Sensitivity Map")
        plt.xlabel("X (arcsec)")
        plt.ylabel("Y (arcsec)")
        plt.show()

#================================COMPUTING SENSITIVITY MAP================================================
    
    @staticmethod
    def _get_available_eval_modes():
        modes= [
            #"optimize_integral_over_field",
            "compute_over_simple_grid"
        ]
        return modes

    def _first_grid_to_eval(self, mode, args):
        if mode =="compute_over_simple_grid":
            # This is the default mode, just a simple grid
            # The grid is already defined in the constructor
            # The grid is a 2D array of points over the image plane
            # The points are in arcseconds
            # The grid is centered on the center of the image
            # The grid is in the same coordinate system as the image
            return self.target_grid

        # While the output sensitivity map is a grid,
        # the scientifical quantity we need to approximate better
        # is probably the integral of the sensitivity map over the grid.
        pass
        
    def _new_grid_from_old_grid(self, old_grid, old_grid_values, mode):
        # This is a little bit stupid, but follows previous code.
        # If no bottleneck, fine.
        pass
    


    def _grid_of_pos_sub_to_cat_dict(self, grid_of_pos_sub, criterion_values):
        # This is a little bit stupid, but follows previous code.
        # If no bottleneck, fine.

        #repeat the base dict in the cat_dict as many times as the number of positions in the grid
        cat_dict= {"SL_systems": []}
        for i in range(grid_of_pos_sub.shape[0]):
            #copy the base dict
            system_dict= copy.deepcopy(self.no_substructure_system)
            #add the substructure to the system dict
            system_dict["lens_model"]["mass_components"].append(self.substructure_to_test_dict)
            #add the position of the substructure to the system dict
            system_dict["lens_model"]["mass_components"][-1]["position"] = grid_of_pos_sub[i, :]
            #add the criterion value to the system dict
            raise(NotImplementedError("Criterion values should be values on the grid, see the following line."))
            system_dict["lens_model"]["mass_components"][-1][self.sub_detection_criterion] = criterion_values[i]
            #add the system dict to the catalog dict
            cat_dict["SL_systems"].append(system_dict)
        
        # after this, we can update the catalog of the dataset, and compute with a sensible batch size (in particular if using broadcasting).
        return cat_dict
        
    
    def _update_cat_data_set_load(self, cat_dict, batch_size_data_loader):
        self.batch_size_data_loader= batch_size_data_loader

        self.dataset_eval.update_catalog_dict(cat_dict)
#NOTE:  the best batch size for image generation is not necessarly the best for the classifier.
        self.dataloader_eval = custom_dataloader(dataset=self.dataset_eval, batch_size=batch_size_data_loader, shuffle=False)

    def _eval_batch_gen(self, eval_batch_size):
        # the batch size for the evaluation must be a multiple of the batch size for the dataloader

        if eval_batch_size % self.batch_size_data_loader != 0:
            raise ValueError(f"Eval batch size {eval_batch_size} is not a multiple of the dataloader batch size {self.batch_size_data_loader}.")
        

        loader_batches_per_eval_batch= eval_batch_size // self.batch_size_data_loader
        total_eval_batches= len(self.dataloader_eval)// loader_batches_per_eval_batch

        loader_iter= iter(self.dataloader_eval)
        current_eval_batch= 0
        while current_eval_batch < total_eval_batches:
            # Get the batches from the dataloader
            dataloader_batches = []
            for _ in range(loader_batches_per_eval_batch):
                try:
                    dataloader_batches.append(next(loader_iter)[0])
                except StopIteration:
                    break
            

            # Concatenate the batches
            current_eval_batch += 1
            images_eval_batch= torch.cat(dataloader_batches, dim=0)
            yield images_eval_batch



    def _forward_model_on_images(self, images_batch):
        # Here we select the logits treshold to one hot
        outputs = self.Stage_obj.predict(images_batch)

        print(f"Hardcoding the threshold to 0.5, this should be changed in the future. (File: {__file__}, Line: {inspect.currentframe().f_lineno})")
        #return binary_classification_logits, binary_classification_labels
        # Get the predicted class with the highest score
        _, predicted = torch.max(outputs.data, 1)
        # Convert to one-hot encoding
        return torch.nn.functional.one_hot(predicted, num_classes=2).squeeze(1).cpu().numpy()
    
    def _sys_dict_to_cat_dict(self, system_dict):
        # Convert the system dictionary to a catalog dictionary
        # This is a placeholder, actual implementation will depend on the catalog structure
        return {"SL_systems": [system_dict]}
    


    def _get_suggested_new_bounds(self, new_grid_to_compute, previous_grid=None, criterion_values=None):
        raise(NotImplementedError("This function is not implemented yet."))
    
    def _get_the_bools(self, grid, criterion_values):
        
        cat_dict=self._grid_of_pos_sub_to_cat_dict(self, grid, criterion_values)
        # Update the catalog of the dataset
        print("Hardcoding the batch size for the dataloader to 10, this should be changed in the future.")
        self._update_cat_data_set_load(self, cat_dict, batch_size_data_loader=10)
        # Make the generator for the eval batches
        gen= self._eval_batch_gen(eval_batch_size=20)
        
        one_hot_results=[]
        for images_batch in gen:
            one_hot_results.append(self._forward_model_on_images(images_batch=images_batch))
        return one_hot_results
    

    def _get_criterion_value_on_grid(self, eval_grid, low_bound, high_bound):

        """
            Right now handling only same pace bisecting, but need to improve.
            The suggested bounds for every position are not the same. 
            In case the suggested bounds are too small, we need to scale up to the up to the input bounds.
        """
        criterion_low_bound= self.criterion_low_bound
        # do we want to go up until detects or not? I think we set the max of the training. 
        criterion_high_bound= self.criterion_high_bound
        criterion_interval= criterion_high_bound- criterion_low_bound
        low_bools= self._get_the_bools(eval_grid, criterion_low_bound)
        high_bools= self._get_the_bools(eval_grid, criterion_high_bound)

        grid_mask= low_bools != high_bools
        next_to_be_computed_criterion_values= criterion_low_bound + (criterion_high_bound- criterion_low_bound)/2

        while self.sub_detection_criterion_threshold < criterion_interval:
            # 
            mid_bools= self._get_the_bools(eval_grid, criterion_low_bound + (criterion_high_bound- criterion_low_bound)/2)
            
            # high for 1 0 0 low for 1 1 0
            # which means if mid bool = 0, high_bound-(high_bound-just_computed_criterion_values)/2
            # if mid bool = 1, low_bound+(high_bound-just_computed_criterion_values)/2
            next_to_be_computed_criterion_values= 
            # update the low and high bounds
            low_bound=
            high_bound=

            criterion_interval= high_bound- low_bound

        # Now take the mid point

        criterion_threshold= low_bound + (high_bound- low_bound)/2

        return criterion_threshold


    def compute_sensitivity_map(self, grid_refinement_iterations=None, criterion_error_threshold=None):
        # Load no substructure no noise image, (which size of images?)
     

        self._visualize_clean_and_dirty_image()
        Current_grid= None


        # Make the first grid to evaluate
        first_grid=self._first_grid_to_eval(self.eval_mode, None)
        values_on_grid= self._get_criterion_value_on_grid(first_grid, self.criterion_low_bound, self.criterion_high_bound)

        if grid_refinement_iterations in (None, 0):
            # If no grid refinement iterations, just return the first grid
            self._plot_sensitivity_map(values_on_grid, first_grid)
            return values_on_grid, first_grid
        else:
            raise (NotImplementedError("Grid refinement is not implemented yet."))
            for i in range(grid_refinement_iterations):
                new_grid_to_compute, cumulative_grid, calculated_values= self._new_grid_from_old_grid(Current_grid, values_on_grid, self.eval_mode)
                
    #NOTE:  here, if we do not get alternate values, we need to scale up to the input bounds.
                low_bounds_suggested, high_bounds_suggested = self._get_suggested_new_bounds(new_grid_to_compute, previous_grid= Current_grid, criterion_values= values_on_grid)

                criterion_threshold= self._get_criterion_value_on_grid(new_grid_to_compute, low_bounds_suggested, high_bounds_suggested)

        
            # we now have a comoputed values and the current complete grid.
            # we store that, and eventually interpolate on a regular grid.
            self.final_computed_values_criterion= criterion_threshold
            self.final_computed_grid= cumulative_grid
            # Eventuall








                    



            
