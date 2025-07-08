import json
import os
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torch.fft

import pandas as pd
from astropy.io import fits
import h5py

import math 

from .make_systems_dicts import make_systems_dicts
from config import CATALOGS_DIR, PSFS_DIR
from .make_systems_dicts import mode_mapping
from shared_utils import _plot2D_on_grid, recursive_to_tensor, _rad_to_arcsec, _grid_lens
from lensing_system import LensingSystem






class CatalogManager:
    """
    The core functionality of this class is to manage a catalog of strong lensing systems.
    The features include:
        - Make catalogs, given a single system sampler.
        - Load and save catalogs to JSON files.
        - Generate ready-to-training images from the json catalog.
        - Visualize statistics of the catalog and visulaizations of the systems.
    
    """
    def __init__(self, catalog_name_input=None, catalog_name=None, catalog_description="No description provided.", sampling_mode=None, low_lim_einst_angle_arcsec=None, num_samples=None, lens_grid=None, oversampling_factor=None):
        self.oversampling_factor=oversampling_factor
        if catalog_name_input is not None:
            # See if catalog exists in the CATLOG_DIR
            if not catalog_name_input.endswith(".json"):
                catalog_name_input += ".json"

            self.json_path = os.path.join(CATALOGS_DIR, catalog_name_input)

            if not os.path.exists(self.json_path):
                raise FileNotFoundError(f"Catalog file not found at {self.json_path}.")
            # if the input catalog name does not end with .json, add it
            
            with open(self.json_path, 'r') as f:
                self.catalog = json.load(f)

        elif catalog_name is not None and sampling_mode is not None and num_samples is not None and low_lim_einst_angle_arcsec is not None:
            self.sampling_mode = sampling_mode
            self.initialize_catalog(catalog_name, catalog_description)
            self.sample_systems(sampling_mode, num_samples, low_lim_einst_angle_arcsec, lens_grid=lens_grid)
            self.save_json()
        else:
            raise ValueError("Invalid input. Either provide a json path or catalog_name, sampling_mode, num_samples, and low_lim_einst_angle.")


     
    def get_availabe_modes(self):
        print(f"The available modes are: {json.dumps(mode_mapping, indent=4)}")

    
    def initialize_catalog(self, catalog_name="default_catalog_name", description="No description provided."):
        if not catalog_name.endswith(".json"):
            catalog_name += ".json"

        # Create the directory if it doesn't exist
        os.makedirs(CATALOGS_DIR, exist_ok=True)
        
        self.json_path = os.path.join(CATALOGS_DIR, catalog_name)

        if os.path.exists(self.json_path):
                #for security, I require to delaete the file manually
                raise FileExistsError(f"File already exists at {self.json_path}. Please delete it manually.")


        self.catalog = {
            "Description": description,
            "Used_mode": mode_mapping[self.sampling_mode],
            "Numb_precomp_images": False,
            "SL_systems": []
        }

    def sample_systems(self, mode, num_systems, low_einstein_angle_arcsec, lens_grid=None):
        #adjust the oversampling factor for performance
        system_dicts= make_systems_dicts(num_systems=num_systems, mode=mode, low_einstein_angle_arcsec=0.7, oversampling_factor=self.oversampling_factor, lens_grid=lens_grid)
        self.catalog["SL_systems"] = system_dicts

    def save_json(self):
    # Remove the file if it already exists.
        with open(self.json_path, 'w') as f:
            json.dump(self.catalog, f, indent=4)

    def len(self):
        return len(self.catalog["SL_systems"])
    
    def show_catalog_statistics(self):
        num_systems = len(self.catalog["SL_systems"])
        print(f"Number of systems: {num_systems}")
        print(f"Description: {self.catalog['Description']}")


    def add_precomputed_images_properties(self, number_images, noise_mode):
        #if the key is not there, make it
        if "Precomputed_images" not in self.catalog:
            self.catalog["Precomputed_images"] = []


        self.catalog["Precomputed_images"].append({
            "Number_images": number_images,
            "Noise_mode": noise_mode
        })
        self.save_json()

    def make_hdf5_convolved_psf_images(self, psf_kernel_fits_name, white_noise_stddev=0.0, psf_kernel_size_arcsec=14.0):
        #make a folder in the catalogs directory with the catalog_name + "_images"
        precomputed_images_folder = self.json_path.replace(".json", "_images")
        os.makedirs(precomputed_images_folder, exist_ok=True)
        
        #make the path to the psf kernel fits file
        psf_kernel_fits_path = os.path.join(PSFS_DIR, psf_kernel_fits_name)
        if not os.path.exists(psf_kernel_fits_path):
            raise FileNotFoundError(f"PSF kernel fits file not found at {psf_kernel_fits_path}.")
        
        #make the hdf5 path
        hdf5_file_name = f"{psf_kernel_fits_name.replace('.fits', '')}_images.h5"
        hdf5_path = os.path.join(precomputed_images_folder, hdf5_file_name)


        # Open the FITS file and load the image data
        with fits.open(psf_kernel_fits_path) as hdul:
            data = hdul[0].data
            # Convert the data to the native byte order if necessary
            data = data.byteswap().newbyteorder()

        # Convert the NumPy array to a torch tensor (as a float tensor)
        psf_tensor = torch.from_numpy(data).float()
        print(psf_tensor.shape)
        # check that the third and fourth dimensions are the same and multiple of 4
        if psf_tensor.shape[2] != psf_tensor.shape[3] or psf_tensor.shape[2] % 4 != 0:
            raise ValueError("The PSF kernel must have the same third and fourth dimensions, and both must be multiples of 4.")
        
        # Get the pixel size of the image as half of the psf tensor size
        image_pixel_size = psf_tensor.shape[2] // 2
        image_arcsec_size = psf_kernel_size_arcsec/2.0
        lens_grid=_grid_lens(grid_size_arcsec=image_arcsec_size, image_res=image_pixel_size, device="cpu")
        H, W= image_pixel_size, image_pixel_size
        padding = (W//2, W//2, H//2, H//2)

        #for every image, compute the it and store it in the hdf5 file
        #if the file already exists, delete it
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)
        with h5py.File(hdf5_path, 'w') as f:
            dset = f.create_dataset("images", shape=(len(self.catalog["SL_systems"]), 1, H, W), dtype='float32')
            # Loop over systems and write each image to the dataset.
            for idx, system in enumerate(self.catalog["SL_systems"]):
                one_image_dict=recursive_to_tensor(system, "cpu")
                lensing_system = LensingSystem(one_image_dict, "cpu")
                with torch.no_grad():
                    image= lensing_system(lens_grid)
                
                # Convolve the image with the PSF kernel
                padded_image = F.pad(image, padding, mode='constant', value=0)
                psf_shifted = torch.fft.ifftshift(psf_tensor, dim=(-2, -1))
                # Compute the FFTs along the spatial dimensions (-2, -1)
                image_fft = torch.fft.fftn(padded_image, dim=(-2, -1))
                psf_fft   = torch.fft.fftn(psf_shifted, dim=(-2, -1))
                # Perform the convolution in the Fourier domain using the convolution theorem
                conv_fft = image_fft * psf_fft

                # Add uncorrelated noise in the Fourier domain.
                # We generate complex noise with independent real and imaginary parts.
                noise_std = 0  # standard deviation for the noise (adjust as needed)
                noise_real = torch.randn_like(conv_fft.real) * noise_std
                noise_imag = torch.randn_like(conv_fft.imag) * noise_std
                noise_fourier = torch.complex(noise_real, noise_imag)

                # Add the Fourier-domain noise to the convolution result
                conv_fft_noisy = conv_fft + noise_fourier

                # Apply the inverse FFT to get back to the spatial domain.
                # Take the real part as the final convolved image.
                convolved_image_noisy = torch.fft.ifftn(conv_fft_noisy, dim=(-2, -1)).real

                # Crop the convolved image to the original image size.
                # Since the padded image has shape [1, 1, 2H, 2W],
                # we extract the central region corresponding to [H, W].
                start_h = H // 2
                start_w = W // 2
                cropped_image = convolved_image_noisy[:, :, start_h:start_h+H, start_w:start_w+W]


                # Move the tensor to CPU, detach, and convert to a NumPy array.
                image_np = cropped_image.detach().cpu().numpy().astype('float32')
                dset[idx] = image_np
                print(f"Saved image {idx+1}/{len(self.catalog['SL_systems'])}")
                plt.imshow(cropped_image[0, 0].detach().cpu().numpy())
                plt.show()





        


#=============================================VISUALIZATIONS FROM CHATGPT==========================================
    
   

    def visualize_sample(self, lens_grid, index):
        device = "cpu"
        lens_grid_arcsec = _rad_to_arcsec(lens_grid)
        lensing_system_dict_no_tensor = self.catalog["SL_systems"][index]

        lensing_system_dict = recursive_to_tensor(lensing_system_dict_no_tensor, device)
        lensing_system = LensingSystem(lensing_system_dict, device)
        
        # Create a panel with 3 rows x 2 columns (6 subplots, but we use only 5)
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), facecolor='#1f1f1f')
        axs = axs.flatten()  # Flatten to easily index subplots

        # Plot 1: Lensed Image
        image_tensor = lensing_system(lens_grid)
        _plot2D_on_grid(image_tensor, lens_grid_arcsec, ax=axs[0])
        axs[0].set_title("Lensed Image")

        # Plot 2: Convergence
        convergence_tensor = lensing_system.lens_model.compute_convergence(lens_grid)
        _plot2D_on_grid(convergence_tensor, lens_grid_arcsec, ax=axs[1])
        axs[1].set_title("Convergence")

        # Plot 3: Magnification (with substructure positions if available)
        magnification_tensor = lensing_system.lens_model.compute_magnification(lens_grid)
        magnification_tensor[magnification_tensor > 10] = 10
        magnification_tensor[magnification_tensor < -10] = -10

        # Collect substructure positions
        sub_positions = []
        for mass_comp in lensing_system_dict["lens_model"]["mass_components"]:
            if mass_comp["is_substructure"]:
                sub_pos = mass_comp["params"]["pos"]
                sub_pos_arcsec = _rad_to_arcsec(sub_pos)
                sub_pos_arcsec = sub_pos_arcsec.cpu().numpy()
                sub_positions.append(sub_pos_arcsec)
        
        _plot2D_on_grid(magnification_tensor, lens_grid_arcsec, ax=axs[2])
        if sub_positions:
            axs[2].scatter([pos[0] for pos in sub_positions],
                        [pos[1] for pos in sub_positions],
                        color='red')
        axs[2].set_title("Magnification")

        # Plot 4: Critical Curve
        critical_curve, critical_grid = lensing_system.lens_model.compute_critical_curve(lens_grid=lens_grid)
        critical_grid_arcsec = _rad_to_arcsec(critical_grid)
        _plot2D_on_grid(critical_curve, critical_grid_arcsec, ax=axs[3])
        axs[3].set_title("Critical Curve")

        # Plot 5: Caustic Curve (with source position)
        caustic_curve, caustic_grid, _, _, _ = lensing_system.lens_model.compute_caustics(lens_grid=lens_grid)
        caustic_grid_arcsec = _rad_to_arcsec(caustic_grid)
        _plot2D_on_grid(caustic_curve, caustic_grid_arcsec, ax=axs[4])
        random_pos_source = lensing_system_dict["source_model"]["params"]["position_rad"]
        random_pos_source = _rad_to_arcsec(random_pos_source)
        axs[4].scatter(random_pos_source[0], random_pos_source[1], color='red')
        axs[4].set_title("Caustic Curve")

        # Hide all the unused subplots.
        for j in range(5, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        for ax in axs.flatten():
            ax.set_aspect('equal', adjustable='box')
            ax.tick_params(colors='gray')  # Make tick labels white
            for spine in ax.spines.values():
                spine.set_edgecolor('none')
            ax.xaxis.label.set_color('gray')
            ax.yaxis.label.set_color('gray')
            ax.title.set_color('gray')
            ax.title.set_fontsize(20)
            #set also the ticks of the colorbar, if there, to gray
         
        #plt.tight_layout()
        def on_resize(event):
            fig.tight_layout()
        fig.canvas.mpl_connect('resize_event', on_resize)
        plt.show()

        from rich.console import Console, Group
        from rich.table import Table

        def create_nested_table(d, max_depth=5, current_depth=0):
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Key", style="dim", no_wrap=True)
            table.add_column("Value", overflow="fold")
            
            for key, value in d.items():
                if isinstance(value, dict) and current_depth < max_depth:
                    sub_table = create_nested_table(value, max_depth, current_depth + 1)
                    table.add_row(key, sub_table)
                elif isinstance(value, list):
                    # Process list elements individually
                    rendered_items = []
                    for i, item in enumerate(value):
                        if isinstance(item, dict) and current_depth < max_depth:
                            rendered_items.append(create_nested_table(item, max_depth, current_depth + 1))
                        else:
                            rendered_items.append(str(item))
                    # Group the list items vertically using Group
                    group = Group(*rendered_items)
                    table.add_row(key, group)
                else:
                    table.add_row(key, str(value))
                    
            return table
        console = Console()
        console.print(create_nested_table(lensing_system_dict_no_tensor))
    
    def get_system_dict(self, idx):
        return  self.catalog["SL_systems"][idx]

    
    @staticmethod
    def _collect_numeric_fields(item, path=""):
        """
        Recursively traverses a dictionary or list item and collects numeric values.
        - Handles both lists AND tuples as potential coordinate pairs
        - Properly separates scalar values from coordinate pairs
        
        Args:
            item: The data structure to process
            path: Current path in the hierarchy
            
        Returns:
            tuple: (scalar_data, pair_data)
                - scalar_data: Maps paths to lists of scalar numeric values
                - pair_data: Maps paths to lists of coordinate pairs (either lists or tuples)
        """
        scalar_data = {}
        pair_data = {}
        
        def add_to_dict(d, key, value):
            """Helper to add a value to a dictionary of lists"""
            if key not in d:
                d[key] = []
            d[key].append(value)
        
        def process(obj, current_path):
            # Case 1: Dictionary handling
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_path = f"{current_path}.{k}" if current_path else k
                    process(v, new_path)
            
            # Case 2: Tuple or List handling
            elif isinstance(obj, (list, tuple)):
                # Check if this is a coordinate pair: length 2 and all numeric
                is_coord_pair = (len(obj) == 2 and 
                                all(isinstance(x, (int, float)) and not isinstance(x, bool) 
                                    for x in obj))
                
                if is_coord_pair:
                    add_to_dict(pair_data, current_path, obj)
                else:
                    # Not a coordinate pair, process individual elements
                    for idx, elem in enumerate(obj):
                        elem_path = f"{current_path}[{idx}]"
                        if isinstance(elem, (int, float)) and not isinstance(elem, bool):
                            add_to_dict(scalar_data, elem_path, elem)
                        else:
                            process(elem, elem_path)
            
            # Case 3: Scalar value handling
            elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
                add_to_dict(scalar_data, current_path, obj)
        
        # Start processing
        process(item, path)
        
        return scalar_data, pair_data

    def visualize_statistics(self):
        """
        Provides a flexible statistical overview of the catalog by scanning for all numerical entries.
        - Scalar numerical data are displayed as histograms.
        - Data that appear as pairs (lists or tuples of two numbers) are shown as 2D histograms.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        
        systems = self.catalog.get("SL_systems", [])
        combined_scalars = {}
        combined_pairs = {}
        
        # Process each system
        for system in systems:
            scalars, pairs = self._collect_numeric_fields(system, "")
            
            # Combine results
            for key, values in scalars.items():
                if key not in combined_scalars:
                    combined_scalars[key] = []
                combined_scalars[key].extend(values)
            
            for key, pair_list in pairs.items():
                if key not in combined_pairs:
                    combined_pairs[key] = []
                combined_pairs[key].extend(pair_list)
        
        print("=== Flexible Catalog Analysis ===")
        print(f"Aggregated {len(combined_scalars)} scalar numeric keys and {len(combined_pairs)} pair keys across {len(systems)} systems.")
        
        # Plot scalar data as histograms
        scalar_keys = list(combined_scalars.keys())
        if scalar_keys:
            n_plots = len(scalar_keys)
            cols = 4
            rows = math.ceil(n_plots / cols)
            fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            
            # Handle both single-plot and multi-plot cases
            if rows * cols == 1:
                axs = np.array([axs])
            axs = axs.flatten()
            
            for i, key in enumerate(scalar_keys):
                if i < len(axs):
                    ax = axs[i]
                    ax.hist(combined_scalars[key], bins=20)
                    ax.set_title(f"Histogram for {key}")
                    ax.set_xlabel(key)
                    ax.set_ylabel("Frequency")
            
            # Hide unused subplots
            for j in range(min(i+1, len(axs)), len(axs)):
                fig.delaxes(axs[j])
            
            plt.tight_layout()
            plt.show()

        # Plot pair data as 2D histograms
        pair_keys = list(combined_pairs.keys())
        if pair_keys:
            n_plots = len(pair_keys)
            cols = min(4, max(1, n_plots))
            rows = math.ceil(n_plots / cols)
            fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            
            # Handle both single-plot and multi-plot cases
            if rows * cols == 1:
                axs = np.array([axs])
            axs = axs.flatten()
            
            for i, key in enumerate(pair_keys):
                if i < len(axs):
                    ax = axs[i]
                    pairs_list = combined_pairs[key]
                    
                    # Convert tuples to lists if needed - this ensures we can extract x,y components
                    processed_pairs = []
                    for pair in pairs_list:
                        if isinstance(pair, (list, tuple)) and len(pair) == 2:
                            processed_pairs.append([float(pair[0]), float(pair[1])])
                    
                    if processed_pairs:
                        # Extract x and y coordinates
                        x_vals = [pair[0] for pair in processed_pairs]
                        y_vals = [pair[1] for pair in processed_pairs]
                        
                        # Create 2D histogram
                        try:
                            h = ax.hist2d(x_vals, y_vals, bins=20)
                            ax.set_title(f"2D Histogram for {key}")
                            ax.set_xlabel(f"{key} (x-component)")
                            ax.set_ylabel(f"{key} (y-component)")
                            plt.colorbar(h[3], ax=ax)
                        except Exception as e:
                            ax.text(0.5, 0.5, f"Error: {str(e)}", 
                                    horizontalalignment='center',
                                    verticalalignment='center',
                                    transform=ax.transAxes)
                    else:
                        ax.text(0.5, 0.5, "No valid pairs to plot", 
                                horizontalalignment='center',
                                verticalalignment='center',
                                transform=ax.transAxes)
            
            # Hide unused subplots
            for j in range(min(i+1, len(axs)), len(axs)):
                fig.delaxes(axs[j])
            
            plt.tight_layout()
            plt.show()

    # def randomly_pos_sources_in_caustics(self, lens_grid, max_distance_arcsec=0.0, max_dist_source_std=False, verbose=False, final_shuffle_arcsec=0.0):
    #     """
    #     For every SL system in the catalog, repositions the source by computing
    #     the caustics of its lens model and sampling a random position inside one
    #     of the caustic clusters. If no valid source position is found, the original
    #     position is kept.
        
    #     Parameters:
    #         max_distance (float or None): Maximum allowed distance from a caustic boundary 
    #                                     when sampling a random source position. If None,
    #                                     defaults to 0.
    #     """    
    #     print("Repositioning sources inside caustics...")
    #     # Loop over each SL system in the catalog.
    #     for system in self.catalog["SL_systems"]:
    #         if max_dist_source_std:
    #             z_source=system["source"]["params"]["z"]
    #             D_s = cosmology.angular_diameter_distance(0, z_source) * 1000  # kpc
    #             max_distance_arcsec = util._arcsec_to_rad(system["source"]["params"]["s_kpc"]/D_s) # kpc to rad
    #         # Get lens parameters and construct a LensModel.
    #         lens_params = system["lens"]
    #         lens = LensModel(config_dict=lens_params)
                                    
    #         # Sample a new source position using the clusters.
    #         new_source_position = lens.random_pos_inside_caustic(lens_grid, max_distance_arcsec=max_distance_arcsec, verbose=verbose)
    #         #print(util._rad_to_arcsec(np.array(new_source_position)))
    #         if final_shuffle_arcsec > 0:
    #             new_source_position += np.random.uniform(-final_shuffle_arcsec, final_shuffle_arcsec, 2)


    #         if new_source_position is not None:
    #             # Update the source position in the catalog.
    #             system["source"]["params"]["position"] = new_source_position
    #         else:
    #             print(f"Warning: No valid source position found for system {system.get('system_index', 'unknown')}. Original position retained.")

               
    # def make_training_images(self, uncropped_grid , folder_name=None):
    #     print("Making training images...")
    #     # Ensure the folder exists; if it does, empty it once before processing.
    #     if folder_name is None:
    #         folder_name = self.json_path.replace(".json", "_images")

    #     if not os.path.exists(folder_name):
    #         os.makedirs(folder_name)
    #     else:
    #         for file in os.listdir(folder_name):
    #             os.remove(os.path.join(folder_name, file))
        
    #     for system in self.catalog["SL_systems"]:
    #         # Get lens and source parameters.
    #         lens_params = system["lens"]
    #         source_params = system["source"]
            
    #         # Construct LensModel and SourceModel objects.
    #         lens = LensModel(config_dict=lens_params)
    #         source = make_source_from_dict(source_params)
            
    #         # Generate the lensed image.
    #         image_tensor= lens.compute_image(source, uncropped_grid)
            
    #         # Convert tensor to a NumPy array.
    #         image_np = image_tensor.detach().cpu().numpy()
            
    #         # If the image is one-channel (shape: [1, H, W]), remove the singleton channel dimension.
    #         if image_np.ndim == 3 and image_np.shape[0] == 1:
    #             image_np = image_np.squeeze(0)
            
    #         # Scale pixel values from [0, 1] to [0, 255] and convert to uint8.
    #         image_np = (image_np * 255).astype(np.uint8)
            
    #         # Create a grayscale image using PIL with mode 'L'.
    #         img = Image.fromarray(image_np, mode='L')
            
    #         # Save the image as a PNG file (lossless format).
    #         image_path = os.path.join(folder_name, f"system_{system.get('system_index', 'unknown')}.png")
    #         img.save(image_path)
    #     self.catalog["Has_precomputed_images"] = True
    #     self.catalog["Image_folder_path"] = folder_name
    #     self.save_json()
    #     return 0

    # def make_training_images_hdf5(self, uncropped_grid, hdf5_path=None):
    #     """
    #     Generate training images for all systems and save them directly to an HDF5 file.
        
    #     Parameters:
    #         uncropped_grid: The grid to be used when generating images.
    #         hdf5_path (str, optional): Path to save the HDF5 file.
    #                                 If None, the path is generated from self.json_path.
        
    #     Returns:
    #         str: The path to the saved HDF5 file.
    #     """
    #     print("Generating training images and saving to an HDF5 file...")

    #     # Determine the HDF5 file path using self.json_path if not provided.
    #     if hdf5_path is None:
    #         folder_name = self.json_path.replace(".json", "_images")
    #         if not os.path.exists(folder_name):
    #             os.makedirs(folder_name)
    #         hdf5_path = os.path.join(folder_name, "images_dataset.h5")

    #     # Determine the number of images.
    #     num_images = len(self.catalog["SL_systems"])

    #     # Generate one sample image to determine the shape.
    #     sample_system = self.catalog["SL_systems"][0]
    #     lens_params = sample_system["lens"]
    #     source_params = sample_system["source"]
    #     lens = LensModel(config_dict=lens_params)
    #     source = make_source_from_dict(source_params)
    #     with torch.no_grad():
    #         sample_image = lens.compute_image(source, uncropped_grid)
    #     # Ensure sample_image has a channel dimension.
    #     if sample_image.ndim == 2:
    #         sample_image = sample_image.unsqueeze(0)
    #     # (C, H, W) shape.
    #     C, H, W = sample_image.shape

    #     # Create an HDF5 file and preallocate a dataset.
    #     with h5py.File(hdf5_path, 'w') as f:
    #         dset = f.create_dataset("images", shape=(num_images, C, H, W), dtype='float32')
    #         # Loop over systems and write each image to the dataset.
    #         for idx, system in enumerate(self.catalog["SL_systems"]):
    #             lens_params = system["lens"]
    #             source_params = system["source"]
    #             lens = LensModel(config_dict=lens_params)
    #             source = make_source_from_dict(source_params)
                
    #             with torch.no_grad():
    #                 image_tensor = lens.compute_image(source, uncropped_grid)
    #             if image_tensor.ndim == 2:
    #                 image_tensor = image_tensor.unsqueeze(0)
                
    #             # Move the tensor to CPU, detach, and convert to a NumPy array.
    #             image_np = image_tensor.detach().cpu().numpy().astype('float32')
    #             dset[idx] = image_np
    #             print(f"Saved image {idx+1}/{num_images}")

    #     # Update catalog information.
    #     self.catalog["HDF5_path"] = hdf5_path
    #     self.catalog["Has_precomputed_images_HDF5"] = True
    #     self.catalog["Image_folder_path"] = os.path.dirname(hdf5_path)
    #     self.save_json()

    #     return hdf5_path




if __name__ == "__main__":
    try:
        # Example usage
        catalog_manager = CatalogManager(catalog_name="my_catalog.json", sampling_mode="example_mode", num_samples=10, low_lim_einst_angle=0.7)
    except Exception as e:
        print("Error encountered:", e)
