import torch
from astropy.cosmology import Planck18 as cosmo
import numpy as np
from lensing_system import LensModel

# Assuming these are defined in your project:
from shared_utils import units, util, recursive_to_tensor, _arcsec_to_rad, _rad_to_arcsec
from .distributions import uniform_prior, sample_redshift_comoving_volume, log_uniform_prior, random_pos_in_circle, resample_theta
from tqdm import tqdm
# from distributions import uniform_prior, sample_redshift_comoving_volume
from .archived_modes import mode_mapping_archived


mode_mapping_here = {
    "example_mode": {
            "redshifts_and_vel_disp": "min_einstein_angle_vel_disp_prior_100_400",
            "main_lens_mode": "Single_SIS",
            "substructure_mode": "No_substructure",
            "source_mode": "Gaussian_blob_no_rotation",
            "source_positioning": "source_random_pos_in_caustics", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
    "conor_train_gauss_source_10e11": {
        "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
        "main_lens_mode": "Single_PEMD_with_no_shear_conor_like_train",
        "substructure_mode": "1_or_0_NFW_6x6_arcsec_mass_conc_prior_conor_like_min_10e11",
        "source_mode": "Gaussian_blob_no_rotation",
        "source_positioning": "inside_ein_angle_in_source_plane_0dot3", #or inside_ein_angle_in_source_plane
        "Precompute_quantities": True,

    },
    
    "conor_train_gauss_source_10e11_resample_theta": {
        "redshifts_and_vel_disp": "resample_theta",
        "main_lens_mode": "Single_PEMD_with_no_shear_conor_like_train",
        "substructure_mode": "1_or_0_NFW_6x6_arcsec_mass_conc_prior_conor_like_min_10e11",
        "source_mode": "Gaussian_blob_no_rotation",
        "source_positioning": "inside_ein_angle_in_source_plane_0dot3", #or inside_ein_angle_in_source_plane
        "Precompute_quantities": True,

    },
    
    "conor_train_gauss_source_10e10_resample_theta": {
        "redshifts_and_vel_disp": "resample_theta",
        "main_lens_mode": "Single_PEMD_with_no_shear_conor_like_train",
        "substructure_mode": "1_or_0_NFW_6x6_arcsec_mass_conc_prior_conor_like_min_10e10",
        "source_mode": "Gaussian_blob_no_rotation",
        "source_positioning": "inside_ein_angle_in_source_plane_0dot3", #or inside_ein_angle_in_source_plane
        "Precompute_quantities": True,

    },
    
    "conor_train_gauss_source_10e9_resample_theta": {
        "redshifts_and_vel_disp": "resample_theta",
        "main_lens_mode": "Single_PEMD_with_no_shear_conor_like_train",
        "substructure_mode": "1_or_0_NFW_6x6_arcsec_mass_conc_prior_conor_like_min_10e9",
        "source_mode": "Gaussian_blob_no_rotation",
        "source_positioning": "inside_ein_angle_in_source_plane_0dot3", #or inside_ein_angle_in_source_plane
        "Precompute_quantities": True,

    },
    
    "conor_train_gauss_source_10e8_6_resample_theta": {
        "redshifts_and_vel_disp": "resample_theta",
        "main_lens_mode": "Single_PEMD_with_no_shear_conor_like_train",
        "substructure_mode": "1_or_0_NFW_6x6_arcsec_mass_conc_prior_conor_like_min_10e8_6",
        "source_mode": "Gaussian_blob_no_rotation",
        "source_positioning": "inside_ein_angle_in_source_plane_0dot3", #or inside_ein_angle_in_source_plane
        "Precompute_quantities": True,

    }
}


mode_mapping = {**mode_mapping_here, **mode_mapping_archived}

def make_systems_dicts(num_systems=10, mode="example_mode", low_einstein_angle_arcsec=0.7, oversampling_factor=20, lens_grid=None, mode_mapping=mode_mapping):  
    """
    Create a list of lensing system configurations.
    """
    mode_mapping = mode_mapping

    # Retrieve the configuration for the given mode.    
    try:
        mode_config = mode_mapping[mode]
    except KeyError:
        raise KeyError(f"Unknown mode: {mode}, the available modes are: {list(mode_mapping.keys())}, to add a new mode, please update the mode_mapping dictionary in make_systems_dicts.py")
        
    #print("Selected systems generation: ", mode_config)

    # Convert the lower Einstein angle limit from arcsec to radians.
    low_einstein_angle = units._arcsec_to_rad(low_einstein_angle_arcsec)

    # Get sampled values for redshifts, velocity dispersion, and distances.
    z_lens, z_source, vel_disp, D_l, D_s, D_ls, einstein_angle= get_zs_and_veldisp(
                                                    mode=mode_config["redshifts_and_vel_disp"],
                                                    min_einstein_angle=low_einstein_angle,
                                                    num_samples=num_systems,
                                                    oversampling_factor=oversampling_factor)

    lensing_system_list = []
    for i in tqdm(range(num_systems), desc="Generating lensing systems"):
        # Pass the sampled redshift and velocity dispersion to the dict-generating functions.
        main_lens_dict = get_main_lens_dict(mode_config["main_lens_mode"], z_lens[i], vel_disp[i])
        source_dict = get_source_dict(mode_config["source_mode"], z_source[i])
        
        if mode_config["Precompute_quantities"]:
            precomputed_quantities = {
                "D_l": D_l[i],
                "D_s": D_s[i],
                "D_ls": D_ls[i],
                "Theta_E": einstein_angle[i]
            }
        else:
            precomputed_quantities = {}

        substructure_list = get_substructure_dict(mode_config["substructure_mode"], z_lens[i], precomp_dict=precomputed_quantities)
        num_substructures = len(substructure_list)

        config_dict = {
            "system_index": i,
            "precomputed": precomputed_quantities,
            "lens_model": {
                "num_substructures": num_substructures,
                "mass_components": main_lens_dict + substructure_list
            },
            "source_model": source_dict
        }
        if mode_config["source_positioning"]=="source_random_pos_in_caustics":
            if lens_grid is None:
                raise ValueError("The lens grid must be provided for random position of the source.")
            # NOTE: Here the device is manually set to cpu. 
            config_dict_temp = recursive_to_tensor(config_dict, device="cpu")
            lens_model = LensModel(config_dict_temp["lens_model"], precomp_dict=config_dict_temp["precomputed"], device="cpu")
            config_dict["source_model"] = get_source_dict(mode_config["source_mode"], z_source[i], lens_model=lens_model, lens_grid=lens_grid)
        
        
        elif mode_config["source_positioning"]=="inside_ein_angle_in_source_plane_0dot3":
            if mode_config["Precompute_quantities"]==False:
                raise ValueError("This source positioning mode requires to have precomputed quantities, in particular theta E")
                
            einstein_angle_here=precomputed_quantities["Theta_E"]
            #the follwing outputs an array of positions
            random_position_in_einstein_circle=random_pos_in_circle(radius=einstein_angle_here*0.3, n_samples=1)
            config_dict["source_model"]["params"]["position_rad"]=random_position_in_einstein_circle[0].tolist()
            
        else:
            raise ValueError("The source_positioning mode selected is mispelled or not available")
        lensing_system_list.append(config_dict)
    
    return lensing_system_list


"""
============================================MAIN LENS DICT================================================
"""
def get_main_lens_dict(main_lens_mode, z, vel_disp):
    available_main_lens_modes = [
        "Single_SIS",
        "Single_SIS_fixed_vel_disp",
        "Single_NFW",
        "Single_PEMD",
        "Single_PEMD_experimental",
        "Single_PEMD_with_external_shear",
        "Single_PEMD_with_external_shear_conor_like_but_low_ellipticity",
        "Single_PEMD_with_external_no_shear_conor_like_but_low_ellipticity",
        "Single_PEMD_with_external_no_shear_conor_like_but_medium_ellipticity",
        "Single_PEMD_with_no_shear_conor_like_train"
        
    ]

    if main_lens_mode not in available_main_lens_modes:
        raise ValueError(f"Unknown main lens mode: {main_lens_mode}, Check the available modes in make_systems_dicts.py")
    
    if main_lens_mode == "Single_SIS":
        return [{
            "type": "SIS",
            "is_substructure": False,
            "params": {
                "pos": [0.0, 0.0],
                "redshift": z,
                "vel_disp": vel_disp
            }
        }]
    if main_lens_mode == "Single_PEMD_experimental":
        return [{
            "type": "PEMD",
            "is_substructure": False,
            "params": {
                "pos": [0.0, 0.0],
                "redshift": z,
                "vel_disp": vel_disp,
                "slope": uniform_prior(n_samples=1, min_value=0.9, max_value=1.1).tolist(),
                "orient": 0.0,
                "q": uniform_prior(n_samples=1, min_value=0.8, max_value=1.0).tolist()
            }
        }]
    if main_lens_mode == "Single_PEMD_with_external_shear_conor_like_but_low_ellipticity":
        return [{
                "type": "PEMD",
                "is_substructure": False,
                "params": {
                    "pos": [0.0, 0.0],
                    "redshift": z,
                    "vel_disp": vel_disp,
                    #slope normal distribution on 1 with 0.1 sigma
                    "slope": np.random.normal(1, 0.1),
                    "orient": uniform_prior(n_samples=1, min_value=0, max_value=np.pi).tolist(),
                    "q": uniform_prior(n_samples=1, min_value=0.8, max_value=1.0).tolist()
                }
            },
            {
                "type": "ExternalPotential",
                "is_substructure": False,
                "params": {
                    "shear_center": [0.0, 0.0],
                    "shear_strength": uniform_prior(n_samples=1, min_value=0.00, max_value=0.1).tolist(),
                    "shear_angle_arcsec": uniform_prior(n_samples=1, min_value=0, max_value=2*np.pi).tolist(),
                }
            }
        ]
    
    if main_lens_mode == "Single_PEMD_with_external_no_shear_conor_like_but_low_ellipticity":
        return [{
                "type": "PEMD",
                "is_substructure": False,
                "params": {
                    "pos": [0.0, 0.0],
                    "redshift": z,
                    "vel_disp": vel_disp,
                    #slope normal distribution on 1 with 0.1 sigma
                    "slope": np.random.normal(1, 0.1),
                    "orient": uniform_prior(n_samples=1, min_value=0, max_value=np.pi).tolist(),
                    "q": uniform_prior(n_samples=1, min_value=0.8, max_value=1.0).tolist()
                }
            }
        ]
    
    if main_lens_mode == "Single_PEMD_with_external_no_shear_conor_like_but_medium_ellipticity":
        return [{
                "type": "PEMD",
                "is_substructure": False,
                "params": {
                    "pos": [0.0, 0.0],
                    "redshift": z,
                    "vel_disp": vel_disp,
                    #slope normal distribution on 1 with 0.1 sigma
                    "slope": np.random.normal(1, 0.1),
                    "orient": uniform_prior(n_samples=1, min_value=0, max_value=np.pi).tolist(),
                    "q": uniform_prior(n_samples=1, min_value=0.6, max_value=1.0).tolist()
                }
            }
        ]
    
    if main_lens_mode == "Single_PEMD_with_no_shear_conor_like_train":
        return [{
                "type": "PEMD",
                "is_substructure": False,
                "params": {
                    "pos": [0.0, 0.0],
                    "redshift": z,
                    "vel_disp": vel_disp,
                    #slope normal distribution on 1 with 0.1 sigma
                    "slope": np.random.normal(1, 0.1),
                    "orient": uniform_prior(n_samples=1, min_value=0, max_value=np.pi).tolist(),
                    "q": uniform_prior(n_samples=1, min_value=0.2, max_value=1.0).tolist()
                }
            }
        ]
    

"""
============================================SUBSTRUCTURE DICT================================================
"""
def get_substructure_dict(substructure_mode, lens_z, precomp_dict=None):
    available_substructure_modes = [
        "Zero_or_1_NFW_no_mass_max_r_max_relation_in_2_einstein_radii_10e11",
        "Zero_or_1_NFW_no_mass_max_r_max_relation_in_2_einstein_radii_10e10",
        "Zero_or_1_NFW_no_mass_max_r_max_relation_in_2_einstein_radii_10e9",
        "Zero_or_1_NFW_no_mass_max_r_max_relation_in_2_einstein_radii_10e8",
        "Zero_or_1_NFW_no_mass_max_r_max_relation_in_2_einstein_radii_10e7",
        "No_substructure",
        "Conor_like_substructure",
        "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like",
        "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_min_10e11",
        "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_min_10e10",
        "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_min_10e9",
        "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_min_10e8",
        "Zero_or_1_NFW_in_grid_6x6_arcsec_mass_conc_prior_conor_and_more_min_10e11",
        "Zero_or_1_NFW_in_grid_6x6_arcsec_mass_conc_prior_conor_and_more_medium_min_10e11",
        "Zero_or_1_NFW_in_grid_6x6_arcsec_mass_conc_prior_conor_and_more_medium_less_min_10e11",
        "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_rigorous_min_10e11",
        "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_rigorous_min_10e10",
        "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_rigorous_min_10e9",
        "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_rigorous_min_10e8",
        "1_or_0_NFW_6x6_arcsec_mass_conc_prior_conor_like_min_10e11",
        "1_or_0_NFW_6x6_arcsec_mass_conc_prior_conor_like_min_10e10",
        "1_or_0_NFW_6x6_arcsec_mass_conc_prior_conor_like_min_10e8_6",
        "1_or_0_NFW_6x6_arcsec_mass_conc_prior_conor_like_min_10e9"


    ]
    """
        Mode descriptions:
        - Zero_or_1_NFW_no_mass_max_r_max_relation_in_2_einstein_radii:
            - Randomly select zero or one NFW substructure with a mass  1e10 M_sun and r_max 1 kpc.
            - The position is randomly selected within a disc of radius 2*Theta_E.
    """
    def get_r_max_from_M_max(M_max):
            const=2.16258
            A=0.344 #kpc
            B=1.607
            v_max=((1/const)*units.G/A*10**B*M_max)**(1/(B+2))#in km/s
            r_max=A*(v_max/10)**B #in kpc
            return r_max
        
        
    if substructure_mode not in available_substructure_modes:
        raise ValueError(f"Unknown substructure mode: {substructure_mode}, Check the available modes in make_systems_dicts.py")
        
        
        
    
    if substructure_mode == "Zero_or_1_NFW_no_mass_max_r_max_relation_in_2_einstein_radii_10e11":
        assert precomp_dict is not None, "precomp_dict must be provided for this mode, to get einstein angle estimate."
        # Check if the Einstein angle is available in the precomputed quantities.
        assert "Theta_E" in precomp_dict, "Einstein angle not found in precomp_dict."

        Theta_E = precomp_dict["Theta_E"]
        substructure_list = []
        if np.random.rand() < 0.5:
            # Get a random position within a disc of radius 2*Theta_E.
            r = np.sqrt(np.random.rand()) * 2 * Theta_E
            phi = np.random.rand() * 2 * np.pi
            r_max_center= get_r_max_from_M_max(1e11)
            substructure_list.append({
                "type": "NFW",
                "is_substructure": True,
                "params": {
                    "pos": [r * np.cos(phi), r * np.sin(phi)],
                    "mass_max": 1e11,
                    "r_max_kpc":r_max_center,
                    "redshift": lens_z,
                }
            })
        
        return substructure_list
    
        
    if substructure_mode == "Zero_or_1_NFW_no_mass_max_r_max_relation_in_2_einstein_radii_10e10":
        assert precomp_dict is not None, "precomp_dict must be provided for this mode, to get einstein angle estimate."
        # Check if the Einstein angle is available in the precomputed quantities.
        assert "Theta_E" in precomp_dict, "Einstein angle not found in precomp_dict."

        Theta_E = precomp_dict["Theta_E"]
        substructure_list = []
        if np.random.rand() < 0.5:
            # Get a random position within a disc of radius 2*Theta_E.
            r = np.sqrt(np.random.rand()) * 2 * Theta_E
            phi = np.random.rand() * 2 * np.pi
            r_max_low=get_r_max_from_M_max(1e10)
            r_max_high=get_r_max_from_M_max(1e11)
            substructure_list.append({
                "type": "NFW",
                "is_substructure": True,
                "params": {
                    "pos": [r * np.cos(phi), r * np.sin(phi)],
                    "mass_max": np.power(10, 10+np.random.rand()),
                    "r_max_kpc":uniform_prior(1, r_max_low, r_max_high).tolist()[0],
                    "redshift": lens_z,
                }
            })
        
        return substructure_list
    
    if substructure_mode == "Zero_or_1_NFW_no_mass_max_r_max_relation_in_2_einstein_radii_10e9":
        assert precomp_dict is not None, "precomp_dict must be provided for this mode, to get einstein angle estimate."
        # Check if the Einstein angle is available in the precomputed quantities.
        assert "Theta_E" in precomp_dict, "Einstein angle not found in precomp_dict."

        Theta_E = precomp_dict["Theta_E"]
        substructure_list = []
        if np.random.rand() < 0.5:
            # Get a random position within a disc of radius 2*Theta_E.
            r = np.sqrt(np.random.rand()) * 2 * Theta_E
            phi = np.random.rand() * 2 * np.pi
            r_max_low=get_r_max_from_M_max(1e9)
            r_max_high=get_r_max_from_M_max(1e11)
            substructure_list.append({
                "type": "NFW",
                "is_substructure": True,
                "params": {
                    "pos": [r * np.cos(phi), r * np.sin(phi)],
                    "mass_max": np.power(9, 10+np.random.rand()*2),
                    "r_max_kpc":uniform_prior(1, r_max_low, r_max_high).tolist()[0],
                    "redshift": lens_z,
                }
            })
        
        return substructure_list
    
    if substructure_mode == "Zero_or_1_NFW_no_mass_max_r_max_relation_in_2_einstein_radii_10e8":
        assert precomp_dict is not None, "precomp_dict must be provided for this mode, to get einstein angle estimate."
        # Check if the Einstein angle is available in the precomputed quantities.
        assert "Theta_E" in precomp_dict, "Einstein angle not found in precomp_dict."

        Theta_E = precomp_dict["Theta_E"]
        substructure_list = []
        if np.random.rand() < 0.5:
            # Get a random position within a disc of radius 2*Theta_E.
            r = np.sqrt(np.random.rand()) * 2 * Theta_E
            phi = np.random.rand() * 2 * np.pi
            r_max_low=get_r_max_from_M_max(1e8)
            r_max_high=get_r_max_from_M_max(1e11)
            substructure_list.append({
                "type": "NFW",
                "is_substructure": True,
                "params": {
                    "pos": [r * np.cos(phi), r * np.sin(phi)],
                    "mass_max": np.power(8, 10+np.random.rand()*3),
                    "r_max_kpc":uniform_prior(1, r_max_low, r_max_high).tolist()[0],
                    "redshift": lens_z,
                }
            })
        
        return substructure_list
    
    
    
    if substructure_mode == "Conor_like_substructure":
        
        
        assert precomp_dict is not None, "precomp_dict must be provided for this mode, to get einstein angle estimate."
        # Check if the Einstein angle is available in the precomputed quantities.
        assert "Theta_E" in precomp_dict, "Einstein angle not found in precomp_dict."

        Theta_E = precomp_dict["Theta_E"]
        substructure_list = []
        #with equal probability, add either 0 or 1-4 substructures
        if np.random.rand() < 0.5:
            return []
        else:
            num_substructures = np.random.randint(1, 5)
            for i in range(num_substructures):
                # Get a random position within a disc of radius 2*Theta_E.
                r = np.sqrt(np.random.rand()) * 2 * Theta_E
                phi = np.random.rand() * 2 * np.pi
                mass_max=log_uniform_prior(n_samples=1, min_value=10**8.6, max_value=10**11).tolist()[0]
                substructure_list.append({
                    "type": "NFW",
                    "is_substructure": True,
                    "params": {
                        "pos": [r * np.cos(phi), r * np.sin(phi)],
                        "mass_max": mass_max,
                        "r_max_kpc": get_r_max_from_M_max(mass_max),
                        "redshift": lens_z,
                    }
                })
            return substructure_list
        
        

    
    
        


    if substructure_mode == "1_or_0_NFW_6x6_arcsec_mass_conc_prior_conor_like_min_10e11":
        substructure_list=aux_func(precomp_dict, lens_z, low_lim_mass=1e11)

        return substructure_list
    
    if substructure_mode == "1_or_0_NFW_6x6_arcsec_mass_conc_prior_conor_like_min_10e10":
        substructure_list=aux_func(precomp_dict, lens_z, low_lim_mass=1e10)

        return substructure_list


    if substructure_mode == "1_or_0_NFW_6x6_arcsec_mass_conc_prior_conor_like_min_10e9":
        substructure_list=aux_func(precomp_dict, lens_z, low_lim_mass=1e9)

        return substructure_list
    
    if substructure_mode == "1_or_0_NFW_6x6_arcsec_mass_conc_prior_conor_like_min_10e8_6":
        substructure_list=aux_func(precomp_dict, lens_z, low_lim_mass=10**8.6)

        return substructure_list
    


    elif substructure_mode == "No_substructure":
        return []


    # How to perform the stuff:
    # const=torch.tensor(2.16258, device=self.device)
    # c_term=((1+const)/const)**2*(torch.log(1+const)-const/(1+const))
    # A=torch.tensor(0.344, device=self.device) #kpc
    # B=torch.tensor(1.607, device=self.device) 

    # v_max=((1/c_term)*units.G/A*10**B*self.mass_max)**(1/(B+2))#in km/s
    
    # if not r_max:
    #     r_max=A*(v_max/10)**B #in kpc

    
    # self.r_s=r_max/const #in kpc
    # self.rho_s=mass_max/(4.0*torch.pi*self.r_s**3)/(torch.log(1.0+const)-const/(1+const))#in M_sun/kpc^3


"""
============================================SOURCE DICT================================================
"""
def get_source_dict(source_mode, z, lens_model=None, lens_grid=None):
    available_source_modes = [
        "Gaussian_blob_no_rotation",
        "Gaussian_blob_rotation_random_pos_in_caustics",
        "Gaussian_blob_no_rotation_shrink",
        "Sersic_clumps"
    ]

    if source_mode not in available_source_modes:
        raise ValueError(f"Unknown source mode: {source_mode}. Check the available modes in make_systems_dicts.py")
    
    if source_mode == "Gaussian_blob_no_rotation":
        source_dict= {
            "type": "Gaussian_blob",
            "params": {
                "I": 1.0,
                "position_rad": [0.0, 0.0],
                "orient_rad": 0.0,
                "q": 0.8,
                "std_kpc": 0.8,
                "redshift": z
            }
        }
        
    if source_mode == "Gaussian_blob_no_rotation_shrink":
        source_dict= {
            "type": "Gaussian_blob",
            "params": {
                "I": 1.0,
                "position_rad": [0.0, 0.0],
                "orient_rad": 0.0,
                "q": 0.8,
                "std_kpc": 0.2,
                "redshift": z
            }
        }



    if source_mode == "Sersic_clumps":
        cfg = generate_sersic_clumps_config({})      # presampled parameters
        cfg["redshift"] = z                          # add the redshift tag
        source_dict = {
            "type": "Sersic_clumps",
            "params": cfg,
        }



    if lens_model is not None and lens_grid is not None:
        random_pos = lens_model.random_pos_inside_caustics(lens_grid, max_distance_arcsec=0, max_distance_std_caustic=True, verbose=False)
        if random_pos is not None:
            source_dict["params"]["position_rad"] = random_pos
        else:
            pass
    return source_dict
    
        
    # Implement other modes as needed.
    


"""
==================================Auxiliary functions==================================

"""


def get_zs_and_veldisp(mode, min_einstein_angle, num_samples, oversampling_factor=5, max_einstein_angle=None):
    """
    Build np arrays of redshifts for lens and source, velocity dispersions, and distances.
    Samples redshifts using the comoving volume distribution, swaps pairs if necessary,
    and filters by a minimum Einstein angle.
    """
    available_modes = [
        "min_einstein_angle_vel_disp_prior_100_400",
        "min_max_einstein_angle_vel_disp_prior_100_400",
        "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
        "resample_theta"
    ]

    if mode not in available_modes:
        raise ValueError(f"Unknown mode: {mode}")
    
    if mode == "min_einstein_angle_vel_disp_prior_100_400":
        redshifts_pool_lens = sample_redshift_comoving_volume(num_samples * oversampling_factor)
        redshifts_pool_source = sample_redshift_comoving_volume(num_samples * oversampling_factor)
        vel_disp_pool = uniform_prior(num_samples * oversampling_factor, 100, 400)
        
        # Swap pairs where the lens redshift is greater than the source redshift.
        swap_mask = redshifts_pool_lens > redshifts_pool_source
        if np.any(swap_mask):
            redshifts_pool_lens[swap_mask], redshifts_pool_source[swap_mask] = (
                redshifts_pool_source[swap_mask],
                redshifts_pool_lens[swap_mask],
            )
        
        # Compute angular diameter distances.
        D_l = cosmo.angular_diameter_distance(redshifts_pool_lens).value
        D_s = cosmo.angular_diameter_distance(redshifts_pool_source).value
        D_ls = cosmo.angular_diameter_distance_z1z2(redshifts_pool_lens, redshifts_pool_source).value

        # Compute the Einstein angle.
        theta_E = 4 * np.pi * (vel_disp_pool / units.c)**2 * D_ls / D_s

        # Apply the cut on the Einstein angle.
        valid_mask = theta_E > min_einstein_angle
        valid_count = valid_mask.sum()

        if valid_count >= num_samples:
            return (
                redshifts_pool_lens[valid_mask][:num_samples],
                redshifts_pool_source[valid_mask][:num_samples],
                vel_disp_pool[valid_mask][:num_samples],
                D_l[valid_mask][:num_samples],
                D_s[valid_mask][:num_samples],
                D_ls[valid_mask][:num_samples],
                theta_E[valid_mask][:num_samples],
            )
        else:
            percentage_valid = valid_count / (num_samples * oversampling_factor)
            new_oversampling_factor = 1 / percentage_valid + np.sqrt(1 / percentage_valid)
            raise ValueError(
                f"Not enough valid pairs found. Percentage of valid pairs: {percentage_valid:.2%}. "
                f"Suggested new oversampling factor: {new_oversampling_factor}"
            )

    if mode == "min_max_einstein_angle_vel_disp_prior_100_400":
        max_einstein_angle = units._arcsec_to_rad(3)
        redshifts_pool_lens = sample_redshift_comoving_volume(num_samples * oversampling_factor)
        redshifts_pool_source = sample_redshift_comoving_volume(num_samples * oversampling_factor)
        vel_disp_pool = uniform_prior(num_samples * oversampling_factor, 100, 400)
        
        # Swap pairs where the lens redshift is greater than the source redshift.
        swap_mask = redshifts_pool_lens > redshifts_pool_source
        if np.any(swap_mask):
            redshifts_pool_lens[swap_mask], redshifts_pool_source[swap_mask] = (
                redshifts_pool_source[swap_mask],
                redshifts_pool_lens[swap_mask],
            )
        
        # Compute angular diameter distances.
        D_l = cosmo.angular_diameter_distance(redshifts_pool_lens).value
        D_s = cosmo.angular_diameter_distance(redshifts_pool_source).value
        D_ls = cosmo.angular_diameter_distance_z1z2(redshifts_pool_lens, redshifts_pool_source).value

        # Compute the Einstein angle.
        theta_E = 4 * np.pi * (vel_disp_pool / units.c)**2 * D_ls / D_s

        # Apply the cut on the Einstein angle.
        valid_mask = (theta_E > min_einstein_angle) & (theta_E < max_einstein_angle)
        valid_count = valid_mask.sum()

        if valid_count >= num_samples:
            return (
                redshifts_pool_lens[valid_mask][:num_samples],
                redshifts_pool_source[valid_mask][:num_samples],
                vel_disp_pool[valid_mask][:num_samples],
                D_l[valid_mask][:num_samples],
                D_s[valid_mask][:num_samples],
                D_ls[valid_mask][:num_samples],
                theta_E[valid_mask][:num_samples],
            )
        else:
            percentage_valid = valid_count / (num_samples * oversampling_factor)
            new_oversampling_factor = 1 / percentage_valid + np.sqrt(1 / percentage_valid)
            raise ValueError(
                f"Not enough valid pairs found. Percentage of valid pairs: {percentage_valid:.2%}. "
                f"Suggested new oversampling factor: {new_oversampling_factor}"
            )
    elif mode == "min_max_einstein_angle_vel_disp_prior_50_400_conor_like":
        max_einstein_angle = units._arcsec_to_rad(3)
        redshifts_pool_lens = sample_redshift_comoving_volume(num_samples * oversampling_factor, zmax=4.0)
        redshifts_pool_source = sample_redshift_comoving_volume(num_samples * oversampling_factor, zmax=6.0)
        vel_disp_pool = uniform_prior(num_samples * oversampling_factor, 50, 400)
        
        # Swap pairs where the lens redshift is greater than the source redshift.
        swap_mask = redshifts_pool_lens > redshifts_pool_source
        if np.any(swap_mask):
            redshifts_pool_lens[swap_mask], redshifts_pool_source[swap_mask] = (
                redshifts_pool_source[swap_mask],
                redshifts_pool_lens[swap_mask],
            )
        
        # Compute angular diameter distances.
        D_l = cosmo.angular_diameter_distance(redshifts_pool_lens).value
        D_s = cosmo.angular_diameter_distance(redshifts_pool_source).value
        D_ls = cosmo.angular_diameter_distance_z1z2(redshifts_pool_lens, redshifts_pool_source).value

        # Compute the Einstein angle.
        theta_E = 4 * np.pi * (vel_disp_pool / units.c)**2 * D_ls / D_s

        # Apply the cut on the Einstein angle.
        valid_mask = (theta_E > min_einstein_angle) & (theta_E < max_einstein_angle)
        valid_count = valid_mask.sum()

        if valid_count >= num_samples:
            return (
                redshifts_pool_lens[valid_mask][:num_samples],
                redshifts_pool_source[valid_mask][:num_samples],
                vel_disp_pool[valid_mask][:num_samples],
                D_l[valid_mask][:num_samples],
                D_s[valid_mask][:num_samples],
                D_ls[valid_mask][:num_samples],
                theta_E[valid_mask][:num_samples],
            )
        else:
            percentage_valid = valid_count / (num_samples * oversampling_factor)
            new_oversampling_factor = 1 / percentage_valid + np.sqrt(1 / percentage_valid)
            raise ValueError(
                f"Not enough valid pairs found. Percentage of valid pairs: {percentage_valid:.2%}. "
                f"Suggested new oversampling factor: {new_oversampling_factor}"
            )
            
            
            
    elif mode=="resample_theta":
        #z_l, z_s, val, D_l, D_s, D_ls, theta 
        from astropy.cosmology import Planck18 as cosmo
        import numpy as np
        
        zmin, zmax = 0.0, 5.0
        Nz         = 10_000
        z_grid     = np.linspace(zmin, zmax, Nz)
        # comoving volume V(z) and comoving distance χ(z):
        vol_grid   = cosmo.comoving_volume(z_grid).value
        chi_grid   = cosmo.comoving_distance(z_grid).value
        
        c= units.c
        
        results = resample_theta(
            num_samples,
            oversampling_factor,
            min_einstein_angle, 
            c,
            z_grid,
            vol_grid,
            chi_grid
        )
        return results
# ----------------------------------------------------------------------
# __main__ block for rapid testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Test with 5 systems for a quick check.
        systems = make_systems_dicts(num_systems=5,
                                     mode="example_mode",
                                     low_einstein_angle_arcsec=0.7,
                                     oversampling_factor=20)
        print("Generated lensing systems:")
        for system in systems:
            print(system)
    except Exception as e:
        print("Error encountered:", e)

        
        
        
        
def aux_func( precomp_dict, lens_z, low_lim_mass=None):
        from astropy import units as u
        from astropy.constants import G
        
        # HERE NOT USING THIS BUT FIXING BOUNDS AS CONOR
        # def get_r_max(M_max):
        #     A=0.344*u.kpc #kpc
        #     B=1.607

        #     M_max = M_max * u.M_sun  # Convert to astropy units
            
        #     const_1=(np.log(2.163+1.)+1/(2.163+1)-1)*4*np.pi
            
        #     print(const_1)
            
        #     G=G.to(u.kpc**3/u.M_sun*u.s**(-2))
            
        #     const_2=(    10* u.km/u.s     /1.64/np.sqrt(G)   *(2.163/A)**(1/B)    )**2
        #     print(const_2)
            
            
        #     r_s=((M_max/const_1/const_2).to(u.kpc**(1 + 2/B)))**(B/(B+2))
            
        #     r_max=2.163*r_s
            
        #     return r_max.to(u.kpc).value
        

        assert precomp_dict is not None, "precomp_dict must be provided for this mode, to get einstein angle estimate."
        # Check if the Einstein angle is available in the precomputed quantities.
        assert "Theta_E" in precomp_dict, "Einstein angle not found in precomp_dict."

        Theta_E = precomp_dict["Theta_E"]
        substructure_list = []
        #with equal probability, add either 0 or 1-4 substructures
        if np.random.rand() < 0.5:
            return []
        else:
            num_substructures = 1
            for i in range(num_substructures):
                # Get a random position within a disc of radius 2*Theta_E.
                r = np.sqrt(np.random.rand()) * 2 * Theta_E
                phi = np.random.rand() * 2 * np.pi
                if low_lim_mass==None:
                    raise ValueError("low_lim_mass must be provided for this mode, to get einstein angle estimate.")

                else:
                    mass_max=log_uniform_prior(n_samples=1, min_value=low_lim_mass, max_value=10**11).tolist()[0]
                    r_max_bounds_low=1.5#kpc

                r_max_bounds_high=28.0#kpc
                substructure_list.append({
                    "type": "NFW",
                    "is_substructure": True,
                    "params": {
                        "pos": [units._arcsec_to_rad(np.random.rand()*6.0-3), units._arcsec_to_rad(np.random.rand()*6.0-3)],
                        "mass_max": mass_max,
                        "r_max_kpc": uniform_prior(n_samples=1, min_value=r_max_bounds_low, max_value=r_max_bounds_high).tolist()[0],
                        "redshift": lens_z,
                    }
                })
            return substructure_list
        
        
        
        
        
def aux_func2( precomp_dict, lens_z, low_lim_mass=None):

        def get_r_max_from_M_max(M_max):
            const=2.16258
            A=0.344 #kpc
            B=1.607
            v_max=((1/const)*units.G/A*10**B*M_max)**(1/(B+2))#in km/s
            r_max=A*(v_max/10)**B #in kpc
            return r_max

        assert precomp_dict is not None, "precomp_dict must be provided for this mode, to get einstein angle estimate."
        # Check if the Einstein angle is available in the precomputed quantities.
        assert "Theta_E" in precomp_dict, "Einstein angle not found in precomp_dict."

        Theta_E = precomp_dict["Theta_E"]
        substructure_list = []
        #with equal probability, add either 0 or 1-4 substructures
        if np.random.rand() < 0.5:
            return []
        else:
            num_substructures = 1
            for i in range(num_substructures):
                # Get a random position within a disc of radius 2*Theta_E.
                r = np.sqrt(np.random.rand()) * 2 * Theta_E
                phi = np.random.rand() * 2 * np.pi
                if low_lim_mass==None:
                    mass_max=log_uniform_prior(n_samples=1, min_value=10**8.6, max_value=10**11).tolist()[0]
                    r_max_bounds_low=get_r_max_from_M_max(10**8.6)

                else:
                    mass_max=log_uniform_prior(n_samples=1, min_value=low_lim_mass, max_value=10**11).tolist()[0]
                    r_max_bounds_low=get_r_max_from_M_max(10**8.6)

                r_max_bounds_high=get_r_max_from_M_max(10**11)
                substructure_list.append({
                    "type": "NFW",
                    "is_substructure": True,
                    "params": {
                        "pos": [units._arcsec_to_rad(np.random.rand()*6.0-3), units._arcsec_to_rad(np.random.rand()*6.0-3)],
                        "mass_max": mass_max,
                        "r_max_kpc": uniform_prior(n_samples=1, min_value=r_max_bounds_low, max_value=r_max_bounds_high).tolist()[0],
                        "redshift": lens_z,
                    }
                })
            return substructure_list
        





def generate_sersic_clumps_config(precomp_dict, rng=None):
    """Return a JSON‑friendly *config_dict* for :class:`SersicClumps`.

    Keys follow the new naming convention: ``position_rad`` +
    ``relative_pos_single_blobs``.
    """

    rng = rng or np.random.default_rng()

    # ---------- number of clumps ----------------------------------------
    N = int(rng.integers(1, 5))

    # ---------- intrinsic properties ------------------------------------
    I = [1.0] * N
    
    # here in kpc, transformed to rad in SersicClumps
    R_ser_kpc = rng.uniform(0.1, 1.0, size=N).tolist()
    n = [1.0] * N

    ellipticity = []
    for _ in range(N):
        while True:
            ex, ey = rng.uniform(-0.5, 0.5, size=2)
            if ex * ex + ey * ey <= 0.16:
                ellipticity.append([float(ex), float(ey)])
                break

    # ---------- spatial layout -----------------------------------------
    if N == 1:
        relative_pos_single_blobs = [[0.0, 0.0]]
    else:
        standard_dev_kpc=0.1
        
        standard_dev_rad=standard_dev_kpc/(precomp_dict["D_s"]*1000)
        #print(f"standard dev bewteen sersic in arcsec: {_rad_to_arcsec(standard_dev_rad)}")
        
        sigma = standard_dev_rad
        rho   =0.# rng.uniform(-0.25, 0.25)
        cov   = np.array([[sigma ** 2, rho * sigma ** 2],
                          [rho * sigma ** 2, sigma ** 2]])
        
        relative_pos_single_blobs = rng.multivariate_normal(
            [0.0, 0.0], cov, size=N).tolist()

    # The *global* source centre; callers will often overwrite this later
    # position_rad = [0.0, 0.0] ASSIGNED ELSEWHERE

    # ---------- assemble ------------------------------------------------
    return {
        "I": I,
        "relative_pos_single_blobs": relative_pos_single_blobs,
        #"position_rad": position_rad.tolist(), assigned elswhere
        "R_ser_kpc": R_ser_kpc,
        "n": n,
        "ellipticity": ellipticity,
    }








