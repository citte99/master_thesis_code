"""
    The sampling procedure is the following:

    Geometry

    Lens

    Source

    Substructure

"""
from enum import Enum
from dataclasses import dataclass,  asdict, field
import torch
from abc import ABC, abstractmethod
from typing import Union, Optional, List
import numpy as np

from .distributions import uniform_prior, sample_redshift_comoving_volume
from .distributions import log_uniform_prior, random_pos_in_circle, resample_theta
from shared_utils import units
from shared_utils import _arcsec_to_rad
from config import CATALOGS_DIR
import os
# the idea here is that we will write 
# to a file both the methods used for
# sampling and their parameters


@dataclass
class SamplingInputs:
    ThetaE_min   : np.float64 = np.float64(0.)
    ThetaE_max   : np.float64 = np.inf
    prior_ThetaE : Optional[torch.Tensor] = None
    prior_lens_pos: Optional[torch.Tensor] = None
    prior_lens_VelDisp: Optional[torch.Tensor] = None
    prior_lens_q      : Optional[torch.Tensor] = None
    prior_lens_slope_normal: Optional[torch.Tensor] = None #mean, std
    prior_lens_orient : Optional[torch.Tensor] = None
    prior_shear_s: Optional[torch.Tensor] = None
    prior_shear_d: Optional[torch.Tensor] = None

    prior_sub_max_n: Optional[torch.Tensor] = None
    prior_sub_pos: Optional[torch.Tensor] = None
    prior_sub_log_mass: Optional[torch.Tensor] = None
    prior_sub_r_max: Optional[torch.Tensor] = None # set for no strict mass-conc relations.

    prior_source_I: Optional[torch.Tensor] = None           #if only one number, is fixed.Noise properties: is in [erg s^-1 cm ^ -2 arcsec^ -2 ]
    prior_source_std_main: Optional[torch.Tensor] = None    #if only one number, is fixed
    prior_source_q : Optional[torch.Tensor] = None,         #if only one number, is fixed
    prior_source_orient : Optional[torch.Tensor] = None,    #if only one number, is fixed

    prior_source_frac_of_theta_pos : Optional[torch.Tensor] = None 
    

#==================================MAPS OF THE ORDER OF PARAMETERS==========================================
"""
    To broadcast efficiently while generating the images, the lensingsystem class needs plain tensors
    as inputs of its various components. To keep track of what index corresponds to what, these are the
    conventions. Respect them or everything will be wrong. If you add new components in the overall system,
    you must come here and add your new component with its convention.

"""
precomp_map = ['D_l', 'D_s', 'D_ls', 'Theta_E']
    
#NOTE: These should be externalized.
mass_param_map = {
    'SIS': ['pos_x','pos_y','redshift','vel_disp'],
    'NFW': ['pos_x', 'pos_y','mass_max','r_max_kpc','redshift'],
    'ExternalPotential': ['shear_x','shear_y','shear_strength','shear_angle_arcsec'],
    'PEMD': ['pos_x', 'pos_y', 'redshift', 'vel_disp', 'slope', 'orient', 'q']

}

source_param_map = {
    'Gaussian_blob': ['position_rad_x', 'position_rad_y', 'I' ,'orient_rad','q','std_kpc','redshift']
    # add other source types here
}
#==========================================================================================================

'''
    For the lens logic structure, we have some problem.
    Some properties of the lens must be defined early on, in particular the geometrical configuration
    and the velocity dispersion have to be computed early, non only if we want to set some particular
    distribution of einstein angle, but also to just set einstein angle boundaries.

    At the same time, these quantities are not reaquired for every lens component, (velocity dispersion).
    This devides the mass components in two classe: those which can assume the role
    of main lens, and those which cannot. The former must have the structure described.
    The latter can be the same models, or also models that do not require those parameters (ext shear).


'''
#=======================================Lens description  ==================================================
@dataclass
class GenericMass_params(ABC):
    @abstractmethod
    def dict_for_pth(self):
        pass
    @abstractmethod
    def dict_for_json(self):
        pass

@dataclass 
class LensParams(GenericMass_params):
    # these are "Main Lens Params", and vel disp is required. We also need it because
    # we need to sample z and vel disp before deciding which particular lens we will use 
    # ( this last thing was problably not necessary).
    z          : Optional[torch.Tensor] = None
    pos        : Optional[torch.Tensor] = None
    vel_disp   : Optional[torch.Tensor] = None

    # the following are needed because I want this class to be not fully ABC
    def dict_for_pth(self):
        raise NotImplementedError("Calling a pseudo abstract method of LensParams")

    def dict_for_json(self):
        raise NotImplementedError("Calling a pseudo abstract method of LensParams")






@dataclass
class PEMD_params(LensParams):
    #'PEMD': ['pos_x', 'pos_y', 'redshift', 'vel_disp', 'slope', 'orient', 'q']
    slope      : Optional[torch.Tensor] = None
    orient     : Optional[torch.Tensor] = None
    q          : Optional[torch.Tensor] = None      

    @classmethod
    def from_base(
        cls,
        base : LensParams,
        *,
        slope : torch.Tensor,
        orient: torch.Tensor,
        q     : torch.Tensor
    ):
        data = asdict(base)
        return cls(**data, slope = slope, orient = orient, q = q)
    
    def dict_for_pth(self):
        dict={
            "PEMD":
                {
                    "params" : torch.stack([torch.as_tensor(a, dtype=torch.float32) for a in [self.pos[:, 0], self.pos[:, 1], self.z, self.vel_disp, self.slope, self.orient, self.q]], dim = 1),
                    "param_map" :   ['pos_x', 'pos_y', 'redshift', 'vel_disp', 'slope', 'orient', 'q'],
                    "sys_idx" : torch.arange(self.q.shape[0]) # NOTE: to use this model as sub, you need to wrap it in sub params,
                                                # and in particular this sys idx has to be taken care of 
                }
            }
        return dict

    def dict_for_json(self):
        pass



#This one is not a candidate for main mass, so only gets generic mass
@dataclass
class EXT_Shear_params(GenericMass_params):
    #'ExternalPotential': ['shear_x','shear_y','shear_strength','shear_angle_arcsec'],
    s   : Optional[torch.Tensor] = None
    d   : Optional[torch.Tensor] = None

    def dict_for_pth(self):
        dict={
            "ExternalPotential":
                {
                    "params" : torch.stack([torch.as_tensor(a, dtype=torch.float32) for a in [np.zeros(self.s.shape), np.zeros(self.s.shape), self.s, self.d]], dim = 1),
                    # the position does not make sense, has to be there but is just irrelevant
                    "param_map" :   ['shear_x','shear_y','shear_strength','shear_angle_rad'],
                    "sys_idx" : torch.arange(self.s.shape[0]) # NOTE: to use this model as sub, you need to wrap it in sub params,
                                                # and in particular this sys idx has to be taken care of 
                }
            }
        return dict

    def dict_for_json(self):
        raise NotImplementedError("")



@dataclass 
class SubParams(GenericMass_params):
    pos_abs       : Optional[torch.Tensor] = None
    belonging_index       : Optional[torch.Tensor] = None
    z : torch.Tensor = torch.inf # NOTE : in this research, in the calculation we 
                                      # only use the precomputed quantities, so this is irrelevant.
                                      # The substructrues are considered to be at the same redshift 
                                      # of the lens. I set this to inf, so that if accidentally called
                                      # it breaks stuff.

    def dict_for_pth(self):
        raise NotImplementedError("Calling a pseudo abstract method of SubParams")

    def dict_for_json(self):
        raise NotImplementedError("Calling a pseudo abstract method of SubParams")



@dataclass
class Sub_NFW(SubParams):
    #'NFW': ['pos_x', 'pos_y','mass_max','r_max_kpc','redshift'],
    M_max : Optional[torch.Tensor] = None
    r_max : Optional[torch.Tensor] = None

    @classmethod
    def from_base(
        cls,
        base  : SubParams, 
        *,   
        M_max : torch.Tensor, 
        r_max : torch.Tensor
    ):  
        
        data = asdict(base)
        return cls(**data, M_max=M_max, r_max = r_max)
    
    def dict_for_pth(self):
        zs = np.full_like(self.M_max, self.z) # these are handled according to what is writted in SubParams
        dict={
            "NFW":
                {
                    "params" : torch.stack([torch.as_tensor(a, dtype=torch.float32) for a in [self.pos_abs[:, 0], self.pos_abs[:, 1], self.M_max, self.r_max, zs]], dim = 1),
                    # the position does not make sense, has to be there but is just irrelevant
                    "param_map" :   ['pos_x', 'pos_y','mass_max','r_max_kpc','redshift'],
                    "sys_idx" : torch.tensor(self.belonging_index)
                }
            }
        return dict

    def dict_for_json(self):
        pass



#==========================================SOURCE INTIALIZATION=====================================================
@dataclass
class SourceParams(): #pseudo abc
    z          : Optional[torch.Tensor] = None
    pos_abs        : Optional[torch.Tensor] = None

    def dict_for_pth(self):
        raise NotImplementedError("Calling a pseudo abstract method of SourceParams")

        
    def dict_for_json(self):
        raise NotImplementedError("Calling a pseudo abstract method of SourceParams")




@dataclass 
class Gauss_params(SourceParams):
    #'Gaussian_blob': ['position_rad_x', 'position_rad_y', 'I' ,'orient_rad','q','std_kpc','redshift']
    std_kpc : torch.Tensor = None
    I       : torch.Tensor = None # This one for the noise properties is in [erg s^-1 cm ^ -2 arcsec^ -2 ]
    orient  : torch.Tensor = None # rad
    q       : torch.Tensor = None


    @classmethod
    def from_base(
        cls,
        base : SourceParams,
        *,
        std_kpc : torch.Tensor,
        I       : torch.Tensor,
        q       : torch.Tensor, 
        orient  : torch.Tensor

    ):
        cfg_base = asdict(base)
        return cls(**cfg_base, std_kpc = std_kpc, I = I, q = q, orient = orient)

    def dict_for_pth(self):
        dict={
            "Gaussian_blob":
                {
                    "params" : torch.stack([torch.as_tensor(a, dtype=torch.float32) for a in [self.pos_abs[:, 0], self.pos_abs[:, 1], self.I, self.orient, self.q, self.std_kpc, self.z ]], dim = 1),
                    "param_map" :   ['position_rad_x', 'position_rad_y', 'I' ,'orient_rad','q','std_kpc','redshift'],
                    "sys_idx" : torch.arange(self.I.shape[0])
                }
            }
        return dict

    def dict_for_json(self):
        pass







#===============================================================================================================

@dataclass 
class Precomp:
    #precomp_map = ['D_l', 'D_s', 'D_ls', 'Theta_E']
    
    D_l        : Optional[torch.Tensor] = None
    D_ls       : Optional[torch.Tensor] = None
    D_s        : Optional[torch.Tensor] = None
    theta_E    : Optional[torch.Tensor] = None



    def dict_for_pth(self):
        dict={
            "precomputed":
                {
                    "params" : torch.stack([torch.as_tensor(a, dtype=torch.float32) for a in [self.D_l, self.D_s, self.D_ls, self.theta_E]], dim = 1),
                    "param_map" :   ['D_l', 'D_s', 'D_ls', 'Theta_E'],
                    "sys_idx" : torch.arange(self.D_l.shape[0])
                }
            }
        return dict

    def dict_for_json(self):
        pass

        
@dataclass
class FullSysConfig:
    precomp    : Precomp
    lens_par   : LensParams # this is for future applications
    source_par : Union[SourceParams, list]  # this is for future applications                
    sub_par    : Union[SubParams, Sub_NFW]   
    secondary_lens_par : Optional[List[GenericMass_params]] = field(default_factory=list)

    def get_json(self):
        pass

    def get_pth(self, save_path = None):
        """
            Here I translate to the standard input of lensing system broadcasting.
            I also need to add the recording of the sampling parameters and pipeling,
            but that maybe should be in the function sampler rather than here.
        """
        # Note that substructure and lens masses are treated equally in the lensing pypline, as mass components
        # If you think about adding external shear then, LensParams must become a list. 
        # But I think we still could rely on a gerarchy


        # first of all, we need the mapping of the classes with the param maps, and conponents names
        # be careful about adding the indexes to the elements that do not have them (for the subhaloes,
        # you can use their indexes, as they are already ready.)

        # add the main lens to the mass components, 
        # then loop over the secondary components and add them as well.
        # correctly add the substructrures



        # as my phylosofy, I decided to have 1 source per system. 
        # eventual multiple blobs will be handled as a single source.

        # need to watch out for same keys in the masses components: in that case
        # I must concatenate the two same compoents.

        precomp_dict = self.precomp.dict_for_pth()

        masses_dict = {}

        masses_dict.update(self.lens_par.dict_for_pth())

        masses_dict.update(self.sub_par.dict_for_pth())

        for secondary_mass in self.secondary_lens_par:
            masses_dict.update(secondary_mass.dict_for_pth())

        source_dict = self.source_par.dict_for_pth()
        
        have_sub = set(self.sub_par.belonging_index)
        labels = torch.tensor([
            1 if i in have_sub else 0
            for i in range(self.lens_par.z.shape[0])
        ])
        labels_dict = {
            'labels' : {
                "sys_idx" : torch.arange(self.lens_par.z.shape[0]),
                "label_values" : labels
            }
        }

        my_list = [
            precomp_dict,
            { 'mass_components' : masses_dict },
            { 'source_models' : source_dict},
            labels_dict
        ]
        complete_pth_dict = { k: v for d in my_list for k, v in d.items()}
        
        if save_path is not None:
            torch.save(complete_pth_dict, save_path)
            print(f"Flat catalog written to {save_path}")

        return complete_pth_dict



#===============================================================GEOMETRICS SAMPLING=========================================================


import numpy as np
from astropy.cosmology import Planck18 as cosmo



def RedVelDispTrainSampler(
    full_sys_conf   : FullSysConfig,
    Nsamples        : int,
    sampling_inputs : SamplingInputs,
    overSampFac     : int = 20,
)-> FullSysConfig:
    
    Theta_E_min     = units._arcsec_to_rad(sampling_inputs.ThetaE_min)  # Convert Theta_E_min to radians
    Theta_E_max     = units._arcsec_to_rad(sampling_inputs.ThetaE_max)  # Convert Theta_E_max to radians
    
    max_einstein_angle = Theta_E_max
    redshifts_pool_lens = sample_redshift_comoving_volume(Nsamples * overSampFac)
    redshifts_pool_source = sample_redshift_comoving_volume(Nsamples * overSampFac)
    
    vel_disp_pool = uniform_prior(Nsamples * overSampFac, sampling_inputs.prior_lens_VelDisp[0], sampling_inputs.prior_lens_VelDisp[1])
    
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
    valid_mask = (theta_E > Theta_E_min) & (theta_E < max_einstein_angle)
    valid_count = valid_mask.sum()

    if valid_count >= Nsamples:
        # Fill precomputed values
        full_sys_conf.precomp.D_l = D_l[valid_mask][:Nsamples]
        full_sys_conf.precomp.D_ls = D_ls[valid_mask][:Nsamples]
        full_sys_conf.precomp.D_s = D_s[valid_mask][:Nsamples]
        full_sys_conf.precomp.theta_E = theta_E[valid_mask][:Nsamples]

        # Fill lens parameters
        full_sys_conf.lens_par.z = redshifts_pool_lens[valid_mask][:Nsamples]
        full_sys_conf.lens_par.vel_disp = vel_disp_pool[valid_mask][:Nsamples]

        # Fill source parameters
        full_sys_conf.source_par.z = redshifts_pool_source[valid_mask][:Nsamples]

        # Return the full system configuration
        return full_sys_conf
    else:
        percentage_valid = valid_count / (Nsamples * overSampFac)
        new_oversampling_factor = 1 / percentage_valid + np.sqrt(1 / percentage_valid)
        raise ValueError(
            f"Not enough valid pairs found. Percentage of valid pairs: {percentage_valid:.2%}. "
            f"Suggested new oversampling factor: {new_oversampling_factor}"
        )
def RedVelDispRealSampler(
    
):
    raise NotImplementedError("Implement RedVelDispRealSampler!")

class RedshiftsVelDispModes(Enum):
    # Parameters : vel disp prior,
    #              min theta E
    #              max theta E
    #              direct theta prior
    REALISTIC_DISTRIBUTION = RedVelDispRealSampler
    TRAINING_DISTRIBUTION  = RedVelDispTrainSampler

#=======================================================================MAIN LENS SAMPLING========================================================================


def pemd_sampler(
    full_sys_conf   : FullSysConfig,
    Nsamples        : int,
    sampling_inputs : SamplingInputs,
)-> FullSysConfig:
    
    #add a check that these required properties are already well defined: THIS DOES NOT SEEM TO BE NEEDED.
    # z_lens   = full_sys_conf.lens_par.z
    # vel_disp = full_sys_conf.lens_par.vel_disp

    # need to sample slope, q, orient, 
    pos_prior_rad = _arcsec_to_rad(np.array(sampling_inputs.prior_lens_pos))

    pos_x = uniform_prior(n_samples=Nsamples, min_value=pos_prior_rad[0], max_value=pos_prior_rad[1])
    pos_y = uniform_prior(n_samples=Nsamples, min_value=pos_prior_rad[0], max_value=pos_prior_rad[1])
    pos = np.stack([pos_x, pos_y], axis = 1)

    full_sys_conf.lens_par.pos = pos
    # mean, std
    slope = np.random.normal(loc = sampling_inputs.prior_lens_slope_normal[0], scale = sampling_inputs.prior_lens_slope_normal[1], size=Nsamples)
    orient = uniform_prior(Nsamples, sampling_inputs.prior_lens_orient[0], sampling_inputs.prior_lens_orient[1])
    q = uniform_prior(Nsamples, sampling_inputs.prior_lens_q[0], sampling_inputs.prior_lens_q[1])

    lens_params = PEMD_params.from_base(full_sys_conf.lens_par, slope = slope, orient = orient, q = q)

    full_sys_conf.lens_par = lens_params

    return full_sys_conf


class MainLensModes(Enum):
    PEMD = pemd_sampler
    

#============================================================SECONDARY LENS COMPONENTS SAMPLING=======================




def external_shear_sampler(
    full_sys_conf   : FullSysConfig,
    Nsamples        : int,
    sampling_inputs : SamplingInputs,
)-> FullSysConfig:
    
    shear_s_prior = sampling_inputs.prior_shear_s
    s = uniform_prior(Nsamples, shear_s_prior[0], shear_s_prior[1])

    shear_d_prior = sampling_inputs.prior_shear_d
    d = uniform_prior(Nsamples, shear_d_prior[0], shear_d_prior[1])
    ext_shear = EXT_Shear_params(
        s = s,
        d = d
    )

    full_sys_conf.secondary_lens_par.append(ext_shear)
    


    return full_sys_conf


class SecondaryLensModes(Enum):
    EXTERNAL_SHEAR = external_shear_sampler



#=============================================================SUBSTRUCTURE SAMPLING==========================================================================
from shared_utils.physics_relations import r_max_moline

def nfw_subs_base( 
    full_sys_conf   : FullSysConfig,
    Nsamples        : int,
    sampling_inputs : SamplingInputs,
)-> FullSysConfig:
    """
        does not fix r_max, it is done by the specialized functions below
    """

    max_subs = sampling_inputs.prior_sub_max_n

    #sample a number from 0 to max_subs
    rng = np.random.default_rng()
    n_sub = rng.integers(max_subs, size = Nsamples, endpoint= True)

    tot_subs = n_sub.sum()
    belongings = np.repeat(np.arange(0, Nsamples), n_sub)
    # I think its easyier to generate more: Nsamples * max_subs, and just discard the unused ones

    log_M_max = uniform_prior(tot_subs, min_value=sampling_inputs.prior_sub_log_mass[0], max_value=sampling_inputs.prior_sub_log_mass[1])
    M_max = 10**log_M_max

    prior_pos_rad = _arcsec_to_rad(np.array(sampling_inputs.prior_sub_pos))

    pos_rel_center_x = uniform_prior(tot_subs, prior_pos_rad[0],  prior_pos_rad[1])
    pos_rel_center_y = uniform_prior(tot_subs, prior_pos_rad[0],  prior_pos_rad[1])

    pos_rel_center = np.stack([pos_rel_center_x, pos_rel_center_y], axis = 1)

    pos_main_lens = np.repeat(full_sys_conf.lens_par.pos, n_sub, axis = 0)
    pos_abs = pos_main_lens + pos_rel_center

    base_sub = SubParams(
        pos_abs=pos_abs,
        belonging_index=belongings
    )

    full_sys_conf.sub_par=Sub_NFW.from_base(
        base_sub,
        M_max=M_max,
        r_max= None #this is temporary, until it is filled by the 'child' functions
    )


    return full_sys_conf


def nfw_subs_fixed_r_max( 
    full_sys_conf   : FullSysConfig,
    Nsamples        : int,
    sampling_inputs : SamplingInputs,
)-> FullSysConfig:
    """
        fixes r max from M _max trough Moline et al relation as in Conor
    """
    #call the base configurator for NFW
    full_sys_conf = nfw_subs_base(
        full_sys_conf=full_sys_conf,
        Nsamples=Nsamples,
        sampling_inputs=sampling_inputs
    )
    # then set the unset r_max
    M_max = full_sys_conf.sub_par.M_max
    r_max = r_max_moline(M_max)

    full_sys_conf.sub_par.r_max = r_max
    return full_sys_conf

def nfw_subs_free_r_max( 
    full_sys_conf   : FullSysConfig,
    Nsamples        : int,
    sampling_inputs : SamplingInputs,
)-> FullSysConfig:
    """
        sets r max according to prior
    """
    #call the base configurator for NFW
    full_sys_conf = nfw_subs_base(
        full_sys_conf=full_sys_conf,
        Nsamples=Nsamples,
        sampling_inputs=sampling_inputs
    )
    # then set the unset r_max
    tot_subs = len (full_sys_conf.sub_par.belonging_index)
    
    r_max = uniform_prior(tot_subs, min_value=sampling_inputs.prior_sub_r_max[0], max_value=sampling_inputs.prior_sub_r_max[12])

    full_sys_conf.sub_par.r_max = r_max
    return full_sys_conf

class SubStrucModes(Enum):
    NFW_subs_FIXED_R_MAX = nfw_subs_fixed_r_max
    NFW_SUBS_FREE_R_MAX  = nfw_subs_free_r_max

#====================================================SOURCE SAMPLING=================================================
def gauss_source( 
    full_sys_conf   : FullSysConfig,
    Nsamples        : int,
    sampling_inputs : SamplingInputs,
)-> FullSysConfig:
    
    I_prior = sampling_inputs.prior_source_I

    match isinstance(I_prior, float):
        case True:
            I = np.zeros(Nsamples)+I_prior
        case False:
            I = uniform_prior(Nsamples, I_prior[0], I_prior[1])
    
    std_prior = sampling_inputs.prior_source_std_main

    match isinstance(std_prior, float):
        case True:
            std = np.zeros(Nsamples)+ std_prior
        case False:
            std = uniform_prior(Nsamples, std_prior[0], std_prior[1])

    q_prior = sampling_inputs.prior_source_q

    match isinstance(q_prior, float):
        case True:
            q = np.zeros(Nsamples)+ q_prior
        case False:
            q = uniform_prior(Nsamples, q_prior[0], q_prior[1])

    orient_prior = sampling_inputs.prior_source_orient

    match isinstance(orient_prior, float):
        case True:
            orient = np.zeros(Nsamples)+ orient_prior
        case False:
            orient = uniform_prior(Nsamples, orient_prior[0], orient_prior[1])


    gauss_source = Gauss_params.from_base(
        full_sys_conf.source_par,
        std_kpc = std,
        I= I,
        q = q,
        orient = orient
    )

    full_sys_conf.source_par = gauss_source

    return full_sys_conf

class SourceModes(Enum):
    GAUSS_SOURCE = gauss_source


#===================================================SOURCE POSITIONING==============================================


def rand_frac_theta_e( 
    full_sys_conf   : FullSysConfig,
    Nsamples        : int,
    sampling_inputs : SamplingInputs,
)-> FullSysConfig:
    
    
    main_lens_pos = full_sys_conf.lens_par.pos

    theta_e = full_sys_conf.precomp.theta_E
    frac_theta = sampling_inputs.prior_source_frac_of_theta_pos
    
    # random pos in circle should be defined better



    pos_abs = random_pos_in_circle(frac_theta, Nsamples)*theta_e[..., None] + main_lens_pos
    
    full_sys_conf.source_par.pos_abs =pos_abs

    return full_sys_conf


class SourcePosModes(Enum):
    RAND_FRAC_THETA_E = rand_frac_theta_e


#==================================================================================================================

# a pipline is made of sampling functions



def Sampler(
        Pipeline : list, #List of sampler functions, FullSysConf -> FullSysConf
        sampling_inputs : SamplingInputs,
        N_samples : int,
        cat_name : str
    ):
    #Initialize systems configurations
    full_sys_conf = FullSysConfig(
        precomp     = Precomp(),
        lens_par    = LensParams(),
        source_par  = SourceParams(),
        sub_par     = SubParams() 
    )

    for sampler in Pipeline:
        full_sys_conf = sampler(full_sys_conf, N_samples, sampling_inputs)
    
    cat_path =  os.path.join(CATALOGS_DIR, ("testin.pth")
    full_sys_conf.
    return full_sys_conf



class SampModes(Enum): # currrently not working, poprobably this jupyter is not working with the suggestions
    REDSHIFTVELDISP = RedshiftsVelDispModes,
    MAINLENS        = MainLensModes,
    SECONDARYLENS   = SecondaryLensModes,
    SUB             = SubStrucModes,
    SOURCE          = SourceModes,
    SOURCEPOS       = SourcePosModes



                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             

def test():

    full_sys_conf = FullSysConfig(
        precomp     = Precomp(),
        lens_par    = LensParams(),
        source_par  = SourceParams(),
        sub_par     = []
    )


    sampling_inputs = SamplingInputs(
        prior_lens_orient=[0, 2.* np.pi],
        prior_lens_pos=[0., 1e-15],
        prior_lens_q= [0.2, 1.0],
        prior_lens_slope_normal=[1.0, 0.1],

        prior_sub_max_n = 4,
        prior_sub_pos= [- 3.0, + 3.0 ],
        prior_sub_log_mass= [6., 11.],

        prior_source_I = 1e-16,         #if only one number, is fixed
        prior_source_std_main = 2., #kpc,
        prior_source_frac_of_theta_pos= 0.3,
        prior_source_orient= 0.,
        prior_source_q = 1. ,

        prior_shear_d = [0., np.pi],
        prior_shear_s = [0., 1]
    )

    full_sys_conf = RedshiftsVelDispModes.TRAINING_DISTRIBUTION(
        full_sys_conf,
        Nsamples= 10,
        sampling_inputs = sampling_inputs
    )
    #print(f"Test RedshiftsVelDispModes.TRAINING_DISTRIBUTION, result type {type(full_sys_conf)}")


    My_Pypeline = [
        RedshiftsVelDispModes.TRAINING_DISTRIBUTION,
        MainLensModes.PEMD,
        SecondaryLensModes.EXTERNAL_SHEAR,
        SubStrucModes.NFW_subs_FIXED_R_MAX,
        SourceModes.GAUSS_SOURCE,
        SourcePosModes.RAND_FRAC_THETA_E

    ]

    full_sys_conf = Sampler(
        My_Pypeline,
        sampling_inputs,
        N_samples=10
    )

    #now we have to test if everything is fine in the configuration file for the lensing model to run

    import os

    path =  os.path.join(CATALOGS_DIR, ("testin.pth"))

    dict_conf = full_sys_conf.get_pth(save_path=path)

    

    from deep_learning import NoNoiseDataset, custom_dataloader

    dataset = NoNoiseDataset(
        catalog_name = 'testin',
        grid_pixel_side = 100, 
        grid_width_arcsec = 6.,
        broadcasting= True  
    )

    dataloader = custom_dataloader(dataset, 2)

    iterator = iter(dataloader)

    image = next(iterator)

    print(image[0].shape)
    
    import matplotlib.pyplot as plt
    
    plt.imshow(image[0].cpu().numpy()[0][0])
    plt.show()
    
    return full_sys_conf
