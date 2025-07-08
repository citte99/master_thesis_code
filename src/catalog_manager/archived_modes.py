mode_mapping_archived={
        
        "example_mode_2": {
            "redshifts_and_vel_disp": "min_einstein_angle_vel_disp_prior_100_400",
            "main_lens_mode": "Single_PEMD_with_external_no_shear_conor_like_but_medium_ellipticity",
            "substructure_mode": "No_substructure",
            "source_mode": "Gaussian_blob_no_rotation",
            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
        "test": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_shear_conor_like_but_low_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_rigorous_min_10e11",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
    "test2": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_shear_conor_like_but_low_ellipticity",
            "substructure_mode": "No_substructure",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
    
    "test3": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_SIS",
            "substructure_mode": "No_substructure",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
    
    
    

        "test_multiple_sersic_sources": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_no_shear_conor_like_but_low_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_rigorous_min_10e11",
            "source_mode": "Sersic_clumps", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
    




        "SIS_10e11_sub": {
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_100_400",
            "main_lens_mode": "Single_SIS",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_2_einstein_radii_10e11",
            "source_mode": "Gaussian_blob_no_rotation",
            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,

        },
        "SIS_10e10_sub": {
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_100_400",
            "main_lens_mode": "Single_SIS",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_2_einstein_radii_10e10",
            "source_mode": "Gaussian_blob_no_rotation",
            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,

        },
        "SIS_10e9_sub": {
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_100_400",
            "main_lens_mode": "Single_SIS",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_2_einstein_radii_10e9",
            "source_mode": "Gaussian_blob_no_rotation",
            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,

        },
        "SIS_10e8_sub": {
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_100_400",
            "main_lens_mode": "Single_SIS",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_2_einstein_radii_10e8",
            "source_mode": "Gaussian_blob_no_rotation",
            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,

        },
    
    
        "conor_similar_train_gauss_source_low_ellipticity": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_shear_conor_like_but_low_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
    
    
    
    
    
        "conor_similar_train_gauss_source_low_ellipticity_min_sub_10e11": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_shear_conor_like_but_low_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_min_10e11",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
        "conor_similar_train_gauss_source_low_ellipticity_min_sub_10e11_no_shear": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_no_shear_conor_like_but_low_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_min_10e11",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
    
        "conor_similar_train_gauss_source_low_ellipticity_min_sub_10e11_no_shear": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_no_shear_conor_like_but_low_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_min_10e11",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
    
    
    
    
    
    
    
    
        "conor_similar_train_gauss_source_medium_ellipticity_min_sub_10e11_no_shear": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_no_shear_conor_like_but_medium_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_min_10e11",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
    
    
        "conor_similar_train_gauss_source_medium_ellipticity_min_sub_10e11_no_shear_rigorous": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_no_shear_conor_like_but_medium_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_rigorous_min_10e11",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
    
    
    
    
    
    
    
    
        "conor_similar_train_gauss_source_medium_ellipticity_min_sub_10e11_no_shear_shrink_source": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_no_shear_conor_like_but_medium_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_min_10e11",
            "source_mode": "Gaussian_blob_no_rotation_shrink", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
    
    
    
        "conor_similar_train_gauss_source_low_ellipticity_min_sub_10e11_no_shear_higher_conc": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_no_shear_conor_like_but_low_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_in_grid_6x6_arcsec_mass_conc_prior_conor_and_more_min_10e11",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
        "conor_similar_train_gauss_source_low_ellipticity_min_sub_10e11_no_shear_medium_conc": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_no_shear_conor_like_but_low_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_in_grid_6x6_arcsec_mass_conc_prior_conor_and_more_medium_min_10e11",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
        "conor_similar_train_gauss_source_low_ellipticity_min_sub_10e11_no_shear_medium_less_conc": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_no_shear_conor_like_but_low_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_in_grid_6x6_arcsec_mass_conc_prior_conor_and_more_medium_less_min_10e11",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
    
    
    
    
        "conor_similar_train_gauss_source_low_ellipticity_min_sub_10e10": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_shear_conor_like_but_low_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_min_10e10",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
        "conor_similar_train_gauss_source_low_ellipticity_min_sub_10e9": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_shear_conor_like_but_low_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_min_10e9",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
        "conor_similar_train_gauss_source_low_ellipticity_min_sub_10e8": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_shear_conor_like_but_low_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_min_10e8",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
    
    
    
    
    
    
        "conor_similar_train_gauss_source_low_ellipticity_min_sub_rigorous_10e11": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_shear_conor_like_but_low_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_rigorous_min_10e11",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
        "conor_similar_train_gauss_source_low_ellipticity_min_sub_rigorous_10e10": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_shear_conor_like_but_low_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_rigorous_min_10e10",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
        "conor_similar_train_gauss_source_low_ellipticity_min_sub_rigorous_10e9": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_shear_conor_like_but_low_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_rigorous_min_10e9",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        },
        "conor_similar_train_gauss_source_low_ellipticity_min_sub_rigorous_10e8": {    
            "redshifts_and_vel_disp": "min_max_einstein_angle_vel_disp_prior_50_400_conor_like",
            "main_lens_mode": "Single_PEMD_with_external_shear_conor_like_but_low_ellipticity",
            "substructure_mode": "Zero_or_1_NFW_no_mass_max_r_max_relation_in_grid_6x6_arcsec_mass_conc_prior_conor_like_rigorous_min_10e8",
            "source_mode": "Gaussian_blob_no_rotation", 

            "source_positioning": "inside_ein_angle_in_source_plane_0dot4", #or inside_ein_angle_in_source_plane
            "Precompute_quantities": True,
        }
}
