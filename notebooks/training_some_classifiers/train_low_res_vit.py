import torch

torch.set_float32_matmul_precision('high')



from substructure_classifier.training_stage_development import Stage
from substructure_classifier.substructure_classifier_development import SubstructureClassifier


example_config=Stage.get_example_config(return_config=True)
my_classifier=SubstructureClassifier("VitNoiseLowRes2")


example_config["training_catalog"]="conor_train_gauss_source_10e11_resample_theta_train"

example_config["validation_like_train_catalog"]="conor_train_gauss_source_10e11_resample_theta_val"
example_config["dataset_class_str"]="SingleTelescopeNoiseDataset"
example_config["dataset_config"]={
            "grid_width_arcsec":8.0,
            "grid_pixel_side":100,
            "broadcasting":True,
            "sky_level":0.05,
            "kernel_size":5,
            "kernel_sigma":1.0,
            "gain":500.0,
            "final_transform": False
}

example_config["samples_used_for_training"]=1000000#1000000
example_config["samples_used_for_validation"]=3000
example_config["batch_size"]=1024
example_config["jump_batch_val"]=400
example_config["learning_rate"]=0.0001
example_config["epochs"]=6
my_stage=Stage(classifier_instance=my_classifier, config=example_config, device="cuda")

my_stage.train(train_ready=True, early_stopping=False)






example_config=Stage.get_example_config(return_config=True)
my_classifier=SubstructureClassifier("VitNoiseLowRes2")


example_config["training_catalog"]="conor_train_gauss_source_10e10_resample_theta_train"

example_config["validation_like_train_catalog"]="conor_train_gauss_source_10e10_resample_theta_val"
example_config["dataset_class_str"]="SingleTelescopeNoiseDataset"
example_config["dataset_config"]={
            "grid_width_arcsec":8.0,
            "grid_pixel_side":100,
            "broadcasting":True,
            "sky_level":0.05,
            "kernel_size":5,
            "kernel_sigma":1.0,
            "gain":500.0,
            "final_transform": False
}

example_config["samples_used_for_training"]=1000000#1000000
example_config["samples_used_for_validation"]=3000
example_config["batch_size"]=1024
example_config["jump_batch_val"]=400
example_config["learning_rate"]=0.00001
example_config["epochs"]=3
my_stage=Stage(classifier_instance=my_classifier, config=example_config, device="cuda")

my_stage.train(train_ready=True, early_stopping=False)






example_config["training_catalog"]="conor_train_gauss_source_10e9_resample_theta_train"

example_config["validation_like_train_catalog"]="conor_train_gauss_source_10e9_resample_theta_val"
example_config["dataset_class_str"]="SingleTelescopeNoiseDataset"
example_config["dataset_config"]={
            "grid_width_arcsec":8.0,
            "grid_pixel_side":100,
            "broadcasting":True,
            "sky_level":0.05,
            "kernel_size":5,
            "kernel_sigma":1.0,
            "gain":500.0,
            "final_transform": False
}

example_config["samples_used_for_training"]=1000000#1000000
example_config["samples_used_for_validation"]=3000
example_config["batch_size"]=1024
example_config["jump_batch_val"]=400
example_config["learning_rate"]=0.00001
example_config["epochs"]=3
my_stage=Stage(classifier_instance=my_classifier, config=example_config, device="cuda")

my_stage.train(train_ready=True, early_stopping=False)





example_config["training_catalog"]="conor_train_gauss_source_10e8_6_resample_theta_train"

example_config["validation_like_train_catalog"]="conor_train_gauss_source_10e8_6_resample_theta_val"
example_config["dataset_class_str"]="SingleTelescopeNoiseDataset"
example_config["dataset_config"]={
            "grid_width_arcsec":8.0,
            "grid_pixel_side":100,
            "broadcasting":True,
            "sky_level":0.05,
            "kernel_size":5,
            "kernel_sigma":1.0,
            "gain":500.0,
            "final_transform": False
}

example_config["samples_used_for_training"]=1000000#1000000
example_config["samples_used_for_validation"]=3000
example_config["batch_size"]=1024
example_config["jump_batch_val"]=400
example_config["learning_rate"]=0.00001
example_config["epochs"]=3
my_stage=Stage(classifier_instance=my_classifier, config=example_config, device="cuda")

my_stage.train(train_ready=True, early_stopping=False)
