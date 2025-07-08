from substructure_classifier.training_stage_development import Stage
from substructure_classifier.substructure_classifier_development import SubstructureClassifier


example_config=Stage.get_example_config(return_config=True)
my_classifier=SubstructureClassifier("SimpleResnetWithNoise2")

example_config["training_catalog"]="conor_similar_cat_min_10e11_train_no_shear_small_high_conc.json"
example_config["validation_like_train_catalog"]="conor_similar_cat_min_10e11_val_no_shear_small_high_conc.json"
example_config["dataset_class_str"]="SingleTelescopeNoiseDataset"
example_config["dataset_config"]={
        "grid_width_arcsec":8.0,
        "grid_pixel_side":100,
        "broadcasting":False,
        "sky_level":0.05,
        "kernel_size":5,
        "kernel_sigma":1.0,
        "gain":100.0,
        "final_transform": True
}

example_config["samples_used_for_training"]=100000
example_config["samples_used_for_validation"]=3000
example_config["batch_size"]=256
example_config["jump_batch_val"]=50
example_config["learning_rate"]=0.001
example_config["epochs"]=5
my_stage=Stage(classifier_instance=my_classifier, config=example_config, device="cuda")

my_stage.train(train_ready=True, early_stopping=True)

example_config["training_catalog"]="conor_similar_cat_min_10e11_train_no_shear_small.json"
example_config["validation_like_train_catalog"]="conor_similar_cat_min_10e11_val_no_shear_small.json"
example_config["dataset_class_str"]="SingleTelescopeNoiseDataset"
example_config["dataset_config"]={
        "grid_width_arcsec":8.0,
        "grid_pixel_side":100,
        "broadcasting":False,
        "sky_level":0.05,
        "kernel_size":5,
        "kernel_sigma":1.0,
        "gain":100.0,
        "final_transform": True
}

example_config["samples_used_for_training"]=100000
example_config["samples_used_for_validation"]=3000
example_config["batch_size"]=256
example_config["jump_batch_val"]=50
example_config["learning_rate"]=0.0001
example_config["epochs"]=5
my_stage=Stage(classifier_instance=my_classifier, config=example_config, device="cuda")

my_stage.train(train_ready=True, early_stopping=False)
