from substructure_classifier.training_stage_development import Stage
from substructure_classifier.substructure_classifier_development import SubstructureClassifier



example_config=Stage.get_example_config(return_config=True)
my_classifier=SubstructureClassifier("SimpleResnet3")

example_config["training_catalog"]="SIS_10e8_sub_train"
example_config["validation_like_train_catalog"]="SIS_10e8_sub_val_extended"
example_config["dataset_class_str"]="NoNoiseDataset"
example_config["dataset_config"]={
        "grid_width_arcsec":6.0,
        "grid_pixel_side":100,
        "broadcasting":False
}

example_config["samples_used_for_training"]=4000
example_config["samples_used_for_validation"]=1000
example_config["batch_size"]=128
example_config["jump_batch_val"]=7
example_config["learning_rate"]=0.0000005
example_config["epochs"]=4
my_stage=Stage(classifier_instance=my_classifier, config=example_config, device="cuda")

my_stage.train(train_ready=True, early_stopping=False)