#!/usr/bin/env python
# coding: utf-8

# In[15]:


from substructure_classifier.substructure_classifier_development import SubstructureClassifier
import json
example_config=SubstructureClassifier.get_example_classifier_config()
print(json.dumps(example_config, indent=4))

example_config["classifier_name"]="SimpleResnetWithNoise1"

example_config["active_val_cats_live"]=[
    ["SIS_10e10_sub_val", "SingleTelescopeNoiseDataset", {
        "grid_width_arcsec":6.0,
        "grid_pixel_side":100,
        "broadcasting":False,
        "sky_level":0.05,
        "kernel_size":5,
        "kernel_sigma":1.0,
        "gain":100.0,

    }

    ],
     ["SIS_10e9_sub_val", "SingleTelescopeNoiseDataset", {
        "grid_width_arcsec":6.0,
        "grid_pixel_side":100,
        "broadcasting":False,
        "sky_level":0.05,
        "kernel_size":5,
        "kernel_sigma":1.0,
        "gain":100.0,

    }

    ]
]

example_config["NN_config"]={
        "num_classes":2
}


my_classifier = SubstructureClassifier("SimpleResnetWithNoise1", config_dict=example_config)


# In[ ]:





# In[16]:


from substructure_classifier.training_stage_development import Stage
from substructure_classifier.substructure_classifier_development import SubstructureClassifier


example_config=Stage.get_example_config(return_config=True)
my_classifier=SubstructureClassifier("SimpleResnetWithNoise1")

example_config["training_catalog"]="conor_similar_cat_min_10e11_train_small.json"
example_config["validation_like_train_catalog"]="conor_similar_cat_min_10e11_val.json"
example_config["dataset_class_str"]="SingleTelescopeNoiseDataset"
example_config["dataset_config"]={
        "grid_width_arcsec":6.0,
        "grid_pixel_side":100,
        "broadcasting":False,
        "sky_level":0.05,
        "kernel_size":5,
        "kernel_sigma":1.0,
        "gain":100.0,
        "final_transform": True
}

example_config["samples_used_for_training"]=100000
example_config["samples_used_for_validation"]=1000
example_config["batch_size"]=1024
example_config["jump_batch_val"]=50
example_config["learning_rate"]=0.001
example_config["epochs"]=5
my_stage=Stage(classifier_instance=my_classifier, config=example_config, device="cuda")

my_stage.train(train_ready=True, early_stopping=True)

example_config["training_catalog"]="conor_similar_cat_min_10e11_train_small.json"
example_config["validation_like_train_catalog"]="conor_similar_cat_min_10e11_val.json"
example_config["dataset_class_str"]="SingleTelescopeNoiseDataset"
example_config["dataset_config"]={
        "grid_width_arcsec":6.0,
        "grid_pixel_side":100,
        "broadcasting":False,
        "sky_level":0.05,
        "kernel_size":5,
        "kernel_sigma":1.0,
        "gain":100.0,
        "final_transform": True
}

example_config["samples_used_for_training"]=100000
example_config["samples_used_for_validation"]=1000
example_config["batch_size"]=1024
example_config["jump_batch_val"]=50
example_config["learning_rate"]=0.0001
example_config["epochs"]=5
my_stage=Stage(classifier_instance=my_classifier, config=example_config, device="cuda")

my_stage.train(train_ready=True, early_stopping=True)


# In[17]:


from substructure_classifier.training_stage_development import Stage
from substructure_classifier.substructure_classifier_development import SubstructureClassifier



example_config=Stage.get_example_config(return_config=True)
my_classifier=SubstructureClassifier("SimpleResnetWithNoise1")

example_config["training_catalog"]="conor_similar_cat_min_10e11_train_small.json"
example_config["validation_like_train_catalog"]="conor_similar_cat_min_10e11_val.json"
example_config["dataset_class_str"]="SingleTelescopeNoiseDataset"
example_config["dataset_config"]={
        "grid_width_arcsec":6.0,
        "grid_pixel_side":100,
        "broadcasting":False,
        "sky_level":0.05,
        "kernel_size":5,
        "kernel_sigma":1.0,
        "gain":100.0,
        "final_transform": True
}


example_config["samples_used_for_training"]=100000
example_config["samples_used_for_validation"]=1000
example_config["batch_size"]=1024
example_config["jump_batch_val"]=50
example_config["learning_rate"]=0.00001
example_config["epochs"]=5
my_stage=Stage(classifier_instance=my_classifier, config=example_config, device="cuda")

my_stage.train(train_ready=True, early_stopping=True)


# In[ ]:


example_config=Stage.get_example_config(return_config=True)
my_classifier=SubstructureClassifier("SimpleResnetWithNoise1")

example_config["training_catalog"]="conor_similar_cat_min_10e10_train_small.json"
example_config["validation_like_train_catalog"]="conor_similar_cat_min_10e10_val.json"
example_config["dataset_class_str"]="SingleTelescopeNoiseDataset"
example_config["dataset_config"]={
        "grid_width_arcsec":6.0,
        "grid_pixel_side":100,
        "broadcasting":False,
        "sky_level":0.05,
        "kernel_size":5,
        "kernel_sigma":1.0,
        "gain":100.0,
        "final_transform": True
}


example_config["samples_used_for_training"]=100000
example_config["samples_used_for_validation"]=1000
example_config["batch_size"]=1024
example_config["jump_batch_val"]=50
example_config["learning_rate"]=0.0003
example_config["epochs"]=5
my_stage=Stage(classifier_instance=my_classifier, config=example_config, device="cuda")

my_stage.train(train_ready=True, early_stopping=True)


# In[ ]:


example_config=Stage.get_example_config(return_config=True)
my_classifier=SubstructureClassifier("SimpleResnetWithNoise1")

example_config["training_catalog"]="conor_similar_cat_min_10e10_train_small.json"
example_config["validation_like_train_catalog"]="conor_similar_cat_min_10e10_val.json"
example_config["dataset_class_str"]="SingleTelescopeNoiseDataset"
example_config["dataset_config"]={
        "grid_width_arcsec":6.0,
        "grid_pixel_side":100,
        "broadcasting":False,
        "sky_level":0.05,
        "kernel_size":5,
        "kernel_sigma":1.0,
        "gain":100.0,
        "final_transform": True
}


example_config["samples_used_for_training"]=100000
example_config["samples_used_for_validation"]=2000
example_config["batch_size"]=1024
example_config["jump_batch_val"]=50
example_config["learning_rate"]=0.00001
example_config["epochs"]=5
my_stage=Stage(classifier_instance=my_classifier, config=example_config, device="cuda")

my_stage.train(train_ready=True, early_stopping=True)


# In[9]:


example_config=Stage.get_example_config(return_config=True)
my_classifier=SubstructureClassifier("SimpleResnetWithNoise1")

example_config["training_catalog"]="conor_similar_cat_min_10e10_train_small.json"
example_config["validation_like_train_catalog"]="conor_similar_cat_min_10e10_val.json"
example_config["dataset_class_str"]="SingleTelescopeNoiseDataset"
example_config["dataset_config"]={
        "grid_width_arcsec":6.0,
        "grid_pixel_side":100,
        "broadcasting":False,
        "sky_level":0.05,
        "kernel_size":5,
        "kernel_sigma":1.0,
        "gain":100.0,
        "final_transform": True
}


example_config["samples_used_for_training"]=100000
example_config["samples_used_for_validation"]=2000
example_config["batch_size"]=1024
example_config["jump_batch_val"]=50
example_config["learning_rate"]=0.000001
example_config["epochs"]=5
my_stage=Stage(classifier_instance=my_classifier, config=example_config, device="cuda")

my_stage.train(train_ready=True, early_stopping=True)


# In[ ]:


example_config=Stage.get_example_config(return_config=True)
my_classifier=SubstructureClassifier("SimpleResnetWithNoise1")

example_config["training_catalog"]="conor_similar_cat_min_10e9_train_small.json"
example_config["validation_like_train_catalog"]="conor_similar_cat_min_10e9_val.json"
example_config["dataset_class_str"]="SingleTelescopeNoiseDataset"
example_config["dataset_config"]={
        "grid_width_arcsec":6.0,
        "grid_pixel_side":100,
        "broadcasting":False,
        "sky_level":0.05,
        "kernel_size":5,
        "kernel_sigma":1.0,
        "gain":100.0,
        "final_transform": True
}


example_config["samples_used_for_training"]=100000
example_config["samples_used_for_validation"]=1000
example_config["batch_size"]=1024
example_config["jump_batch_val"]=50
example_config["learning_rate"]=0.0001
example_config["epochs"]=5
my_stage=Stage(classifier_instance=my_classifier, config=example_config, device="cuda")

my_stage.train(train_ready=True, early_stopping=True)

example_config=Stage.get_example_config(return_config=True)

my_classifier=SubstructureClassifier("SimpleResnetWithNoise1")

example_config["training_catalog"]="conor_similar_cat_min_10e9_train_small.json"
example_config["validation_like_train_catalog"]="conor_similar_cat_min_10e9_val.json"
example_config["dataset_class_str"]="SingleTelescopeNoiseDataset"
example_config["dataset_config"]={
        "grid_width_arcsec":6.0,
        "grid_pixel_side":100,
        "broadcasting":False,
        "sky_level":0.05,
        "kernel_size":5,
        "kernel_sigma":1.0,
        "gain":100.0,
        "final_transform": True
}


example_config["samples_used_for_training"]=100000
example_config["samples_used_for_validation"]=1000
example_config["batch_size"]=1024
example_config["jump_batch_val"]=50
example_config["learning_rate"]=0.00001
example_config["epochs"]=5
my_stage=Stage(classifier_instance=my_classifier, config=example_config, device="cuda")

my_stage.train(train_ready=True, early_stopping=True)


# In[ ]:


example_config=Stage.get_example_config(return_config=True)
my_classifier=SubstructureClassifier("SimpleResnetWithNoise1")

example_config["training_catalog"]="conor_similar_cat_train_small"
example_config["validation_like_train_catalog"]="conor_similar_cat_val"
example_config["dataset_class_str"]="SingleTelescopeNoiseDataset"
example_config["dataset_config"]={
        "grid_width_arcsec":6.0,
        "grid_pixel_side":100,
        "broadcasting":False,
        "sky_level":0.05,
        "kernel_size":5,
        "kernel_sigma":1.0,
        "gain":100.0,
        "final_transform": True
}


example_config["samples_used_for_training"]=100000
example_config["samples_used_for_validation"]=2500
example_config["batch_size"]=1024
example_config["jump_batch_val"]=50
example_config["learning_rate"]=0.0001
example_config["epochs"]=5
my_stage=Stage(classifier_instance=my_classifier, config=example_config, device="cuda")

my_stage.train(train_ready=True, early_stopping=True)


# In[ ]:


example_config=Stage.get_example_config(return_config=True)
my_classifier=SubstructureClassifier("SimpleResnetWithNoise4")

example_config["training_catalog"]="conor_similar_cat_train_small"
example_config["validation_like_train_catalog"]="conor_similar_cat_val"
example_config["dataset_class_str"]="SingleTelescopeNoiseDataset"
example_config["dataset_config"]={
        "grid_width_arcsec":6.0,
        "grid_pixel_side":100,
        "broadcasting":False,
        "sky_level":0.05,
        "kernel_size":5,
        "kernel_sigma":1.0,
        "gain":100.0,
        "final_transform": True
}


example_config["samples_used_for_training"]=100000
example_config["samples_used_for_validation"]=2500
example_config["batch_size"]=1024
example_config["jump_batch_val"]=50
example_config["learning_rate"]=0.00001
example_config["epochs"]=5
my_stage=Stage(classifier_instance=my_classifier, config=example_config, device="cuda")

my_stage.train(train_ready=True, early_stopping=True)


# In[ ]:


example_config=Stage.get_example_config(return_config=True)
my_classifier=SubstructureClassifier("SimpleResnetWithNoise4")

example_config["training_catalog"]="conor_similar_cat_train_small"
example_config["validation_like_train_catalog"]="conor_similar_cat_val"
example_config["dataset_class_str"]="SingleTelescopeNoiseDataset"
example_config["dataset_config"]={
        "grid_width_arcsec":6.0,
        "grid_pixel_side":100,
        "broadcasting":False,
        "sky_level":0.05,
        "kernel_size":5,
        "kernel_sigma":1.0,
        "gain":100.0,
        "final_transform": True
}


example_config["samples_used_for_training"]=100000
example_config["samples_used_for_validation"]=2500
example_config["batch_size"]=1024
example_config["jump_batch_val"]=50
example_config["learning_rate"]=0.000001
example_config["epochs"]=5
my_stage=Stage(classifier_instance=my_classifier, config=example_config, device="cuda")

my_stage.train(train_ready=True, early_stopping=True)


# In[ ]:




