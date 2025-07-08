import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os 
import json

from shared_utils import _grid_lens
from deep_learning import ResNet50
import numpy as np

from config import TRAINED_CLASSIFIERS_DIR
import hashlib
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os 
import json

from shared_utils import _grid_lens
from deep_learning import ResNet50
import numpy as np

from config import TRAINED_CLASSIFIERS_DIR
import hashlib

# Import dataset classes
from deep_learning import AlmaSinglePsfDataset
from deep_learning import custom_dataloader

class Stage:
    """
    A stage represents a particular stage in the training process.
    It is characterized by:
        -the parent stage, whose parameters are used to initialize the current stage.
        -the catalog it is trained on
        -the percentage of the catalog it is trained on
        -the number of epochs it is trained for
        -the batch size
        -the learning rate (eventually the mode, for evolving learning rates)
    
    The NN model and dataset are now configurable based on the classifier config.
    The lens_grid is fixed by the classifier class.
    """
    def __init__(self, classifier_name, stage_config_dict=None, stage_id=None):
        self._initialize_classifier_properties(classifier_name)

        if stage_config_dict is not None and stage_id is None:
            self._initialize_new_stage(stage_config_dict)
        
        elif stage_config_dict is None and stage_id is not None:
            self._load_existing_stage(stage_id)

        else:
            raise ValueError("Either stage_config_dict or stage_id must be provided, but not both.")
        
        # Initialize arrays to store training and test losses
        self.train_losses = []
        self.test_catalog_losses = {catalog: [] for catalog in self.test_catalogs}

    def _initialize_classifier_properties(self, classifier_name):
        self.classifier_name = classifier_name
        classifier_config_path = os.path.join(TRAINED_CLASSIFIERS_DIR, classifier_name, "classifier_config.json")
        with open(classifier_config_path, "r") as f:
            self.classifier_config = json.load(f)

        # Initialize model based on config
        self.NN_model = self._create_model_from_config(self.classifier_config)
        
        # Setup grid parameters
        grid_size = self.classifier_config.get("grid_params", {}).get("width_arcsec", 8.0)
        grid_pixel_width = self.classifier_config.get("grid_params", {}).get("pixel_width", 224)
        self.grid_lens = _grid_lens(grid_size_arcsec=grid_size, image_res=grid_pixel_width, device="cpu")
        
        # Other properties
        self.parent_stage = self.classifier_config.get("active_stage")
        self.test_catalogs = self.classifier_config.get("test_catalogs", [])
        
        # Dataset type
        self.dataset_type = self.classifier_config.get("dataset_type", "ResNetDataset")
        self.dataset_params = self.classifier_config.get("dataset_params", {})
        
    def _create_model_from_config(self, config):
        """Create the appropriate neural network model based on the configuration."""
        model_type = config.get("NN_model", "ResNet50")
        
        if model_type == "ResNet50":
            return ResNet50(num_classes=2)
        
        elif model_type == "VisionTransformer":
            # Extract ViT-specific parameters
            vit_params = config.get("vit_params", {})
            return VisionTransformer_Custom(
                num_classes=2,
                img_size=vit_params.get("img_size", 224),
                patch_size=vit_params.get("patch_size", 32),
                embed_dim=vit_params.get("embed_dim", 384),
                depth=vit_params.get("depth", 6),
                num_heads=vit_params.get("num_heads", 6),
                mlp_ratio=vit_params.get("mlp_ratio", 2.0),
                drop_rate=vit_params.get("drop_rate", 0.1)
            )
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _create_dataset(self, catalog_name, samples_used, uncropped_grid, mode="on_gpu_generation"):
        """Create the appropriate dataset based on the configuration."""
        if self.dataset_type == "ResNetDataset":
            print("outdated dataset class ResNetDataset")
            raise NotImplementedError("ResNetDataset is outdated. Please use NoNoiseDataset instead.")
            return ResNetDataset(
                catalog_name, 
                use_only_a_percent=use_only_a_percent, 
                mode=mode, 
                uncropped_grid=uncropped_grid
            )
        
        elif self.dataset_type == "AlmaSinglePsfDataset":
            # Extract noise parameters from config
            noise_std = self.dataset_params.get("noise_std", 0.0)
            threshold = self.dataset_params.get("threshold", None)
            
            return AlmaSinglePsfDataset(
                catalog_name=catalog_name,
                psf_name="fake_psf",  # PSF handling should be added if needed
                samples_used=samples_used, 
                image_data_type=torch.float32,
                noise_std=noise_std,
                threshold=None,
                broadcasting=False
            )
          
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def _initialize_new_stage(self, stage_config_dict):
        # The parent stage is set by the higher classifier class. If there is none active one there, we spawn an orphan stage.
        if self.parent_stage is not None:
            self._initialize_new_stage_from_parent(stage_config_dict)
        else:
            self._initialize_new_parentless_stage(stage_config_dict)

    def _initialize_new_parentless_stage(self, stage_config_dict):
        # Make the stage id by hashing the stage config + the parent stage id
        stage_config_str = json.dumps(stage_config_dict, sort_keys=True)
        self.stage_id = hashlib.md5((stage_config_str + str(self.parent_stage)).encode()).hexdigest()
        self.train_catalog = stage_config_dict["train_catalog"]
        self.samples_used_for_train= stage_config_dict["samples_used_for_train"]
        self.samples_used_for_test = stage_config_dict["samples_used_for_test"]
        self.epochs = stage_config_dict["epochs"]
        self.batch_size = stage_config_dict["batch_size"]
        self.lr = stage_config_dict["learning_rate"]
        self.write_stage_config()

    def _initialize_new_stage_from_parent(self, stage_config_dict):
        # Make the stage id by hashing the stage config + the parent stage id
        stage_config_str = json.dumps(stage_config_dict, sort_keys=True)
        self.stage_id = hashlib.md5((stage_config_str + str(self.parent_stage)).encode()).hexdigest()
        self.train_catalog = stage_config_dict["train_catalog"]
        self.samples_used_for_train = stage_config_dict["samples_used_for_train"]
        self.samples_used_for_test = stage_config_dict["samples_used_for_test"]
        self.epochs = stage_config_dict["epochs"]
        self.batch_size = stage_config_dict["batch_size"]
        self.lr = stage_config_dict["learning_rate"]
        self.write_stage_config()

        # Load the parent stage model 
        parent_params_path = os.path.join(TRAINED_CLASSIFIERS_DIR, self.classifier_name, "stages", self.parent_stage, "trained_parameters.pth")
        self.NN_model.load_state_dict(torch.load(parent_params_path))

    def write_stage_config(self):
        stage_config_path = os.path.join(TRAINED_CLASSIFIERS_DIR, self.classifier_name, "stages", self.stage_id, "stage_config.json")
        print(stage_config_path)
        stage_config_dict = {
            "train_catalog": self.train_catalog,
            "samples_used_for_train": self.samples_used_for_train,
            "samples_used_for_test": self.samples_used_for_test,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "parent_stage": self.parent_stage
        }
        # Make the directory if it doesn't exist
        os.makedirs(os.path.dirname(stage_config_path), exist_ok=False)
        with open(stage_config_path, "w") as f:
            json.dump(stage_config_dict, f, indent=4)

    def _load_existing_stage(self, stage_id):
        stage_config_path = os.path.join(TRAINED_CLASSIFIERS_DIR, self.classifier_name, "stages", stage_id, "stage_config.json")
        with open(stage_config_path, "r") as f:
            stage_config_dict = json.load(f)

        self.stage_id = stage_id
        self.train_catalog = stage_config_dict["train_catalog"]
        self.samples_used_for_train = stage_config_dict["samples_used_for_train"]
        self.samples_used_for_test = stage_config_dict["samples_used_for_test"]
        self.epochs = stage_config_dict["epochs"]
        self.batch_size = stage_config_dict["batch_size"]
        self.lr = stage_config_dict["learning_rate"]
        self.parent_stage = stage_config_dict["parent_stage"]

        params_path = os.path.join(TRAINED_CLASSIFIERS_DIR, self.classifier_name, "stages", self.stage_id, "trained_parameters.pth")
        self.NN_model.load_state_dict(torch.load(params_path))

        # Load loss data if available
        loss_path = os.path.join(TRAINED_CLASSIFIERS_DIR, self.classifier_name, "stages", self.stage_id, "loss_data.npz")
        if os.path.exists(loss_path):
            loss_data = np.load(loss_path, allow_pickle=True)
            self.train_losses = loss_data['train_losses'].tolist()
            self.test_catalog_losses = loss_data.get('test_catalog_losses', {})
            if not isinstance(self.test_catalog_losses, dict):
                self.test_catalog_losses = self.test_catalog_losses.item()  # Convert from numpy object array to dict

    def evaluate_on_test_catalogs(self, model, device, criterion, test_loaders):
        """Evaluate the model on all test catalogs and return dictionary of losses"""
        model.eval()  # Set model to evaluation mode
        test_losses = {}
        
        with torch.no_grad():  # No gradients needed for evaluation
            for catalog, loader in test_loaders.items():
                running_loss = 0.0
                for images, labels in loader:
                    # Ensure images have a channel dimension
                    if images.ndim == 3:
                        images = images.unsqueeze(1)
                    
                    images = images.to(device)
                    labels = labels.to(device).long()
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                
                # Calculate average loss for this catalog
                test_losses[catalog] = running_loss / len(loader)
        
        return test_losses

    def save_loss_data(self):
            """Save the training and test loss data to a file"""
            stage_path = os.path.join(TRAINED_CLASSIFIERS_DIR, self.classifier_name, "stages", self.stage_id)
            loss_path = os.path.join(stage_path, "loss_data.npz")

            # Convert to numpy arrays for saving
            np.savez(
                loss_path, 
                train_losses=np.array(self.train_losses), 
                test_catalog_losses=np.array([self.test_catalog_losses], dtype=object)
            )
            print(f"Loss data saved to {loss_path}")

    def train(self):
        import time

        # Check if there is no pth in the folder
        stage_path = os.path.join(TRAINED_CLASSIFIERS_DIR, self.classifier_name, "stages", self.stage_id)
        assert not any(fname.endswith('.pth') for fname in os.listdir(stage_path)), \
               "It seems like this stage is already trained. Check if it has already trained parameters."

        model = self.NN_model
        print(f"Setting up model type: {self.classifier_config.get('NN_model', 'ResNet50')}")
        model.float()
        print("Setting device for training to CUDA")
        device = "cuda"
        model.to(device)

        print("Moving the grid to device for training")
        grid_lens = self.grid_lens.to(device)

        print(f"Creating dataset of type {self.dataset_type} for training")
        train_dataset = self._create_dataset(
            self.train_catalog, 
            samples_used=self.samples_used_for_train, 
            uncropped_grid=grid_lens,
            mode="on_gpu_generation"
        )

        print("Creating dataloader for training")
        train_loader = custom_dataloader(train_dataset, batch_size=self.batch_size, num_workers=0)

        # Initialize the loaders for the testing catalogs
        test_loaders = {}
        for catalog in self.test_catalogs:
            test_dataset = self._create_dataset(
                catalog, 
                samples_used=self.samples_used_for_test, 
                uncropped_grid=grid_lens,
            )
            test_loaders[catalog] = custom_dataloader(test_dataset, batch_size=self.batch_size, num_workers=0)

        # Set up loss function and optimizer with weight decay for L2 regularization
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)

        num_epochs = self.epochs
        print("\n[DEBUG] Starting training...\n")

        # Reset loss arrays
        self.train_losses = []
        self.test_catalog_losses = {catalog: [] for catalog in self.test_catalogs}

        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            running_loss = 0.0
            total_train_correct = 0
            total_train_samples = 0

            epoch_start_time = time.time()

            for batch_idx, (images, labels) in enumerate(train_loader):
                # Ensure images have a channel dimension: [batch_size, 1, H, W]
                if images.ndim == 3:
                    images = images.unsqueeze(1)
                if batch_idx == 0:
                    print(f"[DEBUG] Epoch {epoch+1} first batch images shape: {images.shape}, labels shape: {labels.shape}")

                images = images.to(device)
                labels = labels.to(device).long()

                optimizer.zero_grad()       # Zero the gradients
                outputs = model(images)     # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()             # Backward pass
                optimizer.step()            # Update parameters

                running_loss += loss.item()

                # Compute training accuracy for this batch
                _, predicted = torch.max(outputs, 1)
                total_train_correct += (predicted == labels).sum().item()
                total_train_samples += labels.size(0)

            # Calculate epoch-level metrics
            epoch_loss = running_loss / len(train_loader)
            train_accuracy = total_train_correct / total_train_samples
            self.train_losses.append(epoch_loss)

            epoch_time = time.time() - epoch_start_time

            # Evaluate on test catalogs after each epoch
            test_losses = self.evaluate_on_test_catalogs(model, device, criterion, test_loaders)
            for catalog, loss in test_losses.items():
                self.test_catalog_losses[catalog].append(loss)

            # Print progress with additional details
            test_loss_str = ", ".join([f"{cat}: {loss:.4f}" for cat, loss in test_losses.items()])
            print(f"[DEBUG] Epoch {epoch+1} - Time: {epoch_time:.2f}s, Train Loss: {epoch_loss:.4f}, "
                  f"Train Accuracy: {train_accuracy*100:.2f}%, Test Losses: {test_loss_str}")

            # Save loss data after each epoch
            self.save_loss_data()


        # --- Saving the Model Parameters ---

        # Save the trained model's parameters to a file.
        model_save_path = os.path.join(stage_path, "trained_parameters.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"\n[DEBUG] Model parameters saved to {model_save_path}")

        # Edit the active stage in the classifier config
        classifier_config_path = os.path.join(TRAINED_CLASSIFIERS_DIR, self.classifier_name, "classifier_config.json")
        with open(classifier_config_path, "r") as f:
            classifier_config = json.load(f)
        
        classifier_config["active_stage"] = self.stage_id
        
        with open(classifier_config_path, "w") as f:
            json.dump(classifier_config, f, indent=4)

    def evaluate_accuracy_on_test_catalogs(self, device=None):
        """
        Evaluate the model's accuracy on all test catalogs.
        
        Args:
            device (torch.device, optional): The device to run evaluation on.
                                            If None, uses CUDA if available, else CPU.
        
        Returns:
            dict: Dictionary mapping catalog names to their accuracy percentages
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to the appropriate device
        self.NN_model.to(device)
        # Set the model to evaluation mode
        self.NN_model.eval()
        
        # Dictionary to store accuracy results for each catalog
        catalog_accuracies = {}
        
        # Move grid_lens to device if needed
        grid_lens = self.grid_lens.to(device)
        
        # Evaluate on each test catalog
        for catalog in self.test_catalogs:
            print(f"Evaluating on catalog: {catalog}")
            
            # Create dataset for this catalog using the configured dataset type
            test_dataset = self._create_dataset(
                catalog, 
                samples_used=self.samples_used_for_test,  # Use the entire catalog for evaluation
                uncropped_grid=grid_lens,
                mode="on_gpu_generation"
            )
            test_loader = custom_dataloader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)


            
            correct = 0
            total = 0
            
            # We don't need gradients for evaluation
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(test_loader):
                    
                    images = images.to(device)
                    labels = labels.to(device).long()
                    
                    outputs = self.NN_model(images)
                    # Get the predicted class with the highest score
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            # Calculate and store accuracy for this catalog
            if total > 0:  # Avoid division by zero
                accuracy = 100 * correct / total
                catalog_accuracies[catalog] = accuracy
                print(f"Accuracy on {catalog}: {accuracy:.2f}%")
            else:
                catalog_accuracies[catalog] = 0
                print(f"Warning: No samples in catalog {catalog}")
        
        return catalog_accuracies