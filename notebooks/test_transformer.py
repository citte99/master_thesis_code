import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from catalog_manager import CatalogManager
from lensing_system import LensingSystem
from shared_utils import recursive_to_tensor, _grid_lens
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import gc
import math

# Re-define the model classes (must match the training implementation)
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Handle non-standard input sizes
        if H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0:
            pad_h = self.patch_size[0] - (H % self.patch_size[0])
            pad_w = self.patch_size[1] - (W % self.patch_size[1])
            
            if pad_h < self.patch_size[0] and pad_w < self.patch_size[1]:
                x = F.pad(x, (0, pad_w, 0, pad_h))
                _, _, H, W = x.shape
        
        x = self.proj(x)  # B, E, H', W'
        x = x.flatten(2).transpose(1, 2)  # B, N, E
        
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self, 
        img_size=224,
        patch_size=16, 
        in_channels=1, 
        num_classes=2, 
        embed_dim=768,
        depth=12,
        num_heads=12, 
        mlp_ratio=4., 
        qkv_bias=True, 
        representation_size=None,
        distilled=False,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.0, 
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + (2 if distilled else 1), embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer
            )
            for _ in range(depth)
        ])
        
        # Final normalization layer
        self.norm = norm_layer(embed_dim)
        
        # Classification head
        self.distilled = distilled
        self.num_classes = num_classes
        
        # Pre-logits if needed
        if representation_size:
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
        else:
            self.pre_logits = nn.Identity()
        
        # Setup for either standard or distilled classification
        if distilled:
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head_dist = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self.apply(self._init_weights_linear)
    
    def _init_weights_linear(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        B = x.shape[0]
        
        # Apply patch embedding
        x = self.patch_embed(x)
        
        # Interpolate position embeddings for non-standard sizes
        if x.size(1) != self.pos_embed.size(1) - 1:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            pos_embed = self.pos_embed[:, 1:, :]
            
            seq_length = x.size(1)
            pos_embed_width = int(math.sqrt(pos_embed.size(1)))
            pos_embed = pos_embed.reshape(1, pos_embed_width, pos_embed_width, -1).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed, 
                size=int(math.sqrt(seq_length)), 
                mode='bicubic', 
                align_corners=False
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, -1, x.size(2))
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)
        else:
            pos_embed = self.pos_embed
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        
        if self.distilled:
            dist_token = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_token, x), dim=1)
        
        # Add position embeddings
        x = x + pos_embed
        x = self.pos_drop(x)
        
        # Process through transformer blocks
        x = self.blocks(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Return appropriate tokens
        if self.distilled:
            return x[:, 0], x[:, 1]
        else:
            return x[:, 0]
    
    def forward(self, x):
        if self.distilled:
            x, x_dist = self.forward_features(x)
            x = self.head(x)
            x_dist = self.head_dist(x_dist)
            return (x + x_dist) / 2
        else:
            x = self.forward_features(x)
            x = self.head(x)
            return x


class OptimizedDatasetWithNoise(Dataset):
    def __init__(
        self, 
        catalog_name, 
        uncropped_grid, 
        psf_tensor=None, 
        noise_std=0.1, 
        threshold=None,
        use_only_a_percent=100  # Use all data for testing by default
    ):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available.")
            
        self.device = torch.device('cuda')
        
        # Initialize catalog
        self.catalog_obj = CatalogManager(catalog_name_input=catalog_name)
        self.catalog = self.catalog_obj.catalog
        
        # Filter catalog to reduce memory usage
        sl_systems_len = len(self.catalog["SL_systems"])
        max_systems = min(sl_systems_len, int(sl_systems_len * use_only_a_percent / 100))
        
        # Store indices only, not the full data
        self.catalog_indices = list(range(max_systems))
        print(f"Using {max_systems} systems out of {sl_systems_len}")
        
        # Store parameters
        self.uncropped_grid = uncropped_grid.to(self.device)
        print(f"Using grid with shape: {self.uncropped_grid.shape}")
        self.noise_std = noise_std
        self.threshold = threshold
        
        # Process original PSF
        if psf_tensor is not None:
            print(f"Original PSF shape: {psf_tensor.shape}")
            
            # Ensure PSF has correct dimensions for processing
            if psf_tensor.ndim == 2:
                self.psf = psf_tensor.unsqueeze(0).unsqueeze(0)
            elif psf_tensor.ndim == 3:
                self.psf = psf_tensor.unsqueeze(0)
            else:
                self.psf = psf_tensor
                
            print(f"Processed PSF shape: {self.psf.shape}")
            
            # Shift PSF for FFT - save GPU memory by pre-computing
            self.psf_shifted = torch.fft.ifftshift(self.psf, dim=(-2, -1))
        else:
            self.psf = None
            self.psf_shifted = None

    def __len__(self):
        return len(self.catalog_indices)
    
    def __getitem__(self, idx):
        """Get a single item with memory optimization"""
        # Process only the needed system to conserve memory
        system_idx = self.catalog_indices[idx]
        system_dict = self.catalog["SL_systems"][system_idx]
        system_dict_tensor = recursive_to_tensor(system_dict, self.device)
        
        # Get label
        num_sub = system_dict_tensor["lens_model"]["num_substructures"]
        label = int(num_sub > 0)
        label_tensor = torch.tensor(label, device=self.device).long()
        
        # Generate image
        lensing_system = LensingSystem(system_dict_tensor, self.device)
        with torch.no_grad():
            # Get base image from lensing system
            image_tensor = lensing_system(self.uncropped_grid)
            
            # Add channel dimension if needed
            if image_tensor.ndim == 2:
                image_tensor = image_tensor.unsqueeze(0)
            
            # Process with PSF and add noise
            if self.psf is not None or self.noise_std > 0:
                image_tensor = self._apply_psf_and_noise(image_tensor.unsqueeze(0)).squeeze(0)
        
        # Clean up memory
        del system_dict_tensor, lensing_system
        torch.cuda.empty_cache()
        
        return image_tensor, label_tensor
    
    def _apply_psf_and_noise(self, images):
        """Apply PSF convolution and noise with memory optimization"""
        # Skip if nothing to apply
        if self.psf is None and self.noise_std <= 0:
            return images
            
        batch_size, channels, H, W = images.shape
        
        # Apply PSF convolution
        if self.psf is not None:
            # Pad the images
            padding = (W//2, W//2, H//2, H//2)
            padded_images = F.pad(images, padding, mode='constant', value=0)
            
            # Compute FFT of padded images
            images_fft = torch.fft.fftn(padded_images, dim=(-2, -1))
            
            # Get appropriate slice of PSF_fft if dimensions don't match
            psf_fft = torch.fft.fftn(self.psf_shifted, dim=(-2, -1))
            
            if images_fft.shape[-2:] != psf_fft.shape[-2:]:
                print(f"Warning: FFT shape mismatch - images: {images_fft.shape}, PSF: {psf_fft.shape}")
                # Use direct noise addition instead
                processed_images = images + torch.randn_like(images) * self.noise_std
                return processed_images
            
            # Apply convolution in Fourier domain
            conv_fft = images_fft * psf_fft
            
            # Add Fourier domain noise if specified
            if self.noise_std > 0:
                # Add noise in Fourier domain (complex noise)
                noise_real = torch.randn_like(conv_fft.real) * self.noise_std
                noise_imag = torch.randn_like(conv_fft.imag) * self.noise_std
                noise_fourier = torch.complex(noise_real, noise_imag)
                conv_fft = conv_fft + noise_fourier
            
            # Transform back to spatial domain
            convolved_images = torch.fft.ifftn(conv_fft, dim=(-2, -1)).real
            
            # Crop back to original size
            start_h, start_w = H // 2, W // 2
            processed_images = convolved_images[:, :, start_h:start_h+H, start_w:start_w+W]
            
            # Clean up large intermediate tensors
            del padded_images, images_fft, psf_fft, conv_fft, convolved_images
            torch.cuda.empty_cache()
            
        # If only noise is needed (no PSF)
        elif self.noise_std > 0:
            processed_images = images + torch.randn_like(images) * self.noise_std
        else:
            processed_images = images
        
        # Apply thresholding if specified
        if self.threshold is not None:
            processed_images = torch.where(processed_images > self.threshold, 
                                           processed_images, 
                                           torch.zeros_like(processed_images))
            
        return processed_images
    
    def get_batch(self, indices):
        """Optimized batch processing"""
        # Generate images
        images = []
        labels = []
        
        for idx in indices:
            image, label = self.__getitem__(idx)
            images.append(image)
            labels.append(label)
            
        # Stack into batches
        images_batch = torch.stack(images)
        labels_batch = torch.stack(labels)
        
        return images_batch, labels_batch


class MemoryEfficientDataLoader:
    """Custom data loader that handles memory more efficiently"""
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.length = len(dataset)
        
    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.length).tolist()
        else:
            indices = list(range(self.length))
            
        for i in range(0, self.length, self.batch_size):
            # Process in small chunks
            batch_indices = indices[i:i + self.batch_size]
            yield self.dataset.get_batch(batch_indices)
            
            # Clean memory after each batch
            torch.cuda.empty_cache()
            
    def __len__(self):
        return (self.length + self.batch_size - 1) // self.batch_size


def evaluate_model():
    """Evaluate the trained model on test dataset"""
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Using device: {device}")
    
    # Load the original PSF - no substitution (same as training)
    try:
        with fits.open('../psfs/fake_psf.fits') as hdul:
            psf_data = hdul[0].data
            psf_data = psf_data.byteswap().newbyteorder()
        
        # Convert to tensor
        psf_tensor = torch.from_numpy(psf_data).float().to(device)
        
        # Determine PSF dimensions
        if psf_tensor.ndim == 2:
            number_of_pixels = psf_tensor.shape[0]
        elif psf_tensor.ndim == 3:
            number_of_pixels = psf_tensor.shape[1]
        elif psf_tensor.ndim == 4:
            number_of_pixels = psf_tensor.shape[2]
        else:
            raise ValueError(f"Unexpected PSF dimensions: {psf_tensor.shape}")
        
        print(f"PSF side pixels: {number_of_pixels}")
        
    except Exception as e:
        print(f"PSF loading failed: {e}, continuing without PSF")
        number_of_pixels = 200
        psf_tensor = None
    
    # Use half the PSF size for grid
    grid_size = round(number_of_pixels/2)
    print(f"Creating grid with size: {grid_size}")
    
    # Create grid with standard field of view (8.0 as in training)
    grid_lens = _grid_lens(8.0, grid_size, device=device)
    
    # Define ViT configuration - must match the trained model
    patch_size = 32
    
    # Create Vision Transformer model with same configuration as training
    model = VisionTransformer(
        img_size=number_of_pixels,
        patch_size=patch_size,
        in_channels=1,
        num_classes=2,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop_rate=0.0,  # No dropout for evaluation
    ).to(device)
    
    # Load the trained model weights
    model_path = 'vit_strong_lensing.pth'
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Create test dataset with same parameters as training
    test_dataset = OptimizedDatasetWithNoise(
        catalog_name="first_bad_testing_catalog.json",  # Use the test catalog
        uncropped_grid=grid_lens,
        psf_tensor=psf_tensor,
        noise_std=0.1,  # Keep same noise level as training
        threshold=None,
        use_only_a_percent=100  # Use all test data
    )
    
    # Create test data loader with larger batch size for evaluation
    batch_size = 4  # Can use larger batch size for evaluation
    test_loader = MemoryEfficientDataLoader(test_dataset, batch_size, shuffle=False)
    
    # Initialize variables to store predictions and ground truth
    all_predictions = []
    all_labels = []
    all_probs = []  # To store probability scores for ROC curve
    
    # Evaluation loop
    print("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Forward pass
            outputs = model(images)
            
            # Get class predictions and probabilities
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Store predictions, probabilities and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Store probability of positive class
            
            # Clean memory
            del images, labels, outputs, probs
            torch.cuda.empty_cache()
            
            # Print progress
            print(f"Processed batch {batch_idx+1}/{len(test_loader)}")
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=['No Substructure', 'Has Substructure']))
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['No Sub', 'Has Sub'], rotation=45)
    plt.yticks(tick_marks, ['No Sub', 'Has Sub'])
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix visualization to 'confusion_matrix.png'")
    
    # Generate ROC curve
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    print("Saved ROC curve to 'roc_curve.png'")
    
    # Generate Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    average_precision = average_precision_score(all_labels, all_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, 
            label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('precision_recall_curve.png')
    print("Saved Precision-Recall curve to 'precision_recall_curve.png'")
    
    # Save metrics to file
    with open('evaluation_results.txt', 'w') as f:
        f.write("Vision Transformer Evaluation Results\n")
        f.write("===================================\n\n")
        f.write(f"Test Dataset: my_cat_test.json\n")
        f.write(f"Number of test samples: {len(all_labels)}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUC-ROC: {roc_auc:.4f}\n")
        f.write(f"Average Precision: {average_precision:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix) + "\n\n")
        
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_predictions, 
                                    target_names=['No Substructure', 'Has Substructure']))
    
    print("\nSaved detailed results to 'evaluation_results.txt'")
    print("Evaluation complete!")


if __name__ == "__main__":
    evaluate_model()