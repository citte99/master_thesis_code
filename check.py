#!/usr/bin/env python3
"""
Compute dataset-wide mean and standard deviation for multiple interferometric image catalogs
using NoNoiseDataset's get_batch method. Processes 10,000 samples in chunks of 1024.
"""
import math
from deep_learning.NN_datasets import NoNoiseDataset

def compute_mean_std(catalog_name: str,
                     img_size: int = 80,
                     img_width: float = 8.0,
                     upscaling: int = 5,
                     batch_size: int = 1024,
                     samples_used: int = 10000) -> (float, float):
    """
    Compute global mean and std for a given NoNoiseDataset catalog by manually
    drawing batches via its get_batch method.
    """
    # Initialize dataset (prints its own info)
    dataset = NoNoiseDataset(
        catalog_name,
        grid_pixel_side=img_size,
        grid_width_arcsec=img_width,
        broadcasting=True,
        samples_used=samples_used,
        upscaling=upscaling,
    )

    sum_ = 0.0
    sum_sq = 0.0
    count = 0

    # Determine how many batches to draw
    n_batches = math.ceil(samples_used / batch_size)

    for _ in range(n_batches):
        imgs, _ = dataset.get_batch(batch_size)
        # imgs shape: [B, C, H, W]
        B, C, H, W = imgs.shape
        flat = imgs.view(B, C, -1)
        sum_ += float(flat.sum())
        sum_sq += float((flat * flat).sum())
        count += B * H * W

    mean = sum_ / count
    var = (sum_sq / count) - (mean ** 2)
    std = var ** 0.5
    return mean, std


if __name__ == '__main__':
    catalogs = [
        'min_mass_10e11',
        'min_mass_10e10',
        'min_mass_10e9',
        'min_mass_10e8_6'
    ]

    for catalog in catalogs:
        mean, std = compute_mean_std(catalog)
        print(f"Catalog       : {catalog}")
        print(f"Samples used  : 10000")
        print(f"Batch size    : 1024")
        print(f"Mean          : {mean:.6e}")
        print(f"Std           : {std:.6e}\n")
        
# Debug script adapted for your custom dataset that uses get_batch instead of __getitem__.
# """

# import torch
# import torch.nn as nn
# import numpy as np
# from pathlib import Path
# import os

# def setup_single_gpu():
#     """Setup for single GPU debugging (no distributed)"""
#     if torch.cuda.is_available():
#         device = torch.device('cuda:0')
#         torch.cuda.set_device(0)
#     else:
#         device = torch.device('cpu')
#     return device

# def debug_your_data_pipeline():
#     """
#     Debug your actual data preprocessing pipeline using your custom dataset.
#     """
#     print("🔍 DEBUGGING YOUR DATA PREPROCESSING PIPELINE")
#     print("=" * 60)
    
#     device = setup_single_gpu()
#     print(f"Using device: {device}")
    
#     # Import your modules
#     from deep_learning.NN_datasets import NoNoiseDataset
#     from deep_learning.NN_datasets.dataloaders import distributed_dataloader
#     from noise_applicator.noisers.base_noiser import EuclidNoiserInterfPSF
    
#     # 1. Create dataset and get batch directly
#     print("📊 Step 1: Creating Dataset and Getting Batch")
#     print("-" * 40)
    
#     try:
#         dataset = NoNoiseDataset(
#             "min_mass_10e11",
#             grid_pixel_side=80,
#             grid_width_arcsec=8.0,
#             broadcasting=True,
#             samples_used=200,  # Small sample for debugging
#             upscaling=5,
#         )
        
#         # Get batch correctly - pass list of indices, not batch size
#         batch_size = 16
#         indices = list(range(batch_size))  # [0, 1, 2, ..., 15]
#         raw_images, labels = dataset.get_batch(indices)
        
#         print(f"✅ Successfully got batch using get_batch()")
#         print(f"Raw images shape: {raw_images.shape}")
#         print(f"Raw images dtype: {raw_images.dtype}")
#         print(f"Raw images device: {raw_images.device}")
#         print(f"Labels shape: {labels.shape}")
        
#         # Move to device if needed
#         if raw_images.device != device:
#             raw_images = raw_images.to(device)
#             labels = labels.to(device)
#             print(f"Moved data to {device}")
        
#         # Analyze raw data
#         print(f"Raw images range: [{raw_images.min():.6f}, {raw_images.max():.6f}]")
#         print(f"Raw images mean: {raw_images.mean():.6f}")
#         print(f"Raw images std: {raw_images.std():.6f}")
        
#         # Check for problematic values
#         has_nan = torch.isnan(raw_images).any()
#         has_inf = torch.isinf(raw_images).any()
#         print(f"Contains NaN: {has_nan}")
#         print(f"Contains Inf: {has_inf}")
        
#         if has_nan or has_inf:
#             print("❌ Raw data contains NaN/Inf values!")
#             return None
        
#         # Check data distribution
#         print(f"Raw data percentiles:")
#         percentiles = [1, 5, 25, 50, 75, 95, 99]
#         for p in percentiles:
#             val = torch.quantile(raw_images, p/100.0)
#             print(f"  {p}%: {val:.6f}")
        
#     except Exception as e:
#         print(f"❌ Error creating dataset or getting batch: {e}")
#         import traceback
#         traceback.print_exc()
#         return None
    
#     # 2. Test noiser
#     print(f"\n📊 Step 2: Testing Noiser")
#     print("-" * 40)
    
#     try:
#         noiser = EuclidNoiserInterfPSF()
#         noiser.set_device(device)
        
#         print(f"Noiser class: {noiser.__class__.__name__}")
#         print(f"Noiser device: {device}")
        
#         # Apply noiser
#         noised_images = noiser(raw_images)
        
#         print(f"✅ Noiser applied successfully")
#         print(f"Noised images shape: {noised_images.shape}")
#         print(f"Noised images dtype: {noised_images.dtype}")
#         print(f"Noised images range: [{noised_images.min():.6f}, {noised_images.max():.6f}]")
#         print(f"Noised images mean: {noised_images.mean():.6f}")
#         print(f"Noised images std: {noised_images.std():.6f}")
        
#         # Check for problematic values
#         has_nan = torch.isnan(noised_images).any()
#         has_inf = torch.isinf(noised_images).any()
#         print(f"Contains NaN: {has_nan}")
#         print(f"Contains Inf: {has_inf}")
        
#         if has_nan or has_inf:
#             print("❌ Noiser produces NaN/Inf values!")
#             return None
        
#         # Check for extreme values
#         extreme_threshold = 1000
#         extreme_pixels = (torch.abs(noised_images) > extreme_threshold).sum()
#         if extreme_pixels > 0:
#             print(f"⚠️  Found {extreme_pixels} pixels with absolute value > {extreme_threshold}")
#             max_val = torch.abs(noised_images).max()
#             print(f"    Maximum absolute value: {max_val:.2f}")
        
#         # Compare raw vs noised
#         noise_added = (noised_images - raw_images).abs().mean()
#         print(f"Average noise magnitude added: {noise_added:.6f}")
        
#     except Exception as e:
#         print(f"❌ Error applying noiser: {e}")
#         import traceback
#         traceback.print_exc()
#         return None
    
#     # 3. Test your normalizer
#     print(f"\n📊 Step 3: Testing Your Normalizer")
#     print("-" * 40)
    
#     class NormalizerInterf:
#         def __call__(self, imgs):
#             print(f"  Normalizer input shape: {imgs.shape}")
#             print(f"  Normalizer input range: [{imgs.min():.6f}, {imgs.max():.6f}]")
            
#             # Find min and max across spatial dimensions
#             min_val = imgs.amin(dim=(2, 3), keepdim=True)
#             max_val = imgs.amax(dim=(2, 3), keepdim=True)
            
#             print(f"  Min values per image (first few): {min_val.squeeze().cpu().flatten()[:4]}")
#             print(f"  Max values per image (first few): {max_val.squeeze().cpu().flatten()[:4]}")
            
#             # Check for zero range (potential division by zero)
#             range_val = max_val - min_val
#             zero_range_mask = (range_val == 0)
#             zero_range_count = zero_range_mask.sum()
            
#             if zero_range_count > 0:
#                 print(f"  ⚠️  {zero_range_count} images have zero range (min == max)!")
#                 print(f"      This will cause division by zero!")
#                 # Show which images have zero range
#                 for i, has_zero_range in enumerate(zero_range_mask.squeeze()):
#                     if has_zero_range.any():
#                         print(f"        Image {i} has zero range in some channels")
                
#                 # Add small epsilon to prevent division by zero
#                 range_val = torch.clamp(range_val, min=1e-8)
#                 print(f"  Added epsilon to prevent division by zero")
            
#             # Apply normalization: shift to [0,1] then to [-1,1]
#             imgs_normalized = 2 * (imgs - min_val) / range_val - 1
            
#             print(f"  Normalizer output range: [{imgs_normalized.min():.6f}, {imgs_normalized.max():.6f}]")
#             print(f"  Normalizer output mean: {imgs_normalized.mean():.6f}")
#             print(f"  Normalizer output std: {imgs_normalized.std():.6f}")
            
#             return imgs_normalized
    
#     try:
#         normalizer = NormalizerInterf()
#         normalized_images = normalizer(noised_images)
        
#         print(f"✅ Normalizer applied successfully")
        
#         # Check for problematic values
#         has_nan = torch.isnan(normalized_images).any()
#         has_inf = torch.isinf(normalized_images).any()
#         print(f"Contains NaN: {has_nan}")
#         print(f"Contains Inf: {has_inf}")
        
#         if has_nan or has_inf:
#             print("❌ Normalizer produces NaN/Inf values!")
#             # Find where NaN/Inf occur
#             if has_nan:
#                 nan_locations = torch.isnan(normalized_images).nonzero()
#                 print(f"  NaN locations (first 5): {nan_locations[:5]}")
#             if has_inf:
#                 inf_locations = torch.isinf(normalized_images).nonzero()
#                 print(f"  Inf locations (first 5): {inf_locations[:5]}")
#             return None
        
#     except Exception as e:
#         print(f"❌ Error applying normalizer: {e}")
#         import traceback
#         traceback.print_exc()
#         return None
    
#     # 4. Simulate BatchNorm behavior
#     print(f"\n📊 Step 4: Simulating BatchNorm Behavior")
#     print("-" * 40)
    
#     # Calculate what BatchNorm would see
#     # BatchNorm calculates mean and var across (N, H, W) for each channel
#     batch_mean = normalized_images.mean(dim=(0, 2, 3))  # Shape: (C,)
#     batch_var = normalized_images.var(dim=(0, 2, 3), unbiased=False)  # Shape: (C,)
    
#     print(f"Batch statistics that BatchNorm would calculate:")
#     print(f"  Batch mean per channel: {batch_mean}")
#     print(f"  Batch var per channel: {batch_var}")
    
#     # Check if these would cause issues
#     extreme_mean_threshold = 5
#     extreme_var_threshold = 100
#     tiny_var_threshold = 1e-6
    
#     extreme_means = torch.abs(batch_mean) > extreme_mean_threshold
#     extreme_vars = batch_var > extreme_var_threshold
#     tiny_vars = batch_var < tiny_var_threshold
    
#     if extreme_means.any():
#         print(f"  ⚠️  {extreme_means.sum()} channels have extreme batch means (abs > {extreme_mean_threshold})")
    
#     if extreme_vars.any():
#         print(f"  ⚠️  {extreme_vars.sum()} channels have extreme batch variances (> {extreme_var_threshold})")
    
#     if tiny_vars.any():
#         print(f"  ⚠️  {tiny_vars.sum()} channels have tiny batch variances (< {tiny_var_threshold})")
    
#     # 5. Test with a fresh BatchNorm layer
#     print(f"\n📊 Step 5: Testing with Fresh BatchNorm Layer")
#     print("-" * 40)
    
#     try:
#         # Create a BatchNorm layer like in your model
#         bn = nn.BatchNorm2d(normalized_images.size(1)).to(device)
        
#         print(f"Fresh BatchNorm initial stats:")
#         print(f"  Running mean: {bn.running_mean}")
#         print(f"  Running var: {bn.running_var}")
#         print(f"  Momentum: {bn.momentum}")
        
#         # Pass data through BatchNorm in training mode
#         bn.train()
#         bn_output = bn(normalized_images)
        
#         print(f"After one forward pass:")
#         print(f"  Running mean: {bn.running_mean}")
#         print(f"  Running var: {bn.running_var}")
#         print(f"  Output range: [{bn_output.min():.6f}, {bn_output.max():.6f}]")
        
#         # Check if running stats are reasonable
#         if torch.abs(bn.running_mean).max() > 1:
#             print(f"  ⚠️  Running means already large after one pass!")
        
#         if bn.running_var.max() > 10:
#             print(f"  ⚠️  Running variances already large after one pass!")
        
#     except Exception as e:
#         print(f"❌ Error testing BatchNorm: {e}")
#         import traceback
#         traceback.print_exc()
    
#     # 6. Summary and recommendations
#     print(f"\n💡 SUMMARY AND RECOMMENDATIONS")
#     print("-" * 40)
    
#     print("✅ Data pipeline analysis completed!")
    
#     # Based on what we found, provide specific recommendations
#     recommendations = []
    
#     recommendations.extend([
#         "1. 🔧 Replace your normalizer with standard normalization:",
#         "   mean = imgs.mean(dim=(2,3), keepdim=True)",
#         "   std = imgs.std(dim=(2,3), keepdim=True) + 1e-8",
#         "   imgs = (imgs - mean) / std",
#         "",
#         "2. 🔧 Lower BatchNorm momentum significantly:",
#         "   for m in model.modules():",
#         "       if isinstance(m, nn.BatchNorm2d):",
#         "           m.momentum = 0.01  # Much lower than 0.1",
#         "",
#         "3. 🔧 Use SyncBatchNorm for distributed training:",
#         "   model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)",
#         "",
#         "4. 🔧 Consider clamping extreme values:",
#         "   imgs = torch.clamp(imgs, -10, 10)  # Before BatchNorm",
#     ])
    
#     for rec in recommendations:
#         print(rec)
    
#     return {
#         'raw_range': (raw_images.min().item(), raw_images.max().item()),
#         'noised_range': (noised_images.min().item(), noised_images.max().item()),
#         'normalized_range': (normalized_images.min().item(), normalized_images.max().item()),
#         'batch_mean': batch_mean.cpu().numpy(),
#         'batch_var': batch_var.cpu().numpy()
#     }

# # Run in distributed environment but use only rank 0
# def main():
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
#     if local_rank == 0:  # Only run on rank 0 to avoid duplicate output
#         try:
#             results = debug_your_data_pipeline()
#             if results:
#                 print(f"\n✅ Debugging completed successfully!")
#                 print(f"Key findings:")
#                 print(f"  Raw data range: {results['raw_range']}")
#                 print(f"  After noiser: {results['noised_range']}")
#                 print(f"  After normalizer: {results['normalized_range']}")
#                 print(f"  Batch means: {results['batch_mean']}")
#                 print(f"  Batch vars: {results['batch_var']}")
#             else:
#                 print(f"❌ Debugging failed - check the errors above")
                
#         except Exception as e:
#             print(f"❌ Error during debugging: {str(e)}")
#             import traceback
#             traceback.print_exc()
    
#     # Cleanup distributed if needed
#     if torch.distributed.is_initialized():
#         torch.distributed.barrier()

# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# """
# Simple BatchNorm diagnostics script for your trained model.
# Run this in your training environment to check for BatchNorm issues.
# """

# import torch
# import torch.nn as nn
# import os
# from pathlib import Path
# import json

# def diagnose_batchnorm_issues(classifier_name, stage_name="2_min_mass_10e9", checkpoint_name=None):
#     """
#     Simple BatchNorm diagnosis that just loads your model and tests basic issues.
#     """
#     print(f"🔍 Diagnosing BatchNorm issues for {classifier_name}")
#     print("=" * 60)
    
#     # Import your modules
#     from config import TRAINED_CLASSIFIERS_DIR
#     from deep_learning.NN_models import ResNet50
    
#     # 1. Load your trained model
#     model_path = Path(TRAINED_CLASSIFIERS_DIR) / classifier_name
#     stage_path = model_path / stage_name
    
#     if not stage_path.exists():
#         print(f"❌ Stage path not found: {stage_path}")
#         print(f"Available stages: {list(model_path.glob('*_min_mass*'))}")
#         return
    
#     # Find checkpoint
#     checkpoint_path = stage_path / "checkpoints"
#     if checkpoint_name is None:
#         checkpoints = list(checkpoint_path.glob("*.pth"))
#         if not checkpoints:
#             print(f"❌ No checkpoints found in {checkpoint_path}")
#             return
#         checkpoint_file = max(checkpoints, key=os.path.getctime)
#     else:
#         checkpoint_file = checkpoint_path / checkpoint_name
    
#     print(f"📁 Loading checkpoint: {checkpoint_file.name}")
    
#     # Load the model
#     model = ResNet50(num_classes=2)
#     checkpoint = torch.load(checkpoint_file, map_location='cpu')
#     model.load_state_dict(checkpoint['model_state_dict'])
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     print(f"✅ Model loaded on {device}")
    
#     # 2. Analyze BatchNorm layers
#     print("\n📊 BatchNorm Layer Analysis")
#     print("-" * 40)
    
#     bn_layers = []
#     bn_names = []
    
#     def find_bn_layers(module, prefix=''):
#         for name, child in module.named_children():
#             full_name = f"{prefix}.{name}" if prefix else name
#             if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
#                 bn_layers.append(child)
#                 bn_names.append(full_name)
#             else:
#                 find_bn_layers(child, full_name)
    
#     find_bn_layers(model)
#     print(f"Found {len(bn_layers)} BatchNorm layers")
    
#     # Check key properties
#     issues_found = 0
    
#     for i, (layer, name) in enumerate(zip(bn_layers, bn_names)):
#         if i < 10:  # Show details for first 10 layers
#             print(f"\n{name}:")
#             print(f"  Momentum: {layer.momentum}")
#             print(f"  Track running stats: {layer.track_running_stats}")
#             print(f"  Num features: {layer.num_features}")
            
#             # Check for potential issues
#             if layer.momentum > 0.1:
#                 print(f"  ⚠️  High momentum ({layer.momentum}) - may cause instability")
#                 issues_found += 1
            
#             if not layer.track_running_stats:
#                 print(f"  ⚠️  Not tracking running stats!")
#                 issues_found += 1
            
#             # Check running statistics
#             running_mean_range = (layer.running_mean.min().item(), layer.running_mean.max().item())
#             running_var_range = (layer.running_var.min().item(), layer.running_var.max().item())
            
#             print(f"  Running mean range: [{running_mean_range[0]:.4f}, {running_mean_range[1]:.4f}]")
#             print(f"  Running var range: [{running_var_range[0]:.4f}, {running_var_range[1]:.4f}]")
            
#             # Check for extreme values
#             if abs(running_mean_range[0]) > 10 or abs(running_mean_range[1]) > 10:
#                 print(f"  ⚠️  Extreme running means!")
#                 issues_found += 1
            
#             if running_var_range[0] < 0.001 or running_var_range[1] > 100:
#                 print(f"  ⚠️  Extreme running variances!")
#                 issues_found += 1
    
#     # 3. Create dummy data to test train vs eval behavior
#     print(f"\n📊 Train vs Eval Mode Test")
#     print("-" * 40)
    
#     # Create dummy input that matches your data shape
#     dummy_input = torch.randn(16, 3, 80, 80).to(device)  # Batch of 16, 3 channels, 80x80
    
#     # Test in eval mode
#     model.eval()
#     with torch.no_grad():
#         eval_output = model(dummy_input)
    
#     # Test in train mode
#     model.train()
#     with torch.no_grad():
#         train_output = model(dummy_input)
    
#     # Compare outputs
#     output_diff = torch.abs(eval_output - train_output).mean()
#     max_diff = torch.abs(eval_output - train_output).max()
    
#     print(f"Mean output difference (train vs eval): {output_diff:.6f}")
#     print(f"Max output difference (train vs eval): {max_diff:.6f}")
    
#     if output_diff > 1e-3:
#         print("⚠️  WARNING: Large difference between train/eval modes!")
#         print("   This is likely causing your train/test loss divergence!")
#         issues_found += 1
#     else:
#         print("✅ Train/eval outputs are consistent")
    
#     # 4. Test multiple random inputs to see consistency
#     print(f"\n📊 Output Consistency Test")
#     print("-" * 40)
    
#     model.eval()
#     outputs = []
#     for i in range(5):
#         dummy_input = torch.randn(8, 3, 80, 80).to(device)
#         with torch.no_grad():
#             output = model(dummy_input)
#         outputs.append(output.mean().item())
    
#     output_std = torch.tensor(outputs).std().item()
#     print(f"Output mean std across random inputs: {output_std:.6f}")
    
#     if output_std > 1.0:
#         print("⚠️  High variance in outputs - model may be unstable")
#         issues_found += 1
#     else:
#         print("✅ Model outputs are stable")
    
#     # 5. Summary and recommendations
#     print(f"\n💡 DIAGNOSIS SUMMARY")
#     print("-" * 40)
    
#     if issues_found == 0:
#         print("✅ No obvious BatchNorm issues found!")
#         print("   The train/test divergence may be due to other factors:")
#         print("   - Data distribution differences")
#         print("   - Overfitting")
#         print("   - Learning rate schedule")
#     else:
#         print(f"⚠️  Found {issues_found} potential BatchNorm issues!")
        
#     print(f"\n🔧 RECOMMENDATIONS")
#     print("-" * 40)
    
#     recommendations = [
#         "1. Use SyncBatchNorm for distributed training:",
#         "   model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)",
#         "",
#         "2. Lower BatchNorm momentum in your model:",
#         "   for m in model.modules():",
#         "       if isinstance(m, nn.BatchNorm2d):",
#         "           m.momentum = 0.01  # Much lower than default 0.1",
#         "",
#         "3. Consider freezing BatchNorm during fine-tuning:",
#         "   for m in model.modules():",
#         "       if isinstance(m, nn.BatchNorm2d):",
#         "           m.eval()  # Freeze running stats",
#         "",
#         f"4. Your current momentum is {bn_layers[0].momentum} - try 0.01-0.05",
#         "",
#         "5. Monitor BatchNorm stats during training with wandb"
#     ]
    
#     for rec in recommendations:
#         print(rec)
    
#     return {
#         'issues_found': issues_found,
#         'output_difference': output_diff.item(),
#         'bn_layer_count': len(bn_layers),
#         'momentum': bn_layers[0].momentum if bn_layers else None
#     }

# # Run the diagnosis
# if __name__ == "__main__":
#     # Change these to match your setup
#     CLASSIFIER_NAME = "RUN1clusterEpochPatienceNight"
#     STAGE_NAME = "2_min_mass_10e9"  # The stage you mentioned in your error
    
#     try:
#         results = diagnose_batchnorm_issues(
#             classifier_name=CLASSIFIER_NAME,
#             stage_name=STAGE_NAME
#         )
#         print(f"\n✅ Diagnosis completed! Found {results['issues_found']} issues.")
        
#     except Exception as e:
#         print(f"❌ Error during diagnosis: {str(e)}")
#         import traceback
#         traceback.print_exc()