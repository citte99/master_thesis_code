#!/usr/bin/env python3

#!/usr/bin/env python3
"""
Fast computation of global dataset statistics for normalization.
Designed to work with your custom dataset that uses get_batch().
"""

import torch
import numpy as np
from pathlib import Path
import json

def compute_global_dataset_statistics(catalog_name="min_mass_10e11", samples_to_analyze=10000, batch_size=64):
    """
    Compute global mean and std for your dataset efficiently.
    
    Args:
        catalog_name: Name of your catalog
        samples_to_analyze: How many samples to use (more = more accurate)
        batch_size: Batch size for processing
    
    Returns:
        dict with global_mean, global_std, and other useful stats
    """
    
    print(f"🔍 COMPUTING GLOBAL DATASET STATISTICS")
    print("=" * 50)
    print(f"Catalog: {catalog_name}")
    print(f"Samples to analyze: {samples_to_analyze}")
    print(f"Batch size: {batch_size}")
    
    # Import your dataset
    from deep_learning.NN_datasets import NoNoiseDataset
    from noise_applicator.noisers.base_noiser import EuclidNoiserInterfPSF
    
    # Create dataset (same config as your training)
    print(f"\n📊 Creating dataset...")
    dataset = NoNoiseDataset(
        catalog_name,
        grid_pixel_side=80,
        grid_width_arcsec=8.0,
        broadcasting=True,
        samples_used=samples_to_analyze,
        upscaling=5,
    )
    
    # Set up noiser (same as training)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noiser = EuclidNoiserInterfPSF()
    noiser.set_device(device)
    
    print(f"✅ Dataset created with {len(dataset)} samples")
    print(f"Using device: {device}")
    
    # Online statistics computation (memory efficient)
    print(f"\n📊 Computing statistics...")
    
    n_samples = 0
    running_mean = 0.0
    running_m2 = 0.0  # For variance calculation
    
    # Additional statistics
    min_val = float('inf')
    max_val = float('-inf')
    
    # Histogram bins for distribution analysis
    hist_bins = torch.linspace(-1e-13, 1e-13, 1000)
    hist_counts = torch.zeros(len(hist_bins) - 1)
    
    num_batches = (samples_to_analyze + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        # Get batch indices
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, samples_to_analyze)
        indices = list(range(start_idx, end_idx))
        
        if len(indices) == 0:
            break
            
        try:
            # Get batch (same preprocessing as training)
            raw_images, _ = dataset.get_batch(indices)
            raw_images = raw_images.to(device)
            
            # Apply noiser (same as training)
            processed_images = noiser(raw_images)
            
            # Move to CPU for statistics (save GPU memory)
            processed_images = processed_images.cpu()
            
            # Flatten for statistics
            batch_values = processed_images.flatten()
            batch_size_actual = batch_values.numel()
            
            # Online mean and variance (Welford's algorithm)
            for value in batch_values:
                n_samples += 1
                delta = value - running_mean
                running_mean += delta / n_samples
                delta2 = value - running_mean
                running_m2 += delta * delta2
            
            # Update min/max
            batch_min = batch_values.min().item()
            batch_max = batch_values.max().item()
            min_val = min(min_val, batch_min)
            max_val = max(max_val, batch_max)
            
            # Update histogram
            hist_counts += torch.histc(batch_values, bins=len(hist_bins)-1, 
                                     min=hist_bins[0].item(), max=hist_bins[-1].item())
            
            # Progress
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                current_std = torch.sqrt(running_m2 / (n_samples - 1)) if n_samples > 1 else 0
                print(f"  Batch {batch_idx+1:4d}/{num_batches}: "
                      f"mean={running_mean:.2e}, std={current_std:.2e}, "
                      f"range=[{min_val:.2e}, {max_val:.2e}]")
            
        except Exception as e:
            print(f"  ❌ Error in batch {batch_idx}: {e}")
            continue
    
    # Final statistics
    if n_samples > 1:
        global_variance = running_m2 / (n_samples - 1)
        global_std = torch.sqrt(torch.tensor(global_variance)).item()
    else:
        global_variance = 0
        global_std = 1e-14  # Fallback
    
    print(f"\n✅ Statistics computed from {n_samples:,} pixels")
    
    # Results
    results = {
        'global_mean': running_mean,
        'global_std': global_std,
        'global_var': global_variance,
        'min_value': min_val,
        'max_value': max_val,
        'n_samples': n_samples,
        'catalog_name': catalog_name,
        'samples_analyzed': samples_to_analyze,
    }
    
    # Print results
    print(f"\n📊 FINAL RESULTS")
    print("-" * 30)
    print(f"Global mean: {results['global_mean']:.6e}")
    print(f"Global std:  {results['global_std']:.6e}")
    print(f"Global var:  {results['global_var']:.6e}")
    print(f"Min value:   {results['min_value']:.6e}")
    print(f"Max value:   {results['max_value']:.6e}")
    print(f"Range:       {results['max_value'] - results['min_value']:.6e}")
    print(f"Pixels analyzed: {results['n_samples']:,}")
    
    # Additional analysis
    print(f"\n📊 ANALYSIS")
    print("-" * 30)
    
    # Check if data is naturally zero-centered
    if abs(running_mean) < global_std * 0.1:
        print("✅ Data appears naturally zero-centered")
        print("   → Recommendation: Use only std normalization")
        recommended_normalizer = f"imgs / {global_std:.2e}"
    else:
        print("⚠️  Data has significant mean offset")
        print("   → Recommendation: Use mean and std normalization")
        recommended_normalizer = f"(imgs - {running_mean:.2e}) / {global_std:.2e}"
    
    # Check data scale
    typical_scale = global_std
    orders_of_magnitude = np.log10(typical_scale)
    print(f"Typical scale: {typical_scale:.2e} (10^{orders_of_magnitude:.1f})")
    
    if abs(orders_of_magnitude + 14) < 1:  # Close to 1e-14
        print("✅ Data scale matches expected astrophysical range")
    else:
        print("⚠️  Data scale different from expected 1e-14")
    
    # Distribution analysis
    print(f"\n📊 DISTRIBUTION ANALYSIS")
    print("-" * 30)
    
    # Find percentiles
    sorted_values = torch.sort(hist_counts)[0]
    total_counts = hist_counts.sum()
    
    if total_counts > 0:
        # Rough percentile estimation from histogram
        print("Value distribution:")
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        
        for p in percentiles:
            target_count = total_counts * p / 100
            # Find bin closest to this percentile
            cumsum = torch.cumsum(hist_counts, 0)
            bin_idx = torch.searchsorted(cumsum, target_count).item()
            bin_idx = min(bin_idx, len(hist_bins) - 2)
            value_estimate = hist_bins[bin_idx].item()
            print(f"  {p:2d}%: ~{value_estimate:.2e}")
    
    print(f"\n🔧 RECOMMENDED NORMALIZER CODE")
    print("-" * 30)
    print("class GlobalNormalizer(ImgLastProc):")
    print("    def __call__(self, imgs):")
    if abs(running_mean) < global_std * 0.1:
        print(f"        return imgs / {global_std:.6e}")
    else:
        print(f"        return (imgs - {running_mean:.6e}) / {global_std:.6e}")
    
    # Save results
    results_file = f"global_stats_{catalog_name}.json"
    with open(results_file, 'w') as f:
        # Convert to JSON-serializable format
        json_results = {k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v 
                       for k, v in results.items()}
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return results

def quick_stats_check(catalog_name="min_mass_10e11", n_batches=5):
    """
    Quick statistics check with just a few batches for rapid testing.
    """
    print(f"🚀 QUICK STATS CHECK")
    print("=" * 30)
    
    from deep_learning.NN_datasets import NoNoiseDataset
    from noise_applicator.noisers.base_noiser import EuclidNoiserInterfPSF
    
    dataset = NoNoiseDataset(
        catalog_name,
        grid_pixel_side=80,
        grid_width_arcsec=8.0,
        broadcasting=True,
        samples_used=500,  # Small for quick test
        upscaling=5,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noiser = EuclidNoiserInterfPSF()
    noiser.set_device(device)
    
    all_values = []
    
    for i in range(n_batches):
        indices = list(range(i*16, (i+1)*16))
        images, _ = dataset.get_batch(indices)
        images = images.to(device)
        processed = noiser(images)
        all_values.append(processed.cpu().flatten())
    
    all_values = torch.cat(all_values)
    
    mean = all_values.mean().item()
    std = all_values.std().item()
    
    print(f"Quick estimate from {len(all_values)} pixels:")
    print(f"  Mean: {mean:.6e}")
    print(f"  Std:  {std:.6e}")
    print(f"  Range: [{all_values.min().item():.2e}, {all_values.max().item():.2e}]")
    
    return mean, std

if __name__ == "__main__":
    # Quick check first
    print("Running quick check...")
    quick_mean, quick_std = quick_stats_check()
    
    print(f"\n" + "="*60)
    
    # Full analysis
    results = compute_global_dataset_statistics(
        catalog_name="min_mass_10e11",
        samples_to_analyze=5000,  # Adjust based on your needs
        batch_size=32
    )
    
    print(f"\n✅ Global statistics computation completed!")
# """
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