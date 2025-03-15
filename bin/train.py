
import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import tqdm
import numpy as np

import os
from preprocess.tissue_mask import GaussianTissueMask
from preprocess.utils import describe_img, read_image, get_tissue_mask, get_image_patches, resize_patch, normalize_patch, convert_patch_to_tensor

from osgen.loss import structure_preservation_loss, color_alignment_loss, content_loss, style_loss, total_loss
from osgen.pipeline import StyleTransferPipeline
from osgen.vit import extract_style_embedding
from osgen.vae import VAEncoder, VAEDecoder, ConditionedVAEncoder
from osgen.nn import *
from osgen.unet import UNetModel
from osgen.dataloader import PatchDataLoader

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from PIL import Image

from omegaconf import OmegaConf
import numpy as np

# Load config
cfg = OmegaConf.load("configs/config.yml")

# SOLUTION 1: Improved training loop with gradient stability features

# Initialize gradient scaler for mixed precision training
scaler = torch.amp.GradScaler() if device == 'cuda' else None

# Better optimizer with weight decay
optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], 
                        lr=1e-4,  # Start with a lower learning rate 
                        weight_decay=1e-6)  # Add weight decay to prevent overfitting

# Add learning rate scheduler to reduce LR when loss plateaus
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Loss scaling factors with more balanced values
alpha_structure = 1e-4
alpha_content = 1e-3
alpha_style = 1e-5
alpha_ca = 1e-4
gradient_clip_val = 1.0  # More aggressive gradient clipping
learning_rate = 1e-5            # Lower initial learning rate learns more stable features

# Per-parameter gradient clipping threshold
grad_clip_thresholds = {}

losses = []
running_avg_loss = 0

num_epochs = 10

for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0
    
    print(f"Epoch {epoch + 1}/{num_epochs}")
    start_time = time.time()
    
    # Initialize batch counters
    nan_detected = False
    
    for batch_idx in tqdm.tqdm(range(5)):  # Keep your batch range limit for debugging
        # Get batch
        # batch, style = data_loader.test_dataset[batch_idx]
        batch, style = next(iter(data_loader.train_dataset))
        batch = batch.unsqueeze(0).to(device)
        style = style.to(device)
        
        # Extract style embedding
        style_emb = extract_style_embedding(style[0].permute(1,2,0).numpy().astype(np.uint8), device=device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # SOLUTION 4: Dynamic timesteps instead of fixed ones
        # Create random timesteps that vary each batch
        timesteps = torch.randint(0, 1000, (4,)).to(device)
        
        # SOLUTION 3: Gradient scaling with mixed precision
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # Forward pass
                out = model(batch, style_emb, timesteps)
                loss = total_loss(
                    original_image=batch, 
                    generated_image=out[0].unsqueeze(0),
                    lambda_structure=alpha_structure,
                    lambda_content=alpha_content,
                    lambda_style=alpha_style,
                    lambda_color=alpha_ca
                )
            
            # Scale loss and perform backward pass
            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
        else:
            # Forward pass (without mixed precision)
            out = model(batch, style_emb, timesteps)
            loss = total_loss(
                original_image=batch, 
                generated_image=out[0].unsqueeze(0),
                lambda_structure=alpha_structure,
                lambda_content=alpha_content,
                lambda_style=alpha_style,
                lambda_color=alpha_ca
            )
            loss.backward()
            # print("backward")
        

        # save_fig
        if batch_idx % 10 == 0:
            if not os.path.exists('train_results'):
                os.makedirs('train_results')
            fig, ax = plt.subplots(1, 3, figsize=(10, 5))
            ax[0].imshow(batch[0].permute(1, 2, 0).detach().numpy().astype(np.uint8))
            ax[0].set_title("Original Image")
            ax[0].axis("off")
            ax[1].imshow(out[0].detach().permute(1,2,0).numpy())            # Attention
            ax[1].set_title("Generated Image")
            ax[1].axis("off")
            ax[2].imshow(style[0].permute(1, 2, 0).detach().numpy().astype(np.uint8))
            ax[2].set_title("Style Image")
            ax[2].axis("off")
            plt.savefig(f"train_results/epoch_{epoch}_batch_{batch_idx}.png")

        # SOLUTION 4: Advanced gradient clipping
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            gradient_clip_val
        )
        
        # SOLUTION 5: Gradient monitoring and anomaly detection
        # Check for NaN gradients or loss
        if torch.isnan(loss).any():
            print(f"NaN detected in loss at batch {batch_idx}")
            nan_detected = True
            break
            
        # SOLUTION 2: Skip updates with NaN gradients
        nan_grads = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN detected in gradients of {name}")
                nan_grads = True
                
                # Store threshold for adaptive clipping
                if name not in grad_clip_thresholds:
                    grad_clip_thresholds[name] = 1.0
                else:
                    # Reduce threshold for problematic parameters
                    grad_clip_thresholds[name] *= 0.9
                break
                
        if nan_grads:
            continue  # Skip this batch
        
        # Update parameters
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # Update statistics
        losses.append(loss.item())
        epoch_loss += loss.item()
        num_batches += 1
        
        # Compute running average loss for the scheduler
        if running_avg_loss == 0:
            running_avg_loss = loss.item()
        else:
            running_avg_loss = 0.9 * running_avg_loss + 0.1 * loss.item()
            
        # Print loss every few batches
        if batch_idx % 5 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.6f}, Avg: {running_avg_loss:.6f}")
            
    if nan_detected:
        print("NaN values detected, attempting to recover...")
        # Load previous checkpoint or reinitialize problematic parameters
        continue
        
    # Skip epoch update if no batches were processed
    if num_batches == 0:
        print("No batches processed in this epoch. Skipping.")
        continue
    
    avg_epoch_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1} completed in {time.time() - start_time:.2f} seconds")
    print(f"Epoch {epoch+1} loss: {avg_epoch_loss:.6f}")
    
    # Update learning rate based on loss
    scheduler.step(avg_epoch_loss)
    
    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, f'checkpoints/checkpoint_epoch_{epoch+1}.pt')
