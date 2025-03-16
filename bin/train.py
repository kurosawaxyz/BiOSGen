
import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from tqdm import tqdm
import numpy as np

import os

from osgen.pipeline import StyleTransferPipeline
from osgen.vit import extract_style_embedding
from osgen.dataloader import PatchDataLoader
from osgen.loss import total_loss

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from PIL import Image

from omegaconf import OmegaConf
import numpy as np

import argparse


if __name__ == "__main__":

    # Load argparser
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="This script demonstrates argparse usage.")

    # Add arguments
    parser.add_argument("--config_path", type=str, help="Configuration path", required=True)
    parser.add_argument("--style_path", type=str, help="Style tumor path", required=True)
    parser.add_argument("--original_path", type=str, help="Original tumor path", required=True)

    args = parser.parse_args()



    # Load config
    cfg = OmegaConf.load(args.config_path)
    device = cfg.device
    learning_rate = cfg.learning_rate

    # Initialize model
    model = StyleTransferPipeline(device=device)
    model = model.to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")

    # Initialize optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.optimizer.lr,
        weight_decay=1e-8
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2,
        verbose=True
    )

    # Track losses
    losses = []
    val_losses = []

    # Load main configs
    tissue_mask_params = cfg.Image.tissue_mask_params
    patch_extraction_params = cfg.Image.patch_extraction_params

    # Load image and style
    Image.MAX_IMAGE_PIXELS = 100000000000

    IMAGE_PATH = args.original_path
    STYLE_PATH = args.style_path

    data_loader = PatchDataLoader(
            path_src=IMAGE_PATH,
            path_dst=STYLE_PATH,
            tissue_mask_params=tissue_mask_params,
            patch_extraction_params=patch_extraction_params,
            batch_size=3,
            val_ratio=0.15,
            test_ratio=0.15
        )

    # Print information about each split
    print(f"Train dataset size: {len(data_loader.train_dataset)}")
    print(f"Validation dataset size: {len(data_loader.val_dataset)}")
    print(f"Test dataset size: {len(data_loader.test_dataset)}")



    # SOLUTION 1: Improved training loop with gradient stability features

    # Initialize gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler() if device == 'cuda' else None

    # Better optimizer with weight decay
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], 
                            lr=cfg.optimizer.lr,  # Start with a lower learning rate 
                            weight_decay=1e-6)  # Add weight decay to prevent overfitting

    # Add learning rate scheduler to reduce LR when loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Loss scaling factors with more balanced values
    alpha_structure = 1e-4
    alpha_content = 1e-3
    alpha_style = 1e-5
    alpha_ca = 1e-4
    gradient_clip_val = 1.0  # More aggressive gradient clipping

    # Per-parameter gradient clipping threshold
    grad_clip_thresholds = {}

    losses = []
    running_avg_loss = 0

    num_epochs = cfg.num_epochs

    # Output directory for evaluation results
    output_dir = 'train_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        start_time = time.time()
        
        # Initialize batch counters
        nan_detected = False
        
        for batch_idx in tqdm(range(cfg.batch_size)):  # Keep your batch range limit for debugging
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


    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.plot(losses, label='Loss')
    plt.plot(avg_epoch_loss, label='Avg. Epoch Loss', color='red', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(f"{output_dir}/losses.png")

    # Terminal execution
    # python -m bin.train --config_path /Users/hoangthuyduongvu/Desktop/BiOSGen/configs/train_config.yml --style_path /Users/hoangthuyduongvu/Desktop/BiOSGen/data/NKX3/A3_TMA_15_02_IIB_NKX.png --original_path /Users/hoangthuyduongvu/Desktop/BiOSGen/data/HE/A3_TMA_15_02_IVB_HE.png