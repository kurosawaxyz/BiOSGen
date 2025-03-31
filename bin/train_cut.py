import torch 
import matplotlib.pyplot as plt

from osgen.dataloader import PatchDataLoader
from cut.model import i2iTranslationModel


import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

import os


if __name__ == "__main__":

    # Load argparser
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Argparse for training process.")

    # Add arguments
    parser.add_argument("--config_path", type=str, help="Configuration path", required=True)
    parser.add_argument("--style_path", type=str, help="Style tumor path", required=True)
    parser.add_argument("--original_path", type=str, help="Original tumor path", required=True)

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    device = cfg.device
    # learning_rate = cfg.learning_rate

    tissue_mask_params = cfg.Image.tissue_mask_params
    patch_extraction_params = cfg.Image.patch_extraction_params

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

    model = i2iTranslationModel(cfg)
    model.netG.train()
    if hasattr(model, 'netD'):
        model.netD.train()

    D_loss, G_loss = [], []
    num_epochs = cfg.train.num_epochs

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for i in tqdm(range(cfg.train.batch_size)):
            real_A, real_B = next(iter(data_loader.train_dataset))

            input_dict = {
                'src': real_A.unsqueeze(0).to(device),
                'dst': real_B.to(device)
            }
            model.data_dependent_initialize(input_dict)
            model.set_input(input_dict)
            model.optimize_parameters()

            D_loss.append(model.loss_D.item())
            G_loss.append(model.loss_G.item())
            
        print(f"Epoch {epoch + 1}/{num_epochs} - Step {i + 1}/{10}: D_loss: {model.loss_D:.4f}, G_loss: {model.loss_G:.4f}")

    # Plot losses
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(D_loss, label='Discriminator Loss')
    ax[1].plot(G_loss, label='Generator Loss')
    ax[0].set_title("Discriminator Loss")
    ax[1].set_title("Generator Loss")
    ax[0].set_xlabel("Step")
    ax[1].set_xlabel("Step")
    ax[0].set_ylabel("Loss")
    ax[1].set_ylabel("Loss")
    ax[0].legend()
    ax[1].legend()

    if not os.path.exists("train_results"):
        os.mkdir("train_results")
    plt.savefig("train_results/res-losses.png")
    plt.show()

    # Generate images
    with torch.no_grad():
        model.forward()
        output = model.fake_B
    output_image = output[0].squeeze(0).permute(1,2,0).cpu().detach().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(real_A.permute(1, 2, 0).detach().numpy().astype(int))
    ax[0].set_title("Real Image")
    ax[1].imshow(real_B[0].permute(1, 2, 0).detach().numpy().astype(int))
    ax[1].set_title("Fake Image")
    ax[2].imshow(output[0].permute(1, 2, 0).detach().numpy())
    ax[2].set_title("Generated Image")
    # Transpose the image to [height, width, channels]
    plt.savefig("train_results/res-image.png")
    plt.show()


    # Terminal execution
    # python -m bin.train_cut --config_path configs/cut_config.yml --style_path demo/img/A6_TMA_15_02_IVB_NKX.png --original_path demo/img/A4_TMA_15_02_IVB_HE.png