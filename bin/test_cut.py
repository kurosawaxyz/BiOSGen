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

    # Load model
    model.netG.load_state_dict(torch.load("checkpoints/cut/netG.pt"))
    model.netD.load_state_dict(torch.load("checkpoints/cut/netD.pt"))
    
    # Test the model
    model.netG.eval()
    if hasattr(model, 'netD'):
        model.netD.eval()

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

            with torch.no_grad():
                model.forward()
                output = model.fake_B

            # Save the generated image
            output_image = output[0].squeeze(0).permute(1,2,0).cpu().detach().numpy()
            fig, ax = plt.subplots(1, 3, figsize=(10, 5))
            ax[0].imshow(real_A.permute(1, 2, 0).detach().numpy().astype(int))
            ax[0].set_title("Real Image")
            ax[1].imshow(real_B[0].permute(1, 2, 0).detach().numpy().astype(int))
            ax[1].set_title("Fake Image")
            ax[2].imshow(output[0].permute(1, 2, 0).detach().numpy())
            ax[2].set_title("Generated Image")
            # Transpose the image to [height, width, channels]
            plt.savefig(f"archive/cut_test/cut_res-image_{num_epochs}_{i}.png")


# Command to run the script:
# python -m bin.test_cut --config_path configs/cut_config.yml --style_path demo/img/A6_TMA_15_02_IVB_NKX.png --original_path demo/img/A4_TMA_15_02_IVB_HE.png