import torch
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .model import i2iTranslationModel
from .utils import *

# Load configuration
CONFIG_PATH = "/Users/hoangthuyduongvu/Desktop/tumor-augmentation/configs/config.yml"
DATA_CSV_PATH = "/Users/hoangthuyduongvu/Desktop/tumor-augmentation/cut/data.csv"

def train():

    print("Training...")
    args = load_config(CONFIG_PATH)
    data = load_data(DATA_CSV_PATH, args.train.batch_size)
    print("Data loaded!")
    
    device = args.device if torch.cuda.is_available() else "cpu"
    model = i2iTranslationModel(args)#.to(device)
    model.netG.train()
    if hasattr(model, 'netD'):
        model.netD.train()

    D_loss, G_loss = [], []
    num_epochs = 20

    print("Looping...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for i in range(len(data)):
            real_img = preprocess_image(data["real"][0], device)
            fake_img = preprocess_image(data["fake"][0], device)
            print(real_img.shape)
            print(fake_img.shape)
            
            input_dict = {'src': real_img, 'dst': fake_img}
            model.data_dependent_initialize(input_dict)
            model.set_input(input_dict)
            model.optimize_parameters()
            
            D_loss.append(model.loss_D.item())
            G_loss.append(model.loss_G.item())
            print(f"Epoch {epoch + 1}/{num_epochs} - Step {i + 1}/{len(data)}: D_loss: {model.loss_D:.4f}, G_loss: {model.loss_G:.4f}")
    
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
    plt.savefig("assets/res-losses.png")
    plt.show()

    # Save model
    torch.save(model.netG.state_dict(), "checkpoints/netG.pth")
    if hasattr(model, 'netD'):
        torch.save(model.netD.state_dict(), "checkpoints/netD.pth")
    
    print("Model saved!")
    


    # Generate images
    with torch.no_grad():
        model.forward()
        output = model.fake_B
    output_image = output[0].squeeze(0).permute(1,2,0).cpu().detach().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(real_img[0].squeeze(0).permute(1,2,0).detach().int(), cmap='gray')
    ax[0].set_title("Real Image")
    ax[1].imshow(fake_img[0].squeeze(0).permute(1,2,0).detach().int(), cmap='gray')
    ax[1].set_title("Fake Image")
    ax[2].imshow(output_image, cmap='gray')
    ax[2].set_title("Generated Image")
    # Transpose the image to [height, width, channels]
    plt.savefig("assets/res-image.png")
    plt.show()

    

if __name__ == "__main__":
    train()
