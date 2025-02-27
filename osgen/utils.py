import torch 
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForCausalLM 

# moondream
def load_md():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
    ).to(device)
    return model

def describe_img(image_path):
    img = Image.open(image_path)
    enc_image = model.encode_image(img)
    print(model.query(enc_image, "Describe this image."))