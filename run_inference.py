
import os
import random
import torch
import torch.nn as nn
import tifffile
import numpy as np
import argparse
from micromind.networks import PhiNet
from torchvision import transforms, models

def build_phinet(input_channels):
    model = PhiNet(
        (input_channels, 64, 64),
        alpha=0.5,
        beta=0.75,
        t_zero=5,
        num_layers=8,
        h_swish=False,
        squeeze_excite=True,
        include_top=True,
        num_classes=1000,
        divisor=8,
        compatibility=False
    )
    
    model = nn.Sequential(
        model,
        nn.ReLU(),
        nn.Linear(1000, 10, bias=True),
    )
    return model

def build_resnet(input_channels):
    resnet = models.resnet50(pretrained=False)
    resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, 10)
    return resnet


def load_model(checkpoint_path, input_channels, device):
    model = build_resnet(input_channels) if 'resnet' in checkpoint_path else build_phinet(input_channels)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    model = model.to(device).half()
    return model

def preprocess_image(input_img, zero_channels, input_channels):
    image = tifffile.imread(input_img) if type(input_img) == str else input_img
    image = torch.tensor((image / 7500).astype(np.float32))
    image = image.permute(2, 0, 1)
    
    transform = transforms.Normalize(mean=[0.5] * input_channels, std=[0.225] * input_channels)
    image = transform(image)
    # print('image: min', torch.min(image), 'max', torch.max(image), 'mean', torch.mean(image), 'std', torch.std(image))
    
    for ch in zero_channels:
        image[ch, :, :] = 0
    
    return image.unsqueeze(0)  # Add batch dimension

@torch.inference_mode()
def infer(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device).half()
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    return probabilities.cpu().numpy(), predicted_class.cpu().numpy()

def main(input_path, checkpoint_path, zero_channels, input_channels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, input_channels, device)

    directories = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    
    for directory in directories:
        print(f"Processing directory: {directory}")
        
        # Get list of images in the current directory
        images_path = os.path.join(input_path, directory)
        image_files = os.listdir(images_path)
        
        # Choose 10 random images if there are more than 10 available
        random.shuffle(image_files)
        num_images_to_process = min(10, len(image_files))
        selected_images = image_files[:num_images_to_process]
        
        for image_file in selected_images:
            image_path = os.path.join(images_path, image_file)
            
            image_tensor = preprocess_image(image_path, zero_channels, input_channels)
            probabilities, predicted_class = infer(model, image_tensor, device)
            
            print(f"Image: {image_file}")
            print(f"Predicted Class: {predicted_class[0]}")
            print(f"Predicted probability: {np.max(probabilities)}")
        print("-----------------------")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference script for EuroSat PhiNet.')
    parser.add_argument('--image_path', type=str, default='output/', help='Path to the TIFF image.')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/a050_zero_7_9_10.pt', help='Path to the model checkpoint.')
    parser.add_argument('--zero_channels', type=int, nargs='+', default=[7, 9, 10], help='List of zero channels.')
    parser.add_argument('--input_channels', type=int, default=13, help='Number of input channels.')
    
    args = parser.parse_args()
    main(args.image_path, args.checkpoint_path, args.zero_channels, args.input_channels)
