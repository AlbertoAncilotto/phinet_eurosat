import os
import numpy as np
import rasterio
from tqdm import tqdm
import cv2
from run_inference import load_model,preprocess_image, infer

PATCH_SIZE = 64
DISPLAY_BANDS = ['B02_20m', 'B03_20m', 'B04_20m']  # Bands to display as R,G,B
MODEL = load_model('checkpoints\\a050_zero_7_9_10.pt', 13, 'cuda')

class_colormap = np.array([
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.2, 0.8, 0.0],
                [0.0, 1.0, 1.0],
                [0.5, 0.5, 1.0],
                [0.0, 0.5, 1.0],
                [1.0, 1.0, 0.0],])

def classify(input_patch):
    image_tensor = preprocess_image(input_patch, zero_channels=[7,9,10], input_channels=13)
    probabilities, predicted_class = infer(MODEL, image_tensor, 'cuda')
    return int(predicted_class), probabilities

def read_sentinel_bands(tile_path, bands):
    images = []
    for band in tqdm(bands):
        band_path = tile_path + f"{band}.jp2"
        with rasterio.open(band_path) as src:
            images.append(src.read(1))
    return np.stack(images, axis=0)

def display_image_cv2(tile_path, band_path):
    images = []
    for band in band_path:
        band_path = tile_path + f"{band}.jp2"
        with rasterio.open(band_path) as src:
            images.append(src.read(1))
    image = np.stack(images, axis=2) / 5000.
    print(image.shape, np.min(image), np.max(image), np.mean(image))
    image = np.clip(image, 0, 1.0)
    image_res = cv2.resize(image, (600,600))
    cv2.imshow('Full Image', image_res)
    cv2.waitKey(1)
    return image_res


def extract_and_classify_patches(images, patch_size, display_image, oversample=2):
    _, rows, cols = images.shape
    disp_rows, disp_cols, _ = display_image.shape
    display_image = (cv2.cvtColor(cv2.cvtColor(display_image.astype(np.float32), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)*0.5)

    center_row, center_col = rows // 2, cols // 2
    directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # Right, Down, Left, Up
    dir_idx = 0
    steps = 1
    total_steps = 0
    i, j = center_row, center_col
    step_size=patch_size//oversample

    while total_steps < rows * cols:
        for _ in range(2):  # Two turns per spiral layer
            for _ in range(steps):
                if 0 <= i < rows and 0 <= j < cols:
                    patch = images[:, i:i+patch_size, j:j+patch_size]
                    
                    if patch.shape[1] == patch_size and patch.shape[2] == patch_size:
                        patch = np.transpose(patch, (1, 2, 0))
                        _, probabilities = classify(patch)
                        color = (probabilities @ class_colormap).squeeze()
                        
                        x, y, dps = int(i*disp_rows/rows),  int(j*disp_cols/cols), int(step_size*disp_rows/rows)
                        display_image[x:x+dps+1, y:y+dps+1] = color

                        cv2.imshow('Classified Image', display_image)
                        cv2.waitKey(1)

                total_steps += 1
                i += directions[dir_idx][0] * step_size
                j += directions[dir_idx][1] * step_size
            
            dir_idx = (dir_idx + 1) % 4
        steps += 1 


def process_sentinel_tile(tile_path, output_dir, bands):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    disp_img = display_image_cv2(tile_path, DISPLAY_BANDS)
    print('loading data')
    images = read_sentinel_bands(tile_path, bands)
    extract_and_classify_patches(images, PATCH_SIZE, disp_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    tile_path = "IMG_DATA\\R20m\\T32TQR_20240716T100601_"
    output_dir = "output/patches"
    
    bands = [
        'B01_20m',
        'B02_20m',
        'B03_20m', 
        'B04_20m',  
        'B05_20m',
        'B06_20m',
        'B07_20m',
        'B8A_20m', # in model a050_zero_7_9_10, bands 7, 9 and 10 will be zeroed.
        'B8A_20m',
        'B8A_20m', # in model a050_zero_7_9_10, bands 7, 9 and 10 will be zeroed.
        'B11_20m', # in model a050_zero_7_9_10, bands 7, 9 and 10 will be zeroed.
        'B11_20m',
        'B12_20m',
    ]
    
    process_sentinel_tile(tile_path, output_dir, bands)
