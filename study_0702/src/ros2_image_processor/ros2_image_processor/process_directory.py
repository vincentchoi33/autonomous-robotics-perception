import torch
import torchvision
from torchvision.io import read_image, ImageReadMode
from mmengine.model import revert_sync_batchnorm
from mmengine import Config
import vltseg
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
import os
import glob
import numpy as np
from PIL import Image
import cv2

# Register all modules
register_all_modules(True)

def rsz(input, l):
    transform = torchvision.transforms.Resize((l, l))
    return transform(input)

# Cityscapes color palette (RGB values)
CITYSCAPES_PALETTE = [
    [128, 64, 128],   # 0: road
    [244, 35, 232],   # 1: sidewalk
    [70, 70, 70],     # 2: building
    [102, 102, 156],  # 3: wall
    [190, 153, 153],  # 4: fence
    [153, 153, 153],  # 5: pole
    [250, 170, 30],   # 6: traffic light
    [220, 220, 0],    # 7: traffic sign
    [107, 142, 35],   # 8: vegetation
    [152, 251, 152],  # 9: terrain
    [70, 130, 180],   # 10: sky
    [220, 20, 60],    # 11: person
    [255, 0, 0],      # 12: rider
    [0, 0, 142],      # 13: car
    [0, 0, 70],       # 14: truck
    [0, 60, 100],     # 15: bus
    [0, 80, 100],     # 16: train
    [0, 0, 230],      # 17: motorcycle
    [119, 11, 32],    # 18: bicycle
]

def process_image(model, image_path, output_dir, device):
    output_dir_color = "/data/choihy/study_0630/VLTSeg/results1/color"
    output_dir_grayscale = "/data/choihy/study_0630/VLTSeg/results1/grayscale"
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_grayscale, exist_ok=True)
    """Process a single image and save the segmentation result"""
    try:
        # Load and resize image to match checkpoint size (1022x1022)
        image = read_image(image_path, mode=ImageReadMode.RGB).to(torch.float32)
        image = rsz(image, 1022).to(device).unsqueeze(0)
        
        # Normalize image (ImageNet normalization)
        mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(device)
        image = (image - mean) / std
        
        # Get prediction
        with torch.no_grad():
            result = model.predict(image)
            pred_sem_seg = result[0].pred_sem_seg.data.cpu().numpy()[0]  # Shape: (H, W)
        
        # Save grayscale result (class IDs)
        filename = os.path.basename(image_path)
        output_path_grayscale = os.path.join(output_dir_grayscale, filename.replace('.png', '_seg_grayscale.png'))
        pred_image_grayscale = Image.fromarray(pred_sem_seg.astype(np.uint8))
        pred_image_grayscale.save(output_path_grayscale)
        
        # Convert to color using Cityscapes palette
        pred_color = np.zeros((pred_sem_seg.shape[0], pred_sem_seg.shape[1], 3), dtype=np.uint8)
        for class_id in range(19):
            mask = pred_sem_seg == class_id
            pred_color[mask] = CITYSCAPES_PALETTE[class_id]
        
        # Save color result
        output_path_color = os.path.join(output_dir_color, filename.replace('.png', '_seg_color.png'))
        pred_image_color = Image.fromarray(pred_color)
        pred_image_color.save(output_path_color)
        
        print(f"Processed: {filename} -> {output_path_grayscale}, {output_path_color}")
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def main():
    config = "/data/choihy/study_0630/VLTSeg/configs/custom_inference.py"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Input and output directories
    # input_dir = "/data/choihy/study_0630/data/cropped"
    input_dir = "/data/choihy/study_0702/visualization_output"
    output_dir = "/data/choihy/study_0630/VLTSeg/results1"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    
    # Load model
    print("Loading model...")
    cfg = Config.fromfile(config)
    model = MODELS.build(cfg.model)
    model.to(device)
    
    # Load checkpoint
    checkpoint_path = '/data/choihy/study_0630/VLTSeg/vltseg_checkpoint_mapillary+cityscapes_1.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    if device == 'cpu':
        model = revert_sync_batchnorm(model)
    
    model.eval()
    print("Model loaded successfully!")
    
    # Get all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    success_count = 0
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        if process_image(model, image_path, output_dir, device):
            success_count += 1
    
    print(f"\nProcessing complete! {success_count}/{len(image_files)} images processed successfully.")
    print(f"Results saved to: {output_dir}")
    print("Each image generates two files:")
    print("- *_seg_grayscale.png: Grayscale class IDs (0-18)")
    print("- *_seg_color.png: Color visualization using Cityscapes palette")

if __name__ == "__main__":
    main() 