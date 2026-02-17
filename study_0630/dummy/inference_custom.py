import os
import torch
import torchvision
from torchvision.io import read_image, ImageReadMode
from mmengine.model import revert_sync_batchnorm
from mmengine import Config
import vltseg
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
import cv2
import numpy as np
from PIL import Image
import glob

# Register all modules
register_all_modules(True)

def process_image(model, image_path, output_dir, device):
    """Process a single image and save the segmentation result"""
    try:
        # Load and preprocess image
        image = read_image(image_path, mode=ImageReadMode.RGB).to(torch.float32).to(device).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            result = model.predict(image)
            predictions = result[0].seg_logits.data
            seg_pred = predictions.argmax(dim=1).squeeze(0).cpu().numpy()
        
        # Save result
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"seg_{filename}")
        
        # Convert to PIL and save
        seg_image = Image.fromarray(seg_pred.astype(np.uint8))
        seg_image.save(output_path)
        
        print(f"Processed: {filename}")
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def main():
    # Configuration
    config_path = "configs/mask2former_evaclip_2xb8_5k_gta2cityscapes.py"
    checkpoint_path = "/data/choihy/study_0630/VLTSeg/vltseg_checkpoint_mapillary+cityscapes_1.pth"
    input_dir = "/data/choihy/study_0630/data/cropped"
    output_dir = "/data/choihy/study_0630/output_segmentation"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = MODELS.build(Config.fromfile(config_path).model)
    model.to(device)
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    if device == 'cpu':
        model = revert_sync_batchnorm(model)
    
    # Get all image files
    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    success_count = 0
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        if process_image(model, image_path, output_dir, device):
            success_count += 1
    
    print(f"Processing complete! {success_count}/{len(image_files)} images processed successfully.")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 