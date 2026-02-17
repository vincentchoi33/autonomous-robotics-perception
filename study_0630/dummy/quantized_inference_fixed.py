import torch
import torch.nn as nn
from mmengine.model import revert_sync_batchnorm
from mmengine import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules to register them
import vltseg
from vltseg.models.eva02 import EVA02
from vltseg.models.mask2former_head_fixed import Mask2FormerHeadFixed

# Register all modules
register_all_modules(True)

def apply_quantization_to_modules(model):
    """Apply quantization to specific modules"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            quantized_module = torch.quantization.quantize_dynamic(
                module, {nn.Linear}, dtype=torch.qint8
            )
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, quantized_module)
            else:
                setattr(model, child_name, quantized_module)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'backbone' in name:
            try:
                quantized_module = torch.quantization.quantize_dynamic(
                    module, {nn.Conv2d}, dtype=torch.qint8
                )
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, quantized_module)
                else:
                    setattr(model, child_name, quantized_module)
            except Exception as e:
                continue

def load_quantized_model():
    """Load and quantize the model"""
    config = "configs/custom_inference.py"
    device = "cpu"
    
    print("Loading model configuration...")
    cfg = Config.fromfile(config)
    
    # Ensure custom models are registered
    if 'EVA02' not in MODELS.module_dict:
        print("Registering EVA02 model...")
        MODELS.register_module(name='EVA02', module=EVA02)
    
    if 'Mask2FormerHeadFixed' not in MODELS.module_dict:
        print("Registering Mask2FormerHeadFixed model...")
        MODELS.register_module(name='Mask2FormerHeadFixed', module=Mask2FormerHeadFixed)
    
    print("Building model...")
    model = MODELS.build(cfg.model)
    model.to(device)
    
    # Load checkpoint
    checkpoint_path = 'vltseg_checkpoint_mapillary+cityscapes_1.pth'
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    # Convert to CPU
    model = revert_sync_batchnorm(model)
    model.eval()
    
    # Disable xformers
    print("Disabling xformers...")
    for module in model.modules():
        if hasattr(module, 'xattn') and module.xattn:
            module.xattn = False
    
    # Apply quantization
    print("Applying quantization...")
    apply_quantization_to_modules(model)
    
    return model

def preprocess_image(image_path):
    """Preprocess image for inference"""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((1022, 1022)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)

def postprocess_output(output):
    """Postprocess model output"""
    pred_sem_seg = torch.argmax(output[0], dim=0).cpu().numpy().astype(np.uint8)
    
    # Cityscapes color palette
    CITYSCAPES_PALETTE = [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ]
    
    pred_color = np.zeros((pred_sem_seg.shape[0], pred_sem_seg.shape[1], 3), dtype=np.uint8)
    for class_id in range(19):
        mask = pred_sem_seg == class_id
        pred_color[mask] = CITYSCAPES_PALETTE[class_id]
    
    return pred_sem_seg, pred_color

def inference_with_quantized_model(image_path):
    """Run inference using quantized model"""
    print("Loading and quantizing model...")
    model = load_quantized_model()
    
    print("Preprocessing image...")
    input_data = preprocess_image(image_path)
    
    print("Running inference...")
    start_time = time.time()
    with torch.no_grad():
        output = model.encode_decode(input_data, [])
    inference_time = time.time() - start_time
    
    print(f"Inference completed in {inference_time:.3f} seconds")
    
    # Postprocess output
    pred_sem_seg, pred_color = postprocess_output(output)
    
    return pred_sem_seg, pred_color, inference_time

if __name__ == "__main__":
    # Example usage - use a test image if available
    if os.path.exists("/data/choihy/study_0630/data/cropped"):
        # Find first image in images directory
        image_files = [f for f in os.listdir("/data/choihy/study_0630/data/cropped") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            image_path = os.path.join("/data/choihy/study_0630/data/cropped", image_files[0])
            print(f"Using test image: {image_path}")
        else:
            image_path = "your_image.png"  # Replace with your image path
            print("No test images found. Please replace 'your_image.png' with actual image path.")
    else:
        image_path = "your_image.png"  # Replace with your image path
        print("Please replace 'your_image.png' with actual image path.")
    
    try:
        pred_sem_seg, pred_color, inference_time = inference_with_quantized_model(image_path)
        
        # Save results
        Image.fromarray(pred_sem_seg).save("segmentation_quantized_grayscale.png")
        Image.fromarray(pred_color).save("segmentation_quantized_color.png")
        
        print(f"Quantized inference completed successfully!")
        print(f"Inference time: {inference_time:.3f} seconds")
        print(f"Results saved as:")
        print(f"  - segmentation_quantized_grayscale.png")
        print(f"  - segmentation_quantized_color.png")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Make sure the image path is correct and the model files are available.")
        import traceback
        traceback.print_exc()
