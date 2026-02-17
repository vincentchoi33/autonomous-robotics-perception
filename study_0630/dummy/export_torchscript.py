import torch
import torchvision
from mmengine.model import revert_sync_batchnorm
from mmengine import Config
import vltseg
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
import numpy as np
import os

# Register all modules
register_all_modules(True)

def rsz(input, l):
    transform = torchvision.transforms.Resize((l, l))
    return transform(input)

class TorchScriptWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        # Normalize input (ImageNet normalization)
        mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        
        # Use predict method
        result = self.model.predict(x)
        return result[0].seg_logits.data

def export_to_torchscript():
    """Export VLTSeg model to TorchScript format"""
    
    config = "/data/choihy/study_0630/VLTSeg/configs/custom_inference.py"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    
    # Disable xformers for TorchScript export
    print("Disabling xformers for TorchScript compatibility...")
    for module in model.modules():
        if hasattr(module, 'xattn') and module.xattn:
            module.xattn = False
            print(f"Disabled xformers in {type(module).__name__}")
    
    # Create dummy input for TorchScript export
    # Input size: (1, 3, 1022, 1022) - batch_size, channels, height, width
    dummy_input = torch.randn(1, 3, 1022, 1022).to(device)
    
    # TorchScript export path
    torchscript_path = "/data/choihy/study_0630/VLTSeg/vltseg_model.pt"
    
    print("Exporting to TorchScript...")
    
    # 모델 래핑
    torchscript_model = TorchScriptWrapper(model)

    # TorchScript export
    traced_model = torch.jit.trace(torchscript_model, dummy_input)
    
    # Save TorchScript model
    traced_model.save(torchscript_path)
    
    print(f"TorchScript model exported to: {torchscript_path}")
    
    # Test TorchScript inference
    print("Testing TorchScript inference...")
    
    # Load and test the model
    loaded_model = torch.jit.load(torchscript_path)
    test_input = torch.randn(1, 3, 1022, 1022).to(device)
    
    with torch.no_grad():
        output = loaded_model(test_input)
    
    print(f"TorchScript inference successful! Output shape: {output.shape}")
    
    return torchscript_path

def create_torchscript_inference_script():
    """Create a simple TorchScript inference script for edge devices"""
    
    script_content = '''import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Cityscapes color palette
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

def preprocess_image(image_path):
    """Preprocess image for TorchScript inference"""
    # Load and resize image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((1022, 1022)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)

def postprocess_output(output):
    """Postprocess TorchScript output to get segmentation map"""
    # Get class predictions
    pred_sem_seg = torch.argmax(output[0], dim=0).cpu().numpy().astype(np.uint8)
    
    # Convert to color using Cityscapes palette
    pred_color = np.zeros((pred_sem_seg.shape[0], pred_sem_seg.shape[1], 3), dtype=np.uint8)
    for class_id in range(19):
        mask = pred_sem_seg == class_id
        pred_color[mask] = CITYSCAPES_PALETTE[class_id]
    
    return pred_sem_seg, pred_color

def inference_with_torchscript(image_path, torchscript_path):
    """Run inference using TorchScript model"""
    # Load TorchScript model
    model = torch.jit.load(torchscript_path)
    model.eval()
    
    # Preprocess image
    input_data = preprocess_image(image_path)
    
    # Run inference
    with torch.no_grad():
        output = model(input_data)
    
    # Postprocess output
    pred_sem_seg, pred_color = postprocess_output(output)
    
    return pred_sem_seg, pred_color

if __name__ == "__main__":
    # Example usage
    image_path = "your_image.png"
    torchscript_path = "vltseg_model.pt"
    
    pred_sem_seg, pred_color = inference_with_torchscript(image_path, torchscript_path)
    
    # Save results
    Image.fromarray(pred_sem_seg).save("segmentation_grayscale.png")
    Image.fromarray(pred_color).save("segmentation_color.png")
    
    print("Inference completed!")
'''
    
    with open("/data/choihy/study_0630/VLTSeg/torchscript_inference_example.py", "w") as f:
        f.write(script_content)
    
    print("TorchScript inference example script created: torchscript_inference_example.py")

if __name__ == "__main__":
    # Export to TorchScript
    torchscript_path = export_to_torchscript()
    
    # Create inference example script
    create_torchscript_inference_script()
    
    print("\nTorchScript export completed!")
    print(f"TorchScript model: {torchscript_path}")
    print("You can now use this TorchScript model on edge devices.")
    print("Use torchscript_inference_example.py as a reference for TorchScript inference.")
    print("\nNote: TorchScript requires PyTorch runtime, but is more compatible than ONNX for complex models.") 