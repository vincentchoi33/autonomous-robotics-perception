import torch
import torch.nn as nn
from mmengine.model import revert_sync_batchnorm
from mmengine import Config
import vltseg
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
import numpy as np
import os
import time

# Register all modules
register_all_modules(True)

class SimpleQuantizedWrapper(nn.Module):
    """Simplified wrapper for quantization that avoids complex model structures"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Simple forward pass without complex metadata handling
        with torch.no_grad():
            # Use encode_decode directly to avoid predict method issues
            seg_logits = self.model.encode_decode(x, [])
            return seg_logits

def simple_dynamic_quantization():
    """Apply simple dynamic quantization to VLTSeg model"""
    
    config = "/data/choihy/study_0630/VLTSeg/configs/custom_inference.py"
    device = "cpu"  # Use CPU for quantization
    
    print("Loading model for quantization...")
    cfg = Config.fromfile(config)
    model = MODELS.build(cfg.model)
    model.to(device)
    
    # Load checkpoint
    checkpoint_path = '/data/choihy/study_0630/VLTSeg/vltseg_checkpoint_mapillary+cityscapes_1.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    # Convert to CPU and disable sync batch norm
    model = revert_sync_batchnorm(model)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Disable xformers for compatibility
    print("Disabling xformers for quantization compatibility...")
    for module in model.modules():
        if hasattr(module, 'xattn') and module.xattn:
            module.xattn = False
            print(f"Disabled xformers in {type(module).__name__}")
    
    # Create simple wrapper
    wrapped_model = SimpleQuantizedWrapper(model)
    wrapped_model.eval()
    
    # Test original model first
    print("Testing original model...")
    test_input = torch.randn(1, 3, 1022, 1022)
    
    with torch.no_grad():
        start_time = time.time()
        original_output = wrapped_model(test_input)
        original_time = time.time() - start_time
    
    print(f"Original model inference successful! Output shape: {original_output.shape}")
    print(f"Original inference time: {original_time:.3f} seconds")
    
    # Apply dynamic quantization
    print("Applying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        wrapped_model, 
        {torch.nn.Linear, torch.nn.Conv2d}, 
        dtype=torch.qint8
    )
    
    # Test quantized model
    print("Testing quantized model...")
    with torch.no_grad():
        start_time = time.time()
        quantized_output = quantized_model(test_input)
        quantized_time = time.time() - start_time
    
    print(f"Quantized model inference successful! Output shape: {quantized_output.shape}")
    print(f"Quantized inference time: {quantized_time:.3f} seconds")
    
    # Compare outputs
    output_diff = torch.abs(original_output - quantized_output).max().item()
    print(f"Maximum output difference: {output_diff:.6f}")
    
    # Save quantized model
    quantized_path = "/data/choihy/study_0630/VLTSeg/vltseg_model_quantized_simple.pt"
    torch.save(quantized_model.state_dict(), quantized_path)
    print(f"Quantized model saved to: {quantized_path}")
    
    # Compare model sizes
    original_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
    quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)  # MB
    
    print(f"\nModel size comparison:")
    print(f"Original model: {original_size:.2f} MB")
    print(f"Quantized model: {quantized_size:.2f} MB")
    print(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
    
    # Speed comparison
    speedup = original_time / quantized_time
    print(f"Speed improvement: {speedup:.2f}x faster")
    
    return quantized_path

def create_simple_inference_script():
    """Create a simple inference script for the quantized model"""
    
    script_content = '''import torch
import torch.nn as nn
from mmengine.model import revert_sync_batchnorm
from mmengine import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time

# Register modules
register_all_modules(True)

class SimpleQuantizedWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        with torch.no_grad():
            seg_logits = self.model.encode_decode(x, [])
            return seg_logits

def load_quantized_model():
    """Load the quantized model"""
    config = "configs/custom_inference.py"
    device = "cpu"
    
    # Load original model structure
    cfg = Config.fromfile(config)
    model = MODELS.build(cfg.model)
    model.to(device)
    
    # Load checkpoint
    checkpoint_path = 'vltseg_checkpoint_mapillary+cityscapes_1.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    # Convert to CPU
    model = revert_sync_batchnorm(model)
    model.eval()
    
    # Disable xformers
    for module in model.modules():
        if hasattr(module, 'xattn') and module.xattn:
            module.xattn = False
    
    # Wrap model
    wrapped_model = SimpleQuantizedWrapper(model)
    wrapped_model.eval()
    
    # Apply quantization
    quantized_model = torch.quantization.quantize_dynamic(
        wrapped_model, 
        {torch.nn.Linear, torch.nn.Conv2d}, 
        dtype=torch.qint8
    )
    
    return quantized_model

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
    print("Loading quantized model...")
    model = load_quantized_model()
    
    print("Preprocessing image...")
    input_data = preprocess_image(image_path)
    
    print("Running inference...")
    start_time = time.time()
    with torch.no_grad():
        output = model(input_data)
    inference_time = time.time() - start_time
    
    print(f"Inference completed in {inference_time:.3f} seconds")
    
    # Postprocess output
    pred_sem_seg, pred_color = postprocess_output(output)
    
    return pred_sem_seg, pred_color, inference_time

if __name__ == "__main__":
    # Example usage
    image_path = "your_image.png"  # Replace with your image path
    
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
'''
    
    with open("/data/choihy/study_0630/VLTSeg/simple_quantized_inference.py", "w") as f:
        f.write(script_content)
    
    print("Simple quantized inference script created: simple_quantized_inference.py")

def post_training_quantization():
    """Apply post-training quantization with more control"""
    
    config = "/data/choihy/study_0630/VLTSeg/configs/custom_inference.py"
    device = "cpu"
    
    print("Loading model for post-training quantization...")
    cfg = Config.fromfile(config)
    model = MODELS.build(cfg.model)
    model.to(device)
    
    # Load checkpoint
    checkpoint_path = '/data/choihy/study_0630/VLTSeg/vltseg_checkpoint_mapillary+cityscapes_1.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    model = revert_sync_batchnorm(model)
    model.eval()
    
    # Disable xformers
    for module in model.modules():
        if hasattr(module, 'xattn') and module.xattn:
            module.xattn = False
    
    # Create wrapper
    wrapped_model = SimpleQuantizedWrapper(model)
    wrapped_model.eval()
    
    # Set quantization configuration
    wrapped_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare for quantization
    prepared_model = torch.quantization.prepare(wrapped_model)
    
    # Calibrate with dummy data
    print("Calibrating model...")
    for i in range(5):
        dummy_input = torch.randn(1, 3, 1022, 1022)
        prepared_model(dummy_input)
    
    # Convert to quantized model
    quantized_model = torch.quantization.convert(prepared_model)
    
    # Test quantized model
    test_input = torch.randn(1, 3, 1022, 1022)
    with torch.no_grad():
        output = quantized_model(test_input)
    
    print(f"Post-training quantization successful! Output shape: {output.shape}")
    
    # Save model
    ptq_path = "/data/choihy/study_0630/VLTSeg/vltseg_model_ptq.pt"
    torch.save(quantized_model.state_dict(), ptq_path)
    print(f"Post-training quantized model saved to: {ptq_path}")
    
    return ptq_path

if __name__ == "__main__":
    print("=== VLTSeg Simple Quantization ===\n")
    
    # Simple dynamic quantization
    print("1. Applying Simple Dynamic Quantization...")
    try:
        simple_path = simple_dynamic_quantization()
        print("✅ Simple dynamic quantization completed successfully!")
    except Exception as e:
        print(f"❌ Simple dynamic quantization failed: {e}")
        simple_path = None
    
    print("\n" + "="*50 + "\n")
    
    # Post-training quantization (optional)
    print("2. Applying Post-Training Quantization...")
    try:
        ptq_path = post_training_quantization()
        print("✅ Post-training quantization completed successfully!")
    except Exception as e:
        print(f"❌ Post-training quantization failed: {e}")
        print("This is normal - PTQ requires more careful setup")
        ptq_path = None
    
    # Create inference script
    create_simple_inference_script()
    
    print("\n=== Quantization Summary ===")
    if simple_path:
        print("✅ Simple dynamic quantization completed!")
        print(f"📁 Model saved: {simple_path}")
    if ptq_path:
        print("✅ Post-training quantization completed!")
        print(f"📁 Model saved: {ptq_path}")
    
    print("🚀 Use simple_quantized_inference.py for deployment!")
    print("💡 Dynamic quantization is recommended for most use cases") 