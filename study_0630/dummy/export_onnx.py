import torch
import torchvision
from mmengine.model import revert_sync_batchnorm
from mmengine import Config
import vltseg
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
import onnx
import onnxruntime
import numpy as np
import os

# Register all modules
register_all_modules(True)

def rsz(input, l):
    transform = torchvision.transforms.Resize((l, l))
    return transform(input)

class OnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        # 실제 이미지 메타데이터 생성
        batch_img_metas = [{
            'img_shape': (x.shape[2], x.shape[3]),
            'ori_shape': (x.shape[2], x.shape[3]),
            'pad_shape': (x.shape[2], x.shape[3]),
            'scale_factor': (1.0, 1.0),
            'flip': False,
            'flip_direction': None
        }]
        
        seg_logits = self.model.encode_decode(x, batch_img_metas)
        return seg_logits

def export_to_onnx():
    """Export VLTSeg model to ONNX format"""
    
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
    
    # Disable xformers for ONNX export
    print("Disabling xformers for ONNX compatibility...")
    for module in model.modules():
        if hasattr(module, 'xattn') and module.xattn:
            module.xattn = False
            print(f"Disabled xformers in {type(module).__name__}")
    
    # Create dummy input for ONNX export
    # Input size: (1, 3, 1022, 1022) - batch_size, channels, height, width
    dummy_input = torch.randn(1, 3, 1022, 1022).to(device)
    
    # ONNX export path
    onnx_path = "/data/choihy/study_0630/VLTSeg/vltseg_model.onnx"
    
    print("Exporting to ONNX...")
    
    # 모델 래핑
    onnx_model = OnnxWrapper(model)

    # ONNX export
    torch.onnx.export(
        onnx_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX model exported to: {onnx_path}")
    
    # Verify ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")
    
    # Test ONNX inference
    print("Testing ONNX inference...")
    
    # Create ONNX Runtime session
    ort_session = onnxruntime.InferenceSession(onnx_path)
    
    # Prepare test input (normalized)
    test_input = torch.randn(1, 3, 1022, 1022).numpy()
    
    # Normalize input (ImageNet normalization)
    mean = np.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
    std = np.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)
    test_input = (test_input - mean) / std
    
    # Run ONNX inference
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.astype(np.float32)}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"ONNX inference successful! Output shape: {ort_outputs[0].shape}")
    
    return onnx_path

def create_onnx_inference_script():
    """Create a simple ONNX inference script for edge devices"""
    
    script_content = '''import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch

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
    """Preprocess image for ONNX inference"""
    # Load and resize image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((1022, 1022)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image)
    
    # Normalize (ImageNet normalization)
    mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor.unsqueeze(0).numpy()

def postprocess_output(output):
    """Postprocess ONNX output to get segmentation map"""
    # Get class predictions
    pred_sem_seg = np.argmax(output[0], axis=0).astype(np.uint8)
    
    # Convert to color using Cityscapes palette
    pred_color = np.zeros((pred_sem_seg.shape[0], pred_sem_seg.shape[1], 3), dtype=np.uint8)
    for class_id in range(19):
        mask = pred_sem_seg == class_id
        pred_color[mask] = CITYSCAPES_PALETTE[class_id]
    
    return pred_sem_seg, pred_color

def inference_with_onnx(image_path, onnx_path):
    """Run inference using ONNX model"""
    # Create ONNX Runtime session
    ort_session = onnxruntime.InferenceSession(onnx_path)
    
    # Preprocess image
    input_data = preprocess_image(image_path)
    
    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_data.astype(np.float32)}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Postprocess output
    pred_sem_seg, pred_color = postprocess_output(ort_outputs[0])
    
    return pred_sem_seg, pred_color

if __name__ == "__main__":
    # Example usage
    image_path = "your_image.png"
    onnx_path = "vltseg_model.onnx"
    
    pred_sem_seg, pred_color = inference_with_onnx(image_path, onnx_path)
    
    # Save results
    Image.fromarray(pred_sem_seg).save("segmentation_grayscale.png")
    Image.fromarray(pred_color).save("segmentation_color.png")
    
    print("Inference completed!")
'''
    
    with open("/data/choihy/study_0630/VLTSeg/onnx_inference_example.py", "w") as f:
        f.write(script_content)
    
    print("ONNX inference example script created: onnx_inference_example.py")

if __name__ == "__main__":
    # Export to ONNX
    onnx_path = export_to_onnx()
    
    # Create inference example script
    create_onnx_inference_script()
    
    print("\nONNX export completed!")
    print(f"ONNX model: {onnx_path}")
    print("You can now use this ONNX model on edge devices without PyTorch installation.")
    print("Use onnx_inference_example.py as a reference for ONNX inference.") 