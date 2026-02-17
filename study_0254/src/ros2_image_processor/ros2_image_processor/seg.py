#!/usr/bin/env python3

import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from typing import Tuple

class SemanticSegmentation:
    def __init__(self, model_name="facebook/mask2former-swin-tiny-cityscapes-semantic", device=None):
        """
        Mask2Former Semantic Segmentation (Cityscapes)
        Args:
            model_name: Hugging Face model name
            device: 'cuda' or 'cpu' (auto-selected)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic").to(self.device)
        self.model.eval()

        # GPU information output
        if self.device == "cuda":
            print(f"GPU Available: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("GPU not available, using CPU")

        # Cityscapes class information (Mask2Former order)
        self.class_names = [
            'road', 'sidewalk', 'building', 'wall', 'fence',
            'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck',
            'bus', 'train', 'motorcycle', 'bicycle'
        ]
        # Define colors in BGR order (OpenCV uses BGR)
        self.class_colors = [
            [128, 64, 128],   # 0: road
            [232, 35, 244],   # 1: sidewalk
            [70, 70, 70],     # 2: building
            [156, 102, 102],  # 3: wall
            [153, 153, 190],  # 4: fence
            [153, 153, 153],  # 5: pole
            [30, 170, 250],   # 6: traffic light
            [0, 220, 220],    # 7: traffic sign
            [35, 142, 107],   # 8: vegetation
            [152, 251, 152],  # 9: terrain
            [180, 130, 70],   # 10: sky
            [60, 20, 220],    # 11: person (red)
            [0, 0, 255],      # 12: rider (red)
            [142, 0, 0],      # 13: car (blue)
            [70, 0, 0],       # 14: truck (blue)
            [100, 60, 0],     # 15: bus (blue)
            [100, 80, 0],     # 16: train (blue)
            [230, 0, 0],      # 17: motorcycle (blue)
            [32, 11, 119],    # 18: bicycle (dark red)
        ]
        print(f"Mask2Former({model_name}) loaded on {self.device}")

    def segment_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image: Input image (BGR, OpenCV)
        Returns:
            segmentation_map: (H, W) class indices
            color_mask: (H, W, 3) color mask
        """
        # BGR -> RGB, PIL conversion
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess with processor
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Use Mask2Former's exact post-processing method
            # Use processor.post_process_semantic_segmentation() to generate predicted_semantic_map
            predicted_semantic_map = self.processor.post_process_semantic_segmentation(
                outputs, 
                target_sizes=[pil_image.size[::-1]]  # (width, height) -> (height, width)
            )[0]
            
            # Convert GPU tensor to CPU then to numpy
            if predicted_semantic_map.device.type == 'cuda':
                predicted_semantic_map = predicted_semantic_map.cpu()
            
            # Convert to numpy array
            seg = predicted_semantic_map.numpy().astype(np.uint8)
        
        # Resize to original size (if needed)
        if seg.shape[:2] != (image.shape[0], image.shape[1]):
            seg_map = cv2.resize(seg, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            seg_map = seg
        
        # Debug: Check actual class indices
        unique_classes = np.unique(seg_map)
        if len(unique_classes) > 0:
            print(f"Detected class indices: {unique_classes}")
            for class_id in unique_classes:
                if class_id < len(self.class_names):
                    print(f"   Class {class_id}: {self.class_names[class_id]}")
            
        color_mask = self._create_color_mask(seg_map)
        return seg_map, color_mask

    def _create_color_mask(self, seg_map: np.ndarray) -> np.ndarray:
        color_mask = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for class_id, color in enumerate(self.class_colors):
            color_mask[seg_map == class_id] = color
        return color_mask

    def overlay_segmentation(self, image: np.ndarray, seg_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        color_mask = self._create_color_mask(seg_map)
        if image.shape[:2] != color_mask.shape[:2]:
            color_mask = cv2.resize(color_mask, (image.shape[1], image.shape[0]))
        overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
        return overlay

    def analyze_segmentation(self, seg_map: np.ndarray) -> dict:
        total_pixels = seg_map.size
        analysis = {}
        for class_id, class_name in enumerate(self.class_names):
            count = np.sum(seg_map == class_id)
            percent = (count / total_pixels) * 100
            analysis[class_name] = {'pixels': int(count), 'percentage': round(percent, 2)}
        return analysis

    def get_dominant_classes(self, seg_map: np.ndarray, top_k: int = 5) -> list:
        analysis = self.analyze_segmentation(seg_map)
        sorted_classes = sorted(
            analysis.items(),
            key=lambda x: x[1]['percentage'],
            reverse=True
        )
        return sorted_classes[:top_k] 