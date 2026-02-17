import torch
import numpy as np
import cv2
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image

class SegformerInference:
    def __init__(self, model_name="nvidia/segformer-b5-finetuned-cityscapes-1024-1024", device='cuda'):
        """
        Segformer 세그멘테이션 추론 클래스
        
        Args:
            model_name: Hugging Face 모델 이름
            device: 'cuda' 또는 'cpu'
        """
        self.device = device
        
        try:
            # 모델과 feature extractor 로드
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            
            # GPU로 이동
            self.model.to(device)
            self.model.eval()
            
            print(f"Segformer 모델 로드 완료: {model_name}")
            
        except Exception as e:
            print(f"Segformer 모델 로드 실패: {e}")
            self.model = None
            self.feature_extractor = None
    
    def preprocess(self, image):
        """
        이미지 전처리 (OpenCV BGR -> PIL RGB)
        """
        # BGR to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # NumPy to PIL
        pil_image = Image.fromarray(image_rgb)
        
        return pil_image
    
    def segment(self, image):
        """
        세그멘테이션 수행
        
        Args:
            image: 입력 이미지 (BGR, OpenCV 형식)
            
        Returns:
            segmentation_map: 세그멘테이션 맵 (numpy array)
        """
        if self.model is None or self.feature_extractor is None:
            print("모델이 로드되지 않았습니다.")
            return None
        
        with torch.no_grad():
            try:
                # 전처리
                pil_image = self.preprocess(image)
                
                # Feature extraction
                inputs = self.feature_extractor(images=pil_image, return_tensors="pt")
                
                # GPU로 이동
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 추론
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # 예측
                predicted = logits.argmax(dim=1)
                
                # NumPy로 변환
                segmentation_map = predicted[0].detach().cpu().numpy()
                
                return segmentation_map
                
            except Exception as e:
                print(f"세그멘테이션 실패: {e}")
                return None
    
    def segment_with_visualization(self, image):
        """
        시각화와 함께 세그멘테이션 수행
        """
        segmentation_map = self.segment(image)
        
        if segmentation_map is None:
            return None, None
        
        # 컬러 마스크 생성
        color_mask = self.colorize_mask(segmentation_map)
        
        return segmentation_map, color_mask
    
    def colorize_mask(self, segmentation_map):
        """
        세그멘테이션 맵을 컬러로 변환 (Cityscapes 클래스 기준)
        """
        # Cityscapes 클래스 컬러맵
        colors = [
            [128, 64, 128],   # road
            [244, 35, 232],   # sidewalk
            [70, 70, 70],     # building
            [102, 102, 156],  # wall
            [190, 153, 153],  # fence
            [153, 153, 153],  # pole
            [250, 170, 30],   # traffic light
            [220, 220, 0],    # traffic sign
            [107, 142, 35],   # vegetation
            [152, 251, 152],  # terrain
            [70, 130, 180],   # sky
            [220, 20, 60],    # person
            [255, 0, 0],      # rider
            [0, 0, 142],      # car
            [0, 0, 70],       # truck
            [0, 60, 100],     # bus
            [0, 80, 100],     # train
            [0, 0, 230],      # motorcycle
            [119, 11, 32],    # bicycle
        ]
        
        height, width = segmentation_map.shape
        color_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        for class_id in range(min(len(colors), 20)):  # Segformer는 보통 20개 클래스
            color_mask[segmentation_map == class_id] = colors[class_id]
        
        return color_mask
    
    def overlay_segmentation(self, image, segmentation_map, alpha=0.5):
        """
        원본 이미지에 세그멘테이션 오버레이
        """
        if segmentation_map is None:
            return image
        
        # 마스크를 컬러로 변환
        color_mask = self.colorize_mask(segmentation_map)
        
        # 원본 이미지 리사이즈
        image_resized = cv2.resize(image, (segmentation_map.shape[1], segmentation_map.shape[0]))
        
        # 오버레이
        overlay = cv2.addWeighted(image_resized, 1-alpha, color_mask, alpha, 0)
        
        return overlay
    
    def get_class_names(self):
        """
        클래스 이름 반환
        """
        return [
            'road', 'sidewalk', 'building', 'wall', 'fence',
            'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck',
            'bus', 'train', 'motorcycle', 'bicycle'
        ] 