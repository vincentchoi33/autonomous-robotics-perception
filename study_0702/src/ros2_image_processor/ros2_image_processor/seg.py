import torch
import numpy as np
import cv2
from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules
from PIL import Image
import os

class Seg:
    def __init__(self, config_file=None, checkpoint_file=None, device=None):
        """
        VLTSeg 기반 세그멘테이션 추론 클래스
        
        Args:
            config_file: VLTSeg 설정 파일 경로
            checkpoint_file: VLTSeg 체크포인트 파일 경로
            device: 'cuda' 또는 'cpu' (None이면 자동 감지)
        """
        self.device = device
        self.model = None
        self._model_loaded = False
        
        # GPU 사용 가능 여부 자동 감지
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
                print("GPU 사용 가능: CUDA 모드로 실행")
            else:
                self.device = 'cpu'
                print("GPU 사용 불가능: CPU 모드로 실행")
        else:
            self.device = device
            
        # VLTSeg 기본 설정 파일과 체크포인트
        if config_file is None:
            config_file = '/ros2_ws/vltseg_configs/mask2former_evaclip_2xb8_5k_gta2cityscapes.py'
        if checkpoint_file is None:
            checkpoint_file = '../../checkpoints/vltseg_checkpoint_cityscapes_1.pth'
            
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
    
    def _load_model(self):
        """모델을 지연 로딩합니다."""
        if self._model_loaded:
            return True
            
        try:
            # MMSegmentation 모듈 등록
            register_all_modules()
            
            # VLTSeg 모델 초기화
            print(f"VLTSeg 모델 로딩 중...")
            print(f"Config: {self.config_file}")
            print(f"Checkpoint: {self.checkpoint_file}")
            
            # 파일 존재 확인
            if not os.path.exists(self.config_file):
                print(f"Config 파일이 존재하지 않습니다: {self.config_file}")
                return False
            if not os.path.exists(self.checkpoint_file):
                print(f"Checkpoint 파일이 존재하지 않습니다: {self.checkpoint_file}")
                return False
            
            self.model = init_model(self.config_file, self.checkpoint_file, device=self.device)
            
            print(f"VLTSeg 모델 로드 완료 (Device: {self.device})")
            self._model_loaded = True
            return True
            
        except Exception as e:
            print(f"VLTSeg 모델 로드 실패: {e}")
            print("CPU 모드로 재시도 중...")
            try:
                # CPU 모드로 재시도
                self.device = 'cpu'
                self.model = init_model(self.config_file, self.checkpoint_file, device=self.device)
                print(f"VLTSeg 모델 로드 완료 (CPU 모드)")
                self._model_loaded = True
                return True
            except Exception as e2:
                print(f"CPU 모드에서도 로드 실패: {e2}")
                self.model = None
                return False
    
    def segment(self, image):
        """
        세그멘테이션 수행
        
        Args:
            image: 입력 이미지 (BGR, OpenCV 형식)
            
        Returns:
            segmentation_map: 세그멘테이션 맵 (numpy array)
        """
        # 모델이 로드되지 않았으면 지연 로딩 시도
        if not self._load_model():
            print("모델 로딩에 실패했습니다.")
            return None
        
        try:
            # VLTSeg 추론
            result = inference_model(self.model, image)
            segmentation_map = result.pred_sem_seg.data[0].cpu().numpy()
            
            return segmentation_map
            
        except Exception as e:
            print(f"세그멘테이션 실패: {e}")
            return None
    
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
        
        for class_id in range(min(len(colors), 20)):
            color_mask[segmentation_map == class_id] = colors[class_id]
        
        return color_mask
    
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

# 기존 SegformerInference와 호환성을 위해 별칭 제공
# SegformerInference = VLTSegInference 