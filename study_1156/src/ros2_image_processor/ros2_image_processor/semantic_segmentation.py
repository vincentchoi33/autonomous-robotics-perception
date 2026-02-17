import cv2
import numpy as np
import onnxruntime as ort
import os
import requests
import time

class SemanticSegmentation:
    def __init__(self, model_path=None):
        """
        MaskFormer ONNX Semantic Segmentation
        """
        self.model_path = model_path or self._download_model()
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        
        # Cityscapes color palette (BGR)
        self.class_colors = [
            [128, 64, 128],   # road
            [244, 35, 232],  # sidewalk
            [70, 70, 70],    # building
            [102, 102, 156], # wall
            [190, 153, 153], # fence
            [153, 153, 153], # pole
            [250, 170, 30],  # traffic light
            [220, 220, 0],   # traffic sign
            [107, 142, 35],  # vegetation
            [152, 251, 152], # terrain
            [70, 130, 180],  # sky
            [220, 20, 60],   # person
            [255, 0, 0],     # rider
            [0, 0, 142],     # car
            [0, 0, 70],      # truck
            [0, 60, 100],    # bus
            [0, 80, 100],    # train
            [0, 0, 230],     # motorcycle
            [119, 11, 32],   # bicycle
        ]
        
        self.class_names = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
        ]
        
        self._load_model()
        print(f"MaskFormer ONNX Semantic Segmentation initialized")
    
    def _download_model(self):
        """
        MaskFormer 모델 다운로드
        """
        model_path = "/ros2_ws/maskformer_resnet101_cityscapes.onnx"
        
        if not os.path.exists(model_path):
            print("MaskFormer model not found, attempting to download...")
            try:
                # MaskFormer Cityscapes ONNX 모델 URL
                model_url = "https://huggingface.co/onnx-community/maskformer-resnet101-cityscapes/resolve/main/onnx/model.onnx"
                
                print(f"Downloading MaskFormer model from {model_url}")
                response = requests.get(model_url, timeout=60)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"MaskFormer model downloaded successfully: {model_path}")
                
            except Exception as e:
                print(f"Failed to download MaskFormer model: {e}")
                print("Using fallback color-based segmentation")
                return None
        
        return model_path
    
    def _load_model(self):
        """
        ONNX 모델 로드
        """
        if self.model_path and os.path.exists(self.model_path):
            try:
                # ONNX Runtime 세션 생성 (CPU 최적화)
                providers = ['CPUExecutionProvider']
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                
                # 입력/출력 정보 가져오기
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                self.input_shape = self.session.get_inputs()[0].shape
                
                print(f"ONNX model loaded successfully: {self.input_shape}")
                
            except Exception as e:
                print(f"Failed to load ONNX model: {e}")
                self.session = None
        else:
            print("Using fallback color-based segmentation")
            self.session = None
    
    def preprocess_image(self, image):
        """
        이미지 전처리 (DeepLabV3+ 입력 형식)
        """
        # 이미지 크기 조정 (처리 속도 향상)
        height, width = image.shape[:2]
        target_size = (512, 512)  # DeepLabV3+ 표준 입력 크기
        
        # 리사이즈
        resized = cv2.resize(image, target_size)
        
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 정규화 (0-255 -> 0-1)
        normalized = rgb.astype(np.float32) / 255.0
        
        # ImageNet 평균/표준편차로 정규화
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # MaskFormer는 (1, C, H, W) 형태를 요구함
        # (H, W, C) -> (C, H, W)
        normalized = np.transpose(normalized, (2, 0, 1))
        
        # 배치 차원 추가 (1, C, H, W)
        input_tensor = np.expand_dims(normalized, axis=0)
        
        # float32로 명시적 변환 (MaskFormer 요구사항)
        input_tensor = input_tensor.astype(np.float32)
        
        return input_tensor, (height, width)
    
    def fallback_segmentation(self, image):
        """
        ONNX 모델이 없을 때 사용하는 간단한 색상 기반 세그멘테이션
        """
        # 이미지 크기 조정
        height, width = image.shape[:2]
        if width > 320:  # 더 작게 조정
            scale = 320 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # HSV 색상 공간으로 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 간단한 색상 기반 분할
        segmentation_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # 도로 (어두운 회색)
        road_mask = cv2.inRange(hsv, (0, 0, 50), (180, 30, 150))
        segmentation_map[road_mask > 0] = 0
        
        # 하늘 (파란색)
        sky_mask = cv2.inRange(hsv, (100, 50, 100), (130, 255, 255))
        segmentation_map[sky_mask > 0] = 10
        
        # 식물 (녹색)
        vegetation_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
        segmentation_map[vegetation_mask > 0] = 8
        
        # 건물 (회색)
        building_mask = cv2.inRange(hsv, (0, 0, 100), (180, 30, 200))
        segmentation_map[building_mask > 0] = 2
        
        # 차량 (빨간색/파란색)
        vehicle_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | \
                      cv2.inRange(hsv, (110, 100, 100), (130, 255, 255))
        segmentation_map[vehicle_mask > 0] = 13
        
        return segmentation_map
    
    def segment_image(self, image):
        """
        이미지 세그멘테이션 수행
        """
        print("[DEBUG] segment_image called")
        if self.session is not None:
            # ONNX 모델 사용
            try:
                print(f"[DEBUG] Input image shape: {image.shape}")
                # 전처리
                input_tensor, original_shape = self.preprocess_image(image)
                print(f"[DEBUG] Preprocessed input_tensor shape: {input_tensor.shape}")
                t0 = time.time()
                # 추론
                outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
                t1 = time.time()
                print(f"[DEBUG] ONNX inference time: {t1-t0:.3f}s, outputs type: {type(outputs)}, len: {len(outputs)}")
                logits = outputs[0]
                print(f"[DEBUG] logits shape: {logits.shape}")
                # 후처리 - MaskFormer 출력 형태에 맞게 수정
                print(f"[DEBUG] logits[0] shape: {logits[0].shape}")
                
                # MaskFormer 출력이 (100, 20) 형태 - 100개 객체, 20개 클래스
                if logits[0].shape == (100, 20):
                    # 가장 높은 확률의 클래스를 선택
                    class_predictions = np.argmax(logits[0], axis=1)  # (100,)
                    print(f"[DEBUG] class_predictions shape: {class_predictions.shape}")
                    
                    # MaskFormer는 객체 기반이므로, 가장 큰 객체의 클래스를 사용
                    # 또는 모든 객체의 클래스를 평균내서 사용
                    if len(class_predictions) > 0:
                        # 가장 빈번한 클래스를 선택
                        from collections import Counter
                        most_common_class = Counter(class_predictions).most_common(1)[0][0]
                        segmentation_map = np.full((512, 512), most_common_class, dtype=np.uint8)
                    else:
                        segmentation_map = np.zeros((512, 512), dtype=np.uint8)
                else:
                    print(f"[ERROR] Unexpected logits shape: {logits[0].shape}")
                    raise ValueError(f"Unexpected logits shape: {logits[0].shape}")
                
                print(f"[DEBUG] segmentation_map shape (before resize): {segmentation_map.shape}")
                # 원본 크기로 리사이즈
                segmentation_map = cv2.resize(segmentation_map, (original_shape[1], original_shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                print(f"[DEBUG] segmentation_map shape (after resize): {segmentation_map.shape}")
            except Exception as e:
                print(f"[ERROR] ONNX inference failed: {e}, using fallback")
                segmentation_map = self.fallback_segmentation(image)
        else:
            # Fallback 세그멘테이션
            print("[DEBUG] Using fallback segmentation")
            segmentation_map = self.fallback_segmentation(image)
        # 컬러 마스크 생성
        color_mask = self.create_color_mask(segmentation_map)
        print(f"[DEBUG] color_mask shape: {color_mask.shape}")
        return segmentation_map, color_mask
    
    def create_color_mask(self, segmentation_map):
        """
        세그멘테이션 맵을 컬러 마스크로 변환
        """
        height, width = segmentation_map.shape
        color_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        for class_id in range(len(self.class_colors)):
            mask = segmentation_map == class_id
            color_mask[mask] = self.class_colors[class_id]
        
        return color_mask
    
    def overlay_segmentation(self, image, segmentation_map, alpha=0.6):
        """
        원본 이미지에 세그멘테이션 오버레이
        """
        color_mask = self.create_color_mask(segmentation_map)
        
        if color_mask.shape[:2] != image.shape[:2]:
            color_mask = cv2.resize(color_mask, (image.shape[1], image.shape[0]))
        
        overlay = cv2.addWeighted(image, 1-alpha, color_mask, alpha, 0)
        return overlay
    
 