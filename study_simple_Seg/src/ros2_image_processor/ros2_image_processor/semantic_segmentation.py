import cv2
import numpy as np
from sklearn.cluster import KMeans

class SemanticSegmentation:
    def __init__(self, num_classes=19):
        """
        Cityscapes 19-class Semantic Segmentation
        """
        self.num_classes = num_classes
        self.kmeans = KMeans(n_clusters=num_classes, n_init=10, random_state=42)

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
        print(f"Semantic Segmentation initialized with Cityscapes 19 classes")
    
    def preprocess_image(self, image):
        """
        이미지 전처리
        
        Args:
            image: 입력 이미지 (BGR)
            
        Returns:
            processed_image: 전처리된 이미지
        """
        # 이미지 크기 조정 (처리 속도 향상을 위해)
        height, width = image.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # 노이즈 제거
        image = cv2.GaussianBlur(image, (5, 5), 0)
        
        return image
    
    def extract_features(self, image):
        """
        이미지에서 특징 추출
        
        Args:
            image: 입력 이미지
            
        Returns:
            features: 추출된 특징
        """
        # 색상 정보
        color_features = image.reshape(-1, 3)
        
        # 텍스처 정보 (그레이스케일)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sobel 엣지 검출
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 텍스처 특징
        texture_features = gradient_magnitude.reshape(-1, 1)
        
        # 색상과 텍스처 특징 결합
        features = np.hstack([color_features, texture_features])
        
        return features
    
    def segment_image(self, image):
        """
        이미지 세그멘테이션 수행
        
        Args:
            image: 입력 이미지 (BGR)
            
        Returns:
            segmentation_map: 세그멘테이션 맵
            color_mask: 컬러 마스크
        """
        # 전처리
        processed_image = self.preprocess_image(image)
        
        # 특징 추출
        features = self.extract_features(processed_image)
        
        # K-means 클러스터링
        labels = self.kmeans.fit_predict(features)
        
        # 세그멘테이션 맵 생성
        height, width = processed_image.shape[:2]
        segmentation_map = labels.reshape(height, width)
        
        # 컬러 마스크 생성
        color_mask = self.create_color_mask(segmentation_map)
        
        return segmentation_map, color_mask
    
    def create_color_mask(self, segmentation_map):
        """
        세그멘테이션 맵을 컬러 마스크로 변환
        
        Args:
            segmentation_map: 세그멘테이션 맵
            
        Returns:
            color_mask: 컬러 마스크
        """
        height, width = segmentation_map.shape
        color_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        for class_id in range(self.num_classes):
            mask = segmentation_map == class_id
            color_mask[mask] = self.class_colors[class_id]
        
        return color_mask
    
    def overlay_segmentation(self, image, segmentation_map, alpha=0.6):
        """
        원본 이미지에 세그멘테이션 오버레이
        
        Args:
            image: 원본 이미지
            segmentation_map: 세그멘테이션 맵
            alpha: 투명도 (0-1)
            
        Returns:
            overlay: 오버레이된 이미지
        """
        # 컬러 마스크 생성
        color_mask = self.create_color_mask(segmentation_map)
        
        # 원본 이미지 크기에 맞춤
        if color_mask.shape[:2] != image.shape[:2]:
            color_mask = cv2.resize(color_mask, (image.shape[1], image.shape[0]))
        
        # 오버레이
        overlay = cv2.addWeighted(image, 1-alpha, color_mask, alpha, 0)
        
        return overlay
    
 