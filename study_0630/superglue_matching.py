"""
SuperGlue를 사용한 이미지 매칭 예시
- 두 이미지 간의 특징점 매칭
- 매칭 결과 시각화
- 호모그래피 계산 및 이미지 정렬
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os

# SuperGlue 관련 import (실제 설치가 필요할 수 있음)
try:
    from superglue import SuperGlue
    from superpoint import SuperPoint
except ImportError:
    print("SuperGlue가 설치되지 않았습니다. 간단한 SIFT 매칭으로 대체합니다.")
    SUPERGLUE_AVAILABLE = False
else:
    SUPERGLUE_AVAILABLE = True

class SimpleSuperGlue:
    """
    SuperGlue의 간단한 구현 (실제 SuperGlue 대신 SIFT + FLANN 사용)
    """
    
    def __init__(self):
        # SIFT 특징점 검출기
        self.sift = cv2.SIFT_create()
        
        # FLANN 매처
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    def match(self, img1, img2):
        """
        두 이미지 간의 특징점 매칭
        
        Args:
            img1, img2: 입력 이미지들 (BGR)
        
        Returns:
            kp1, kp2: 특징점들
            matches: 매칭 결과
        """
        # 그레이스케일 변환
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 특징점 및 디스크립터 검출
        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return [], [], []
        
        # 매칭
        matches = self.flann.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test 적용
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        return kp1, kp2, good_matches

def visualize_matches(img1, img2, kp1, kp2, matches, title="Feature Matching"):
    """
    매칭 결과 시각화
    
    Args:
        img1, img2: 입력 이미지들
        kp1, kp2: 특징점들
        matches: 매칭 결과
        title: 그래프 제목
    """
    # 매칭 결과 그리기
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # 결과 시각화
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Image 2')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f'{title} ({len(matches)} matches)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('superglue_matching_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return img_matches

def compute_homography(kp1, kp2, matches, min_matches=10):
    """
    호모그래피 행렬 계산
    
    Args:
        kp1, kp2: 특징점들
        matches: 매칭 결과
        min_matches: 최소 매칭 수
    
    Returns:
        H: 호모그래피 행렬
        mask: 인라이어 마스크
    """
    if len(matches) < min_matches:
        print(f"매칭 수가 부족합니다: {len(matches)} < {min_matches}")
        return None, None
    
    # 매칭된 점들의 좌표 추출
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # 호모그래피 계산 (RANSAC 사용)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H, mask

def align_images(img1, img2, H):
    """
    호모그래피를 사용하여 이미지 정렬
    
    Args:
        img1, img2: 입력 이미지들
        H: 호모그래피 행렬
    
    Returns:
        aligned_img: 정렬된 이미지
    """
    if H is None:
        return img2
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 변환된 이미지의 크기 계산
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners2 = cv2.perspectiveTransform(corners1, H)
    
    # 전체 영역을 포함하는 크기 계산
    all_corners = np.concatenate([corners1, corners2])
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # 평행이동 행렬
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    
    # 이미지 변환
    result = cv2.warpPerspective(img1, Ht.dot(H), (xmax-xmin, ymax-ymin))
    
    # 두 번째 이미지를 결과에 합성
    result[t[1]:h2+t[1], t[0]:w2+t[0]] = img2
    
    return result

def main():
    """
    메인 실행 함수
    """
    print("SuperGlue 이미지 매칭 예시")
    print("=" * 50)
    
    # 입력 이미지 경로
    img2_path = "./data/left/left_00204.png"
    img1_path = "./data/right/right_00204.png"
    
    # 이미지 로드
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("입력 이미지를 찾을 수 없습니다.")
        print(f"경로 확인: {img1_path}, {img2_path}")
        return
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("이미지 로드에 실패했습니다.")
        return
    
    print(f"이미지 1 크기: {img1.shape}")
    print(f"이미지 2 크기: {img2.shape}")
    
    # SuperGlue 매칭 (또는 SIFT 대체)
    if SUPERGLUE_AVAILABLE:
        print("SuperGlue를 사용한 매칭...")
        # 실제 SuperGlue 구현은 여기에 추가
        pass
    else:
        print("SIFT + FLANN을 사용한 매칭...")
        matcher = SimpleSuperGlue()
        kp1, kp2, matches = matcher.match(img1, img2)
    
    print(f"검출된 특징점 수: Image1={len(kp1)}, Image2={len(kp2)}")
    print(f"매칭된 특징점 수: {len(matches)}")
    
    # 매칭 결과 시각화
    img_matches = visualize_matches(img1, img2, kp1, kp2, matches, "SIFT + FLANN Matching")
    
    # 호모그래피 계산
    H, mask = compute_homography(kp1, kp2, matches)
    
    if H is not None:
        print("호모그래피 계산 성공!")
        print(f"호모그래피 행렬:\n{H}")
        
        # 호모그래피 행렬을 파일로 저장
        homography_file = "homography_matrix.npy"
        np.save(homography_file, H)
        print(f"호모그래피 행렬 저장: {homography_file}")
        
        # 인라이어 매칭만 추출
        inlier_matches = [m for i, m in enumerate(matches) if mask[i]]
        print(f"인라이어 매칭 수: {len(inlier_matches)}")
        
        # 인라이어 매칭 시각화
        img_inliers = visualize_matches(img1, img2, kp1, kp2, inlier_matches, "Inlier Matches")
        
        # 이미지 정렬
        aligned_img = align_images(img1, img2, H)
        
        # 정렬 결과 저장
        cv2.imwrite("output_images/aligned_result.png", aligned_img)
        print("정렬된 이미지 저장: output_images/aligned_result.png")
        
        # 정렬 결과 시각화
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        plt.title('Image 1')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.title('Image 2')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
        plt.title('Aligned Result')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('image_alignment_result.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 배치 스티칭을 위한 안내
        print("\n" + "="*50)
        print("배치 스티칭을 위해 다음 명령을 실행하세요:")
        print("python batch_stitching_with_homography.py")
        print("="*50)
        
    else:
        print("호모그래피 계산에 실패했습니다.")
    
    print("\n생성된 파일들:")
    print("- superglue_matching_result.png")
    print("- image_alignment_result.png")
    print("- output_images/aligned_result.png")
    if H is not None:
        print(f"- {homography_file}")

if __name__ == "__main__":
    main() 