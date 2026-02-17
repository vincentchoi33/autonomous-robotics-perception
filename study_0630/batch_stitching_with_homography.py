"""
호모그래피 행렬을 사용한 배치 이미지 스티칭
- superglue_matching.py에서 계산된 호모그래피 행렬을 전체 시퀀스에 적용
"""

import cv2
import numpy as np
import os
from pathlib import Path
from glob import glob

def load_homography_matrix(matrix_path):
    if matrix_path and os.path.exists(matrix_path):
        return np.load(matrix_path)
    else:
        print(f"호모그래피 행렬 파일을 찾을 수 없습니다: {matrix_path}")
        return None

def simple_crop(stitched_img, tolerance=5):
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    center_x, center_y = width // 2, height // 2
    # 초기 크기: 전체의 1/2
    cur_w = int(width * 0.5)
    cur_h = int(height * 0.5)
    # 초기 영역
    x1 = center_x - cur_w // 2
    y1 = center_y - cur_h // 2
    x2 = center_x + cur_w // 2
    y2 = center_y + cur_h // 2
    # 경계 보정
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    best_bbox = (x1, y1, x2, y2)
    while True:
        # 다음 크기
        next_w = int(cur_w * 1.1)
        next_h = int(cur_h * 1.1)
        next_x1 = center_x - next_w // 2
        next_y1 = center_y - next_h // 2
        next_x2 = center_x + next_w // 2
        next_y2 = center_y + next_h // 2
        # 경계 보정
        next_x1 = max(0, next_x1)
        next_y1 = max(0, next_y1)
        next_x2 = min(width, next_x2)
        next_y2 = min(height, next_y2)
        region = gray[next_y1:next_y2, next_x1:next_x2]
        if np.any(region <= tolerance):
            break
        # black이 없으면 확장
        best_bbox = (next_x1, next_y1, next_x2, next_y2)
        cur_w, cur_h = next_w, next_h
        # 전체 이미지 크기에 도달하면 중지
        if cur_w >= width or cur_h >= height:
            break
    x1, y1, x2, y2 = best_bbox
    return stitched_img[y1:y2, x1:x2]

def stitch_image_pair(img1, img2, H):
    if H is None:
        return img2
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners2 = cv2.perspectiveTransform(corners1, H)
    all_corners = np.concatenate([corners1, corners2])
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    result = cv2.warpPerspective(img1, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h2+t[1], t[0]:w2+t[0]] = img2
    return result

def batch_stitch_with_homography(left_dir, right_dir, output_dir, H):
    original_dir = os.path.join(output_dir, "original")
    cropped_dir = os.path.join(output_dir, "cropped")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)
    left_files = sorted(list(Path(left_dir).glob("*.png")))
    right_files = sorted(list(Path(right_dir).glob("*.png")))
    n = min(len(left_files), len(right_files))
    for i in range(n):
        left_img = cv2.imread(str(left_files[i]))
        right_img = cv2.imread(str(right_files[i]))
        if left_img is None or right_img is None:
            continue
        stitched = stitch_image_pair(left_img, right_img, H)
        # 원본 저장
        cv2.imwrite(os.path.join(original_dir, f"stitched_{i:05d}.png"), stitched)
        # 크롭 저장
        cropped = simple_crop(stitched)
        cv2.imwrite(os.path.join(cropped_dir, f"stitched_{i:05d}.png"), cropped)

def save_images_to_video(image_dir, output_video_path, fps=30):
    image_files = sorted(glob(os.path.join(image_dir, '*.png')))
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    first_img = cv2.imread(image_files[0])
    height, width = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is not None:
            out.write(img)
    out.release()
    print(f"Saved video: {output_video_path}")

if __name__ == "__main__":
    # 경로 및 파일명 설정
    left_dir = "./data/right"
    right_dir = "./data/left"
    output_dir = "./data/"
    homography_file = "homography_matrix.npy"
    H = load_homography_matrix(homography_file)
    if H is not None:
        print('start stitching')
        batch_stitch_with_homography(left_dir, right_dir, output_dir, H)
        # 동영상 저장
        print('start saving video')
        save_images_to_video(os.path.join(output_dir, "original"), os.path.join(output_dir, "original.mp4"), fps=30)
        print('start saving cropped video')
        save_images_to_video(os.path.join(output_dir, "cropped"), os.path.join(output_dir, "cropped.mp4"), fps=30) 
        print('done')