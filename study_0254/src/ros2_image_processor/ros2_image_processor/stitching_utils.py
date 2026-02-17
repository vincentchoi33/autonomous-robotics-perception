import cv2
import numpy as np
import os

def load_homography_matrix(matrix_path):
    """Load homography matrix."""
    if matrix_path and os.path.exists(matrix_path):
        try:
            H = np.load(matrix_path)
            print(f'Homography matrix loaded from {matrix_path}')
            return H
        except Exception as e:
            print(f'Failed to load homography matrix: {e}')
            return None
    else:
        print(f'Homography matrix file not found: {matrix_path}')
        return None

def simple_crop(stitched_img, tolerance=5):
    """Remove black areas from stitched image."""
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    center_x, center_y = width // 2, height // 2
    cur_w = int(width * 0.5)
    cur_h = int(height * 0.5)
    x1 = center_x - cur_w // 2
    y1 = center_y - cur_h // 2
    x2 = center_x + cur_w // 2
    y2 = center_y + cur_h // 2
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    best_bbox = (x1, y1, x2, y2)
    while True:
        next_w = int(cur_w * 1.1)
        next_h = int(cur_h * 1.1)
        next_x1 = center_x - next_w // 2
        next_y1 = center_y - next_h // 2
        next_x2 = center_x + next_w // 2
        next_y2 = center_y + next_h // 2
        next_x1 = max(0, next_x1)
        next_y1 = max(0, next_y1)
        next_x2 = min(width, next_x2)
        next_y2 = min(height, next_y2)
        region = gray[next_y1:next_y2, next_x1:next_x2]
        if np.any(region <= tolerance):
            break
        best_bbox = (next_x1, next_y1, next_x2, next_y2)
        cur_w, cur_h = next_w, next_h
        if cur_w >= width or cur_h >= height:
            break
    x1, y1, x2, y2 = best_bbox
    return stitched_img[y1:y2, x1:x2]

def stitch_image_pair_homography(img1, img2, H):
    """Stitch two images using homography matrix."""
    if H is None:
        return stitch_images_side_by_side(img1, img2)
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

def stitch_images_side_by_side(left_img, right_img):
    """Simply concatenate two images side by side."""
    if left_img is None or right_img is None:
        return None
    height = min(left_img.shape[0], right_img.shape[0])
    left_resized = cv2.resize(left_img, (int(left_img.shape[1] * height / left_img.shape[0]), height))
    right_resized = cv2.resize(right_img, (int(right_img.shape[1] * height / right_img.shape[0]), height))
    stitched_image = np.hstack((left_resized, right_resized))
    return stitched_image 