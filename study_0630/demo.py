import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# === STEP 0: 파일 불러오기 ===
image_dir = "/data/choihy/study_0630/data/right"  # 이미지 폴더 경로
image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))  # 확장자에 맞게 조정
assert len(image_paths) >= 2, "프레임 이미지가 2개 이상 있어야 합니다."

# === STEP 1: YOLO 모델 로딩 ===
model = YOLO("yolo11n.pt")
track_history = defaultdict(lambda: [])

# === STEP 2: VideoWriter 설정 ===
sample_img = cv2.imread(image_paths[0])
h, w = sample_img.shape[:2]
fps = 10
video_writer = cv2.VideoWriter("yolo_homography_output_right.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# === STEP 3: 프레임 반복 ===
for i in range(len(image_paths) - 1):
    frame_t = cv2.imread(image_paths[i])
    frame_tp1 = cv2.imread(image_paths[i + 1])

    # Homography 추정 (ORB + RANSAC)
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(frame_t, None)
    kp2, des2 = orb.detectAndCompute(frame_tp1, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) < 4:
        print(f"⚠️ Not enough matches at frame {i}, skipping.")
        continue

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

    warped_tp1 = cv2.warpPerspective(frame_tp1, H, (w, h))

    # YOLO tracking (frame_t는 앞에서 persist 처리됨)
    if i == 0:
        model.track(frame_t, persist=True)
    result = model.track(warped_tp1, persist=True)[0]

    # 추적 결과 시각화
    if result.boxes and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()
        frame_out = result.plot()

        for box, track_id in zip(boxes, track_ids):
            x, y, w_box, h_box = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_out, [points], isClosed=False, color=(0, 255, 0), thickness=2)
    else:
        frame_out = warped_tp1.copy()

    video_writer.write(frame_out)
    print(f"✅ Processed frame {i} → {i+1}")

video_writer.release()
print("🎬 Video saved as yolo_homography_output.mp4")