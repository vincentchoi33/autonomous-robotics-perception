# 🚀 Quick Start Guide

이 프로젝트는 ROS 2를 사용한 듀얼 Intel RealSense 카메라 이미지 스티칭 시스템입니다.

## 📋 Prerequisites

- **Docker** (필수)
- **Docker Compose** (필수)
- **Git** (선택사항)

## 🛠️ Installation & Setup

### 1. 프로젝트 다운로드
```bash
git clone <repository-url>
cd study_0702
```

또는 압축 파일을 다운로드하여 압축 해제

### 2. 데이터 다운로드 (필수)
ROS 2 bag 파일을 다운로드하여 `rosbag2_2025_06_16-15_16_29/` 폴더에 저장

### 3. 실행 권한 부여
```bash
chmod +x build.sh
chmod +x start_test.sh
chmod +x test_5_images.sh
```

## 🎯 Quick Execution

### 방법 1: 전체 시스템 실행 (권장)
```bash
# 1. 빌드 및 실행
./build.sh

# 2. 시스템 시작
./start_test.sh
```

### 방법 2: 빠른 테스트 (5개 이미지만)
```bash
# 코드 변경사항 테스트용
./test_5_images.sh
```

## 📁 Project Structure

```
study_0702/
├── src/                          # ROS 2 패키지 소스
│   └── ros2_image_processor/
├── rosbag2_2025_06_16-15_16_29/  # ROS 2 bag 파일 (필수)
├── visualization_output/          # 생성된 이미지 저장 폴더
├── build.sh                      # Docker 이미지 빌드
├── start_test.sh                 # 전체 시스템 실행
├── test_5_images.sh              # 빠른 테스트
├── docker-compose.yml            # Docker Compose 설정
├── Dockerfile                    # Docker 이미지 정의
└── README.md                     # 상세 문서
```

## 🔧 Manual Execution (고급 사용자)

### 1. Docker 이미지 빌드
```bash
docker build -t study_0702-ros2_image_processor .
```

### 2. 컨테이너 실행
```bash
docker-compose up -d
```

### 3. 노드 실행
```bash
# 터미널 1: 이미지 프로세서 노드
docker exec -it ros2_image_processor bash -c "source /ros2_ws/install/setup.bash && ros2 run ros2_image_processor image_processor_node"

# 터미널 2: 이미지 저장 노드
docker exec -it ros2_image_processor bash -c "source /ros2_ws/install/setup.bash && ros2 run ros2_image_processor image_saver_node"

# 터미널 3: ROS bag 재생
docker exec -it ros2_image_processor bash -c "source /ros2_ws/install/setup.bash && ros2 bag play /ros2_ws/rosbag2_2025_06_16-15_16_29 --loop"
```

## 📊 Expected Output

실행 후 다음 파일들이 생성됩니다:
- `visualization_output/processed_image_*.jpg`: 스티칭된 이미지들
- ROS 토픽: `/processed_image`, `/realsense/left/color/image_raw_throttle`, `/realsense/right/color/image_raw_throttle`

## 🐛 Troubleshooting

### 문제 1: Docker 권한 오류
```bash
sudo usermod -aG docker $USER
# 로그아웃 후 다시 로그인
```

### 문제 2: 포트 충돌
```bash
docker stop ros2_image_processor
docker rm ros2_image_processor
```

### 문제 3: 빌드 실패
```bash
docker system prune -a
./build.sh
```

## 📝 Notes

- **첫 실행**: Docker 이미지 빌드에 5-10분 소요
- **메모리**: 최소 4GB RAM 권장
- **저장공간**: 최소 2GB 여유 공간 필요
- **네트워크**: Docker 이미지 다운로드를 위해 인터넷 연결 필요

## 🎉 Success Criteria

성공적으로 실행되면:
1. Docker 컨테이너가 실행 중
2. ROS 2 노드들이 활성화
3. `visualization_output/` 폴더에 이미지 생성
4. ROS 토픽들이 발행 중

---

**💡 Tip**: 문제가 발생하면 `docker logs ros2_image_processor`로 로그를 확인하세요! 