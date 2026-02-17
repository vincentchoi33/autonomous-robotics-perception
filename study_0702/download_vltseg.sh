#!/bin/bash

echo "🚀 VLTSeg 설정 파일 및 체크포인트 다운로드"
echo "=========================================="

# 디렉토리 생성
mkdir -p vltseg_configs
mkdir -p checkpoints

# VLTSeg 저장소 클론
echo "📥 VLTSeg 저장소 클론 중..."
if [ ! -d "VLTSeg" ]; then
    git clone https://github.com/VLTSeg/VLTSeg.git
else
    echo "VLTSeg 저장소가 이미 존재합니다."
fi

# 설정 파일 복사
echo "📋 설정 파일 복사 중..."
cp -r VLTSeg/configs/* vltseg_configs/
cp VLTSeg/configs/mask2former_evaclip_2xb8_5k_gta2cityscapes.py vltseg_configs/

# 체크포인트 다운로드
echo "💾 체크포인트 다운로드 중..."
cd checkpoints

# # GTA_1 체크포인트 다운로드 (가장 성능이 좋은 모델)
# echo "📥 GTA_1 체크포인트 다운로드 중..."
# wget -O vltseg_checkpoint_gta_1.pth "https://zenodo.org/records/14766160/files/vltseg%5Fcheckpoint%5Fgta%5F1.pth?download=1"

# Cityscapes_1 체크포인트 다운로드 (Cityscapes에서 훈련된 모델)
echo "📥 Cityscapes_1 체크포인트 다운로드 중..."
wget -O vltseg_checkpoint_cityscapes_1.pth "https://zenodo.org/records/14766160/files/vltseg%5Fcheckpoint%5Fcityscapes%5F1.pth?download=1"

cd ..

echo "✅ VLTSeg 설정 완료!"
echo "📁 설정 파일: vltseg_configs/"
echo "📁 체크포인트: checkpoints/"
echo ""
echo "사용 가능한 체크포인트:"
# echo "- vltseg_checkpoint_gta_1.pth (GTA → Cityscapes, mIoU: 65.23)"
echo "- vltseg_checkpoint_cityscapes_1.pth (Cityscapes, mIoU: 84.83)" 