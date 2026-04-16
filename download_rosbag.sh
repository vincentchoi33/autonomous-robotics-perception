#!/bin/bash

# ROS 2 Bag File Download Script
# Downloads the rosbag2 file only if it doesn't already exist

BAG_DIR="rosbag2_2025_06_16-15_16_29"
BAG_FILE="${BAG_DIR}/rosbag2_2025_06_16-15_16_29_0.db3"
METADATA_FILE="${BAG_DIR}/metadata.yaml"
DOWNLOAD_URL="https://drive.usercontent.google.com/uc?export=download&id=1DuxtYyOjMT4nnJ7ANGfFyBd3KO8b9S1v"
ZIP_FILE="rosbag2_2025_06_16-15_16_29.zip"

echo "Checking for existing rosbag2 files..."
echo "=========================================="

# Check if bag directory and files already exist
if [ -d "$BAG_DIR" ] && [ -f "$BAG_FILE" ] && [ -f "$METADATA_FILE" ]; then
    echo "Rosbag2 files already exist:"
    echo "   Directory: $BAG_DIR"
    echo "   Database file: $(ls -lh $BAG_FILE | awk '{print $5}')"
    echo "   Metadata file: $(ls -lh $METADATA_FILE | awk '{print $5}')"
    echo ""
    echo "Ready to proceed with existing files!"
    exit 0
fi

echo "Rosbag2 files not found. Starting download..."
echo ""

# Check if we have enough disk space (need ~4GB)
REQUIRED_SPACE_KB=4000000  # 4GB in KB
AVAILABLE_SPACE_KB=$(df . | awk 'NR==2 {print $4}')

if [ "$AVAILABLE_SPACE_KB" -lt "$REQUIRED_SPACE_KB" ]; then
    echo "Error: Insufficient disk space!"
    echo "   Required: 4GB"
    echo "   Available: $(($AVAILABLE_SPACE_KB / 1024))MB"
    echo ""
    echo "Please free up some disk space and try again."
    exit 1
fi

echo "Disk space check passed: $(($AVAILABLE_SPACE_KB / 1024))MB available"
echo ""

# Check for required tools
if ! command -v curl &> /dev/null; then
    echo "Error: curl is not installed!"
    echo "Please install curl: sudo apt-get install curl"
    exit 1
fi

if ! command -v unzip &> /dev/null; then
    echo "Error: unzip is not installed!"
    echo "Please install unzip: sudo apt-get install unzip"
    exit 1
fi

echo "Required tools check passed"
echo ""

# Ask for confirmation
echo -n "📥 Do you want to download the rosbag2 file (2.5GB)? (Y/N): "
read -r confirm

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Download cancelled by user"
    exit 0
fi

echo ""
echo "Starting download..."
echo "   URL: $DOWNLOAD_URL"
echo "   File: $ZIP_FILE"
echo "   Size: ~2.5GB"
echo ""

echo "Downloading rosbag2 file..."
echo "   This may take several minutes depending on your internet connection..."
echo ""

# === file.sh에서 정상 동작하는 다운로드 코드로 대체 ===
echo "Downloading bag..."
curl -L -c cookies.txt 'https://drive.usercontent.google.com/uc?export=download&id=1DuxtYyOjMT4nnJ7ANGfFyBd3KO8b9S1v' | \
  sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
curl -L -b cookies.txt -o rosbag2_2025_06_16-15_16_29.zip \
  'https://drive.usercontent.google.com/download?id=1DuxtYyOjMT4nnJ7ANGfFyBd3KO8b9S1v&confirm='$(<confirm.txt)
rm -f confirm.txt cookies.txt
# === 이후 압축 해제 및 검증 등 기존 로직 유지 ===

# Extract the zip file
echo "   Step 3/3: Extracting files..."
unzip -q "$ZIP_FILE"

# Check if extraction was successful
if [ ! -d "$BAG_DIR" ] || [ ! -f "$BAG_FILE" ] || [ ! -f "$METADATA_FILE" ]; then
    echo "Error: Extraction failed or files are missing"
    rm -f "$ZIP_FILE"
    exit 1
fi

echo "Extraction completed"

# Clean up zip file
rm -f "$ZIP_FILE"

echo ""
echo "Rosbag2 download completed successfully!"
echo "=========================================="
echo "Directory: $BAG_DIR"
echo "Database file: $(ls -lh $BAG_FILE | awk '{print $5}')"
echo "Metadata file: $(ls -lh $METADATA_FILE | awk '{print $5}')"
echo ""
echo "You can now run the image processor:"
echo "   ./start_cpu.sh    # For CPU version"
echo "   ./start_gpu.sh    # For GPU version"
echo "" 