#!/bin/bash

# ROS 2 Image Processor Build Script

set -e

echo "ROS 2 Image Processor - Build Script"
echo "===================================="

# Check if ROS 2 is sourced
if ! command -v ros2 &> /dev/null; then
    echo "❌ ROS 2 not found. Please source ROS 2 first:"
    echo "   source /opt/ros/humble/setup.bash"
    exit 1
fi

echo "✅ ROS 2 found"

# Check if we're in the right directory
if [ ! -f "src/ros2_image_processor/package.xml" ]; then
    echo "❌ Please run this script from the study_0702 directory"
    exit 1
fi

echo "✅ Package structure found"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ install/ log/

# Build the workspace
echo "🔨 Building ROS 2 workspace..."
colcon build --packages-select ros2_image_processor

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "To run the application:"
    echo "1. Source the workspace:"
    echo "   source install/setup.bash"
    echo ""
    echo "2. Launch the image processor:"
    echo "   ros2 launch ros2_image_processor image_processor.launch.py"
    echo ""
    echo "Or use Docker:"
    echo "   docker-compose up"
else
    echo "❌ Build failed!"
    exit 1
fi 