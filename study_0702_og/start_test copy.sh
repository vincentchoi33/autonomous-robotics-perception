#!/bin/bash

# ROS 2 Image Processor Test Script
# Starts all necessary components in sequence

echo "🚀 Starting ROS 2 Image Processor Test..."
echo "========================================"

# Step 1: Start Docker container
echo "📦 Step 1: Starting Docker container..."
docker run --rm -d --name ros2_image_processor \
    -v $PWD/rosbag2_2025_06_16-15_16_29:/ros2_ws/rosbag2_2025_06_16-15_16_29:ro \
    -v $PWD/visualization_output:/ros2_ws/visualization_output:rw \
    study_0702-ros2_image_processor tail -f /dev/null

if [ $? -eq 0 ]; then
    echo "✅ Docker container started successfully"
else
    echo "❌ Failed to start Docker container"
    exit 1
fi

# Step 2: Build and start image processor node
echo "🔨 Step 2: Building and starting image processor node..."
docker exec ros2_image_processor bash -c "
    source /opt/ros/humble/setup.bash && 
    cd /ros2_ws && 
    colcon build --packages-select ros2_image_processor && 
    source install/setup.bash && 
    python3 src/ros2_image_processor/ros2_image_processor/image_processor_node.py &
"

if [ $? -eq 0 ]; then
    echo "✅ Image processor node started successfully"
else
    echo "❌ Failed to start image processor node"
    exit 1
fi

# Step 3: Wait a moment for node to initialize
echo "⏳ Step 3: Waiting for node initialization..."
sleep 3

# Step 4: Start ROS bag playback
echo "📹 Step 4: Starting ROS bag playback..."
docker exec ros2_image_processor bash -c "
    source /opt/ros/humble/setup.bash && 
    ros2 bag play /ros2_ws/rosbag2_2025_06_16-15_16_29
"

echo "🎉 Test completed!"
echo "📁 Check visualization_output/ folder for results"
echo "🔍 Use 'docker logs ros2_image_processor' to see logs" 