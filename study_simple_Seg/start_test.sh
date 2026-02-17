#!/bin/bash

# ROS 2 Image Processor Test Script
# Starts all necessary components in sequence

echo "🚀 Starting ROS 2 Image Processor Test..."
echo "========================================"

# Step 0: Clean up any existing container
echo "🧹 Step 0: Cleaning up any existing container..."
docker stop ros2_image_processor 2>/dev/null || true
docker rm ros2_image_processor 2>/dev/null || true
echo "✅ Cleanup completed"

# Step 0.5: Build Docker image to reflect code changes
echo "🔨 Step 0.5: Building Docker image to reflect code changes..."
docker build -t study_0702-ros2_image_processor .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Failed to build Docker image"
    exit 1
fi

# Step 1: Start Docker container
echo "📦 Step 1: Starting Docker container..."
docker run --rm -d --name ros2_image_processor \
    -v $PWD/rosbag2_2025_06_16-15_16_29:/ros2_ws/rosbag2_2025_06_16-15_16_29:ro \
    -v $PWD/visualization_output:/ros2_ws/visualization_output:rw \
    -v $PWD/homography_matrix.npy:/ros2_ws/homography_matrix.npy:ro \
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

# Step 2.5: Start image saver node
echo "💾 Step 2.5: Starting image saver node..."
docker exec ros2_image_processor bash -c "
    source /opt/ros/humble/setup.bash && 
    source /ros2_ws/install/setup.bash && 
    python3 src/ros2_image_processor/ros2_image_processor/image_saver_node.py &
"

if [ $? -eq 0 ]; then
    echo "✅ Image saver node started successfully"
else
    echo "❌ Failed to start image saver node"
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
echo "🧹 To clean up: docker stop ros2_image_processor" 