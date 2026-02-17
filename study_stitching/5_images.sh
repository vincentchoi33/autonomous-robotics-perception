#!/bin/bash

# Quick Test Script - 5 Image Pairs Only
# Tests the system with just a few images

echo "🧪 Quick Test - 5 Image Pairs"
echo "============================="

# Step 0: Build Docker image (to reflect any code changes)
echo "🔨 Step 0: Building Docker image to reflect code changes..."
docker build -t study_0702-ros2_image_processor .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Failed to build Docker image"
    exit 1
fi

# Step 0.5: Clean up any existing container
echo "🧹 Step 0.5: Cleaning up any existing container..."
docker stop ros2_image_processor 2>/dev/null || true
docker rm ros2_image_processor 2>/dev/null || true
echo "✅ Cleanup completed"

# Step 1: Start Docker container
echo "📦 Step 1: Starting Docker container..."
docker run --rm -d --name ros2_image_processor \
    -v $PWD/rosbag2_2025_06_16-15_16_29:/ros2_ws/rosbag2_2025_06_16-15_16_29:ro \
    -v $PWD/visualization_output:/ros2_ws/visualization_output:rw \
    -v $PWD/homography_matrix.npy:/ros2_ws/homography_matrix.npy:ro \
    study_0702-ros2_image_processor tail -f /dev/null

if [ $? -eq 0 ]; then
    echo "✅ Docker container started"
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

# Step 2.5: Start image saver node
echo "💾 Step 2.5: Starting image saver node..."
docker exec ros2_image_processor bash -c "
    source /opt/ros/humble/setup.bash && 
    source /ros2_ws/install/setup.bash && 
    python3 src/ros2_image_processor/ros2_image_processor/image_saver_node.py &
"

if [ $? -eq 0 ]; then
    echo "✅ Image processor node started"
else
    echo "❌ Failed to start image processor node"
    exit 1
fi

# Step 3: Wait for node initialization
echo "⏳ Step 3: Waiting for node initialization..."
sleep 3

# Step 4: Start ROS bag playback for 5 seconds only
echo "📹 Step 4: Playing ROS bag for 5 seconds (≈5 image pairs)..."
docker exec ros2_image_processor bash -c "
    source /opt/ros/humble/setup.bash && 
    timeout 5 ros2 bag play /ros2_ws/rosbag2_2025_06_16-15_16_29
"

# Step 5: Wait a moment for processing to complete
echo "⏳ Step 5: Waiting for processing to complete..."
sleep 2

# Step 6: Show results
echo "📊 Step 6: Results"
echo "=================="
echo "📁 Generated images:"
ls -la visualization_output/ 2>/dev/null || echo "No images generated yet"

echo ""
echo "📡 Active ROS topics:"
docker exec ros2_image_processor bash -c "
    source /opt/ros/humble/setup.bash && 
    ros2 topic list
" 2>/dev/null || echo "No topics found"

echo ""
echo "🎉 Quick test completed!"
# docker stop ros2_image_processor 2>/dev/null || true
# docker rm ros2_image_processor 2>/dev/null || true
echo "💡 To see logs: docker logs ros2_image_processor"
echo "🧹 To clean up: docker stop ros2_image_processor" 