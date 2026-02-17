#!/bin/bash

# ROS 2 Image Processor Test Script
# Starts all necessary components in sequence

echo "🚀 Starting ROS 2 Image Processor Test..."
echo "========================================"

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
echo "🧹 Step 0: Cleaning up any existing container..."
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

# Step 2.5: Image saving is now integrated into the processor node
echo "💾 Step 2.5: Image saving is integrated into the processor node"
echo "✅ No separate image saver node needed"

# Step 3: Wait a moment for node to initialize
echo "⏳ Step 3: Waiting for node initialization..."
sleep 3

# Step 4: Start ROS bag playback with controlled rate
echo "📹 Step 4: Starting ROS bag playback with controlled rate..."
docker exec ros2_image_processor bash -c "
    source /opt/ros/humble/setup.bash && 
    timeout 180 ros2 bag play /ros2_ws/rosbag2_2025_06_16-15_16_29 --rate 1.0
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
echo "🎉 Test completed!"
echo "📁 Check visualization_output/ folder for results"
echo "🔍 Use 'docker logs ros2_image_processor' to see logs"
echo "🧹 To clean up: docker stop ros2_image_processor" 