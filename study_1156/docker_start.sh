docker-compose down
docker-compose build --no-cache
# docker-compose up

docker exec ros2_image_processor bash -c "cd /ros2_ws && source install/setup.bash && timeout 10 python3 src/ros2_image_processor/ros2_image_processor/image_processor_node.py"