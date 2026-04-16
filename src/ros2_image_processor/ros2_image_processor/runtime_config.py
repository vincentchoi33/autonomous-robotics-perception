import os
from pathlib import Path


DEFAULT_WORKSPACE_ROOT = Path("/ros2_ws")


def build_runtime_config(*, overrides=None, environ=None):
    """Build runtime paths and topic names from overrides plus environment."""
    overrides = overrides or {}
    environ = environ or os.environ

    workspace_root = Path(
        overrides.get("workspace_root")
        or environ.get("ROS2_WS_ROOT")
        or DEFAULT_WORKSPACE_ROOT
    )
    visualization_root = Path(
        overrides.get("visualization_output_root")
        or environ.get("VISUALIZATION_OUTPUT_DIR")
        or (workspace_root / "visualization_output")
    )
    homography_matrix_path = Path(
        overrides.get("homography_matrix_path")
        or environ.get("HOMOGRAPHY_MATRIX_PATH")
        or (workspace_root / "homography_matrix.npy")
    )

    return {
        "left_camera_topic": overrides.get("left_camera_topic", "/realsense/right/color/image_raw_throttle"),
        "right_camera_topic": overrides.get("right_camera_topic", "/realsense/left/color/image_raw_throttle"),
        "processed_image_topic": overrides.get("processed_image_topic", "/processed_image"),
        "segmented_image_topic": overrides.get("segmented_image_topic", "/segmented_image"),
        "visualization_output_root": visualization_root,
        "stitched_output_dir": visualization_root / "stitched",
        "segmented_output_dir": visualization_root / "segmented",
        "homography_matrix_path": homography_matrix_path,
    }
