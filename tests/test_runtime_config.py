import sys
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "ros2_image_processor" / "ros2_image_processor"
sys.path.insert(0, str(PACKAGE_DIR))

from runtime_config import build_runtime_config  # noqa: E402


class RuntimeConfigTests(unittest.TestCase):
    def test_defaults_use_workspace_root(self):
        config = build_runtime_config(environ={})

        self.assertEqual(config["left_camera_topic"], "/realsense/right/color/image_raw_throttle")
        self.assertEqual(config["right_camera_topic"], "/realsense/left/color/image_raw_throttle")
        self.assertEqual(config["stitched_output_dir"], Path("/ros2_ws/visualization_output/stitched"))
        self.assertEqual(config["segmented_output_dir"], Path("/ros2_ws/visualization_output/segmented"))
        self.assertEqual(config["homography_matrix_path"], Path("/ros2_ws/homography_matrix.npy"))

    def test_env_and_overrides_take_precedence(self):
        config = build_runtime_config(
            overrides={
                "left_camera_topic": "/cam/a",
                "processed_image_topic": "/custom/processed",
            },
            environ={
                "ROS2_WS_ROOT": "/tmp/ws",
                "VISUALIZATION_OUTPUT_DIR": "/tmp/output",
                "HOMOGRAPHY_MATRIX_PATH": "/tmp/h.npy",
            },
        )

        self.assertEqual(config["left_camera_topic"], "/cam/a")
        self.assertEqual(config["processed_image_topic"], "/custom/processed")
        self.assertEqual(config["stitched_output_dir"], Path("/tmp/output/stitched"))
        self.assertEqual(config["segmented_output_dir"], Path("/tmp/output/segmented"))
        self.assertEqual(config["homography_matrix_path"], Path("/tmp/h.npy"))


if __name__ == "__main__":
    unittest.main()
