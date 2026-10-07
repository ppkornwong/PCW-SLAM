"""
robot_model.launch.py -- robot URDF + the TF that hangs it under the SLAM pose.

    ros2 launch pcwslam robot_model.launch.py

Point-LIO publishes  camera_init -> aft_mapped  (aft_mapped = the LiDAR/IMU).
This adds           aft_mapped -> base_link     (static, inverse of the mount)
and robot_state_publisher adds base_link -> chassis_link, left/right_wheel_link,
lidar_link, transmitter_link. lidar_link coincides with aft_mapped.

So "where is the robot" is just  tf2 lookup camera_init -> base_link.

args: lidar_x lidar_y lidar_z lidar_yaw   LiDAR position in base_link (m, rad)
      slam_body_frame:=aft_mapped          Point-LIO's odom_child_frame_id
"""

import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def _nodes(context, *_a, **_kw):
    lc = {k: LaunchConfiguration(k).perform(context)
          for k in ("lidar_x", "lidar_y", "lidar_z", "lidar_yaw", "slam_body_frame")}
    x, y, z, yaw = (float(lc[k]) for k in ("lidar_x", "lidar_y", "lidar_z", "lidar_yaw"))

    urdf = os.path.join(get_package_share_directory("pcwslam"),
                        "urdf", "robot.urdf.xacro")
    robot_description = xacro.process_file(
        urdf, mappings={k: lc[k] for k in ("lidar_x", "lidar_y", "lidar_z", "lidar_yaw")}
    ).toxml()

    # base_link in the lidar frame = inverse of (Rz(yaw), t)
    c, s = math.cos(yaw), math.sin(yaw)
    ix, iy, iz = -(c * x + s * y), -(-s * x + c * y), -z

    return [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             name="pcw_robot_state_publisher", output="screen",
             parameters=[{"robot_description": robot_description}]),
        Node(package="tf2_ros", executable="static_transform_publisher",
             name="slam_body_to_base_link", output="screen",
             arguments=["--x", str(ix), "--y", str(iy), "--z", str(iz),
                        "--yaw", str(-yaw), "--pitch", "0", "--roll", "0",
                        "--frame-id", lc["slam_body_frame"],
                        "--child-frame-id", "base_link"]),
    ]


def generate_launch_description():
    return LaunchDescription([
        # MEASURE on the robot -- defaults read off the drawing
        DeclareLaunchArgument("lidar_x", default_value="0.604"),
        DeclareLaunchArgument("lidar_y", default_value="0.0"),
        DeclareLaunchArgument("lidar_z", default_value="0.122"),
        DeclareLaunchArgument("lidar_yaw", default_value="0.0"),  # mount is square now (offset fixed mechanically)
        DeclareLaunchArgument("slam_body_frame", default_value="aft_mapped"),
        OpaqueFunction(function=_nodes),
    ])
