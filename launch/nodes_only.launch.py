"""
nodes_only.launch.py
====================

The four PCW-SLAM add-on nodes WITHOUT starting the SLAM backend -- use this
when the SLAM is already running in another terminal / launch.

    ros2 launch pcwslam nodes_only.launch.py

Same per-node toggles and `params:` / `display:` args as pcwslam.launch.py.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("pcwslam")
    default_params = os.path.join(pkg_share, "config", "pcwslam.yaml")

    args = [
        DeclareLaunchArgument("dynamic_filter", default_value="true"),
        DeclareLaunchArgument("underneath", default_value="true"),
        DeclareLaunchArgument("confidence", default_value="true"),
        DeclareLaunchArgument("transmitter", default_value="true"),
        DeclareLaunchArgument("display", default_value="true"),
        DeclareLaunchArgument("params", default_value=default_params),
    ]
    params = LaunchConfiguration("params")
    common = dict(output="screen", parameters=[params])

    return LaunchDescription(args + [
        Node(package="pcwslam", executable="pcwslam_dynamic_filter",
             name="dynamic_filter_occlusion_aware",
             condition=IfCondition(LaunchConfiguration("dynamic_filter")), **common),
        Node(package="pcwslam", executable="pcwslam_underneath",
             name="intensity_landmark_node",
             condition=IfCondition(LaunchConfiguration("underneath")), **common),
        Node(package="pcwslam", executable="pcwslam_confidence",
             name="confidence_converter",
             condition=IfCondition(LaunchConfiguration("confidence")), **common),
        Node(package="pcwslam", executable="pcwslam_transmitter",
             name="transmitter_alignment_node",
             parameters=[params, {"display_window": LaunchConfiguration("display")}],
             output="screen",
             condition=IfCondition(LaunchConfiguration("transmitter"))),
    ])
