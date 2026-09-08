"""
pcwslam.launch.py -- the mapping node's launch + the 4 pcwslam nodes.

    ros2 launch pcwslam pcwslam.launch.py

args: slam:=false  slam_launch:=<path>  slam_pkg:=<name>  rviz:=false
      dynamic_filter:=false  underneath:=false  confidence:=false
      transmitter:=false  display:=false  params:=<yaml>

The mapping launch is resolved as: slam_launch:= -> $PCWSLAM_SLAM_LAUNCH ->
<slam_pkg>/launch/mapping_unilidar_l1.launch.py . It is included unchanged and
rviz:= is passed straight to it.
"""

import os

from ament_index_python.packages import (PackageNotFoundError,
                                         get_package_share_directory)
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            LogInfo, OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _resolve_slam_launch(context):
    p = LaunchConfiguration("slam_launch").perform(context).strip()
    if p:
        return p
    p = os.environ.get("PCWSLAM_SLAM_LAUNCH", "").strip()
    if p:
        return p
    pkg = LaunchConfiguration("slam_pkg").perform(context).strip()
    try:
        return os.path.join(get_package_share_directory(pkg),
                            "launch", "mapping_unilidar_l1.launch.py")
    except PackageNotFoundError:
        return ""


def _slam(context, *_a, **_kw):
    if LaunchConfiguration("slam").perform(context).lower() != "true":
        return []
    path = _resolve_slam_launch(context)
    if path and os.path.isfile(path):
        return [LogInfo(msg=f"[pcwslam] mapping launch: {path}"),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(path),
                    launch_arguments={
                        "rviz": LaunchConfiguration("rviz").perform(context)
                    }.items())]
    return [LogInfo(msg="[pcwslam] mapping launch not found -- set slam_launch:= "
                        "or $PCWSLAM_SLAM_LAUNCH, or start it yourself and use "
                        "nodes_only.launch.py")]


def generate_launch_description():
    pkg_share = get_package_share_directory("pcwslam")
    default_params = os.path.join(pkg_share, "config", "pcwslam.yaml")

    args = [
        DeclareLaunchArgument("slam", default_value="true"),
        DeclareLaunchArgument("slam_launch", default_value=""),
        DeclareLaunchArgument("slam_pkg", default_value="point_lio"),
        DeclareLaunchArgument("dynamic_filter", default_value="true"),
        DeclareLaunchArgument("underneath", default_value="true"),
        DeclareLaunchArgument("confidence", default_value="true"),
        DeclareLaunchArgument("transmitter", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("display", default_value="true"),
        DeclareLaunchArgument("params", default_value=default_params),
    ]

    params = LaunchConfiguration("params")
    display = LaunchConfiguration("display")
    common = dict(output="screen", parameters=[params])

    dynamic_filter = Node(
        package="pcwslam", executable="pcwslam_dynamic_filter",
        name="dynamic_filter_occlusion_aware",
        condition=IfCondition(LaunchConfiguration("dynamic_filter")), **common)

    underneath = Node(
        package="pcwslam", executable="pcwslam_underneath",
        name="intensity_landmark_node",
        condition=IfCondition(LaunchConfiguration("underneath")), **common)

    confidence = Node(
        package="pcwslam", executable="pcwslam_confidence",
        name="confidence_converter",
        condition=IfCondition(LaunchConfiguration("confidence")), **common)

    transmitter = Node(
        package="pcwslam", executable="pcwslam_transmitter",
        name="transmitter_alignment_node",
        parameters=[params, {"display_window": display}],
        output="screen",
        condition=IfCondition(LaunchConfiguration("transmitter")))

    return LaunchDescription(args + [
        LogInfo(msg="[pcwslam] starting stack: SLAM backend + dynamic_filter + "
                    "underneath + confidence + transmitter"),
        OpaqueFunction(function=_slam),
        dynamic_filter, underneath, confidence, transmitter,
    ])
