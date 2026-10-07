# pcwslam

ROS 2 (Humble) package. Four executables:

| executable | node name | what it does |
|---|---|---|
| `pcwslam_dynamic_filter` | `dynamic_filter_occlusion_aware` | occlusion-aware dynamic-point removal → `/dynamic_points`, `/dynamic_filter_status` |
| `pcwslam_underneath` | `intensity_landmark_node` | under-chassis detection → `/car_landmark_pose`, `/car_bbox_dims`, `/under_car_confidence`, `/detected_wheel_count`, `/lidar_hole_area`, `/wheel_debug`, `/car_debug` |
| `pcwslam_confidence` | `confidence_converter` | scores confidence, calls `SetParameters` on the mapping node (`mapping.robot_stop_conf`, `mapping.landmark_*`) → `/confidence_status` |
| `pcwslam_transmitter` | `transmitter_alignment_node` | RX (car) vs TX (robot) alignment + hang height → `/transmitter_alignment` |
| `pcwslam_all` | — | all four in one process |

## Build

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone <repo-url> pcwslam
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select pcwslam
source install/setup.bash
```

## Run

```bash
# mapping node's launch + its rviz + the 4 nodes
export PCWSLAM_SLAM_LAUNCH=/path/to/mapping_unilidar_l1.launch.py
ros2 launch pcwslam pcwslam.launch.py

# mapping node already running elsewhere:
ros2 launch pcwslam nodes_only.launch.py

# one process:
ros2 run pcwslam pcwslam_all

# one at a time:
ros2 run pcwslam pcwslam_dynamic_filter
ros2 run pcwslam pcwslam_underneath
ros2 run pcwslam pcwslam_confidence
ros2 run pcwslam pcwslam_transmitter
```

`pcwslam.launch.py` args: `slam:=false` `slam_launch:=<path>` `rviz:=false`
`dynamic_filter:=false` `underneath:=false` `confidence:=false` `transmitter:=false`
`robot_model:=false`
`display:=false` `params:=<yaml>`.

`slam:=true` includes the mapping node's own launch file unchanged, resolved from
`slam_launch:=` → `$PCWSLAM_SLAM_LAUNCH` → auto-detected from the built mapping
package. `rviz:=` is passed straight through to it.

## Robot model / TF

`robot_model.launch.py` (included by both launches, `robot_model:=false` to skip)
loads `urdf/robot.urdf.xacro` and hangs it under the SLAM pose:

```
camera_init -> aft_mapped            (Point-LIO)
aft_mapped  -> base_link             (static, inverse of the LiDAR mount)
base_link   -> chassis_link -> transmitter_link
            -> left_wheel_link, right_wheel_link, lidar_link (== aft_mapped)
```

`base_link` = ground under the drive-axle midpoint. Set the LiDAR mount with
`lidar_x:= lidar_y:= lidar_z:= lidar_yaw:=` (defaults read off the drawing).

## Inputs

Needs a mapping node publishing `/cloud_registered`, `/aft_mapped_to_init`,
`/path` and exposing `/laserMapping/set_parameters`.
`/cmd_vel` from your nav stack feeds `pcwslam_confidence` and `pcwslam_transmitter`.

## Params

`config/pcwslam.yaml`, keyed by node name. Set these for your vehicle
(`transmitter_alignment_node`): `car_offset_x/y_m` (RX offset from the bbox
centre), `robot_offset_x/y/z_m` (TX offset from odom). `pcwslam_confidence`
takes no params.
