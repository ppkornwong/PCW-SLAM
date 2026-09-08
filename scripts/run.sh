#!/usr/bin/env bash
# Source the workspace and launch the full stack. Run from anywhere inside it.
#
#   scripts/run.sh                     # SLAM backend (+ its rviz) + 4 nodes
#   scripts/run.sh slam:=false         # nodes only (SLAM already running)
#   scripts/run.sh rviz:=false
#   scripts/run.sh slam_launch:=/path/to/your_slam.launch.py
#
#   export PCWSLAM_SLAM_SETUP=/path/to/your_slam_ws/install/setup.bash    # to source it
#   export PCWSLAM_SLAM_LAUNCH=/path/to/your_slam.launch.py               # to launch it
set -e

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$WS" != "/" ] && [ ! -d "$WS/install" ]; do WS="$(dirname "$WS")"; done
[ -d "$WS/install" ] || { echo "build first: scripts/build.sh"; exit 1; }

source /opt/ros/humble/setup.bash
if [ -n "${PCWSLAM_SLAM_SETUP:-}" ] && [ -f "${PCWSLAM_SLAM_SETUP}" ]; then
    source "${PCWSLAM_SLAM_SETUP}"
fi
source "$WS/install/setup.bash"

exec ros2 launch pcwslam pcwslam.launch.py "$@"
