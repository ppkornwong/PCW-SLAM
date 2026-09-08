#!/usr/bin/env bash
# Build just this package. Run from anywhere inside the colcon workspace.
#
#   export PCWSLAM_SLAM_SETUP=/path/to/your_slam_ws/install/setup.bash   # optional
#   scripts/build.sh
set -e

# walk up to the workspace root (the dir that contains src/)
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$WS" != "/" ] && [ ! -d "$WS/src" ]; do WS="$(dirname "$WS")"; done
[ -d "$WS/src" ] || { echo "not inside a colcon workspace (no src/ found)"; exit 1; }
cd "$WS"

source /opt/ros/humble/setup.bash
if [ -n "${PCWSLAM_SLAM_SETUP:-}" ] && [ -f "${PCWSLAM_SLAM_SETUP}" ]; then
    source "${PCWSLAM_SLAM_SETUP}"
    echo "sourced SLAM backend: ${PCWSLAM_SLAM_SETUP}"
fi

colcon build --symlink-install --packages-select pcwslam "$@"
echo
echo "done. now:  source $WS/install/setup.bash"
