#!/usr/bin/env python3
"""
pcwslam.all_in_one
==================

Runs all four PCW-SLAM add-on nodes in **one process**:

    - DynamicFilterOcclusionAware      (dynamic_filter.py)
    - IntensityLandmarkNode            (underneath_detection.py)
    - ConfidenceConverter              (confidence_converter.py)
    - TransmitterAlignmentNode         (transmitter_alignment.py)

This is the "one file / one command" way to launch the add-ons.  It does NOT
start the SLAM backend itself -- run the SLAM first (see README), then:

    ros2 run pcwslam pcwslam_all
    # or, with the SLAM included:
    ros2 launch pcwslam pcwslam.launch.py

Each node still creates its own OpenCV windows and writes its own CSV log to the
current directory, exactly as when run separately.  A SingleThreadedExecutor is
used on purpose so every callback (including the cv2.imshow display timers) runs
on the main thread -- the heavy work in each node already runs on its own
worker thread, so this stays responsive.

For headless / server use pass ``-p transmitter.display_window:=false`` and set
``PCWSLAM_HEADLESS=1`` (the underneath-detection view has no disable flag; run
the nodes separately and skip that one, or run under Xvfb).
"""

import sys

import rclpy
from rclpy.executors import SingleThreadedExecutor

from pcwslam.dynamic_filter import DynamicFilterOcclusionAware
from pcwslam.underneath_detection import IntensityLandmarkNode
from pcwslam.confidence_converter import ConfidenceConverter
from pcwslam.transmitter_alignment import TransmitterAlignmentNode


NODE_CLASSES = (
    DynamicFilterOcclusionAware,
    IntensityLandmarkNode,
    ConfidenceConverter,
    TransmitterAlignmentNode,
)


def main(args=None):
    rclpy.init(args=args)

    nodes = []
    for cls in NODE_CLASSES:
        try:
            nodes.append(cls())
        except Exception as exc:                       # noqa: BLE001
            print(f"[pcwslam.all_in_one] failed to start {cls.__name__}: {exc}",
                  file=sys.stderr)

    if not nodes:
        print("[pcwslam.all_in_one] no nodes started, exiting", file=sys.stderr)
        rclpy.shutdown()
        return 1

    executor = SingleThreadedExecutor()
    for n in nodes:
        executor.add_node(n)

    n0 = nodes[0]
    n0.get_logger().info(
        "pcwslam.all_in_one: running " + ", ".join(type(n).__name__ for n in nodes))

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for n in nodes:
            try:
                n.destroy_node()
            except Exception:                          # noqa: BLE001
                pass
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:                              # noqa: BLE001
            pass
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
