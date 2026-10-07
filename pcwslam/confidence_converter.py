#!/usr/bin/env python3
"""
confidence_converter  (pcwslam_confidence)
=========================================

The confidence -> SLAM-parameter converter (formerly confidence_convert_2.py):

  robot_stop_conf = cmd(35%) + hole(30%) + detection(35%)   -> Gate 1 (IMU suppression)
  landmark_weight = landmark_conf * wheel_bonus * proximity  -> Gate 2 (residual correction)

  hole rebalanced 35->30 (giving detection the freed 5) because the
  hole/blind-spot signal only firms up late (once the robot is actually
  under the car), so it's intentionally not weighted as heavily as
  detection, which reflects real-time evidence. This formula is
  independent of underneath_detection.py's cross_check_detections() --
  that one scores /under_car_confidence itself and happens to also use
  30% for its own (different) hole term; the two aren't linked, the match
  is incidental.

  * blocking wait_for_service(0.2) so params are not silently dropped at startup
  * landmark source: /car_landmark_pose only, 2 s timeout, weight clamped [5, 100]
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from std_msgs.msg import Int32, Float32, Bool, String  # noqa: F401

from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

import os
import time
import csv
import math
from datetime import datetime

# CSVs always land here no matter where the node was launched from
# (override with the PCWSLAM_LOG_DIR environment variable).
LOG_DIR = os.path.expanduser(os.environ.get('PCWSLAM_LOG_DIR', '~/pcwslam_logs'))


class ConfidenceConverter(Node):

    def __init__(self):
        super().__init__('confidence_converter')

        # ── State variables ───────────────────────────────────────
        self.linear_vel  = 0.0
        self.angular_vel = 0.0

        self.hole_area        = 0.0
        self.hole_score_memory = 0.0
        self.last_hole_time   = time.time()

        self.under_car_conf   = 0.0   # 0-1 from detection node
        self.wheel_count      = 0     # confirmed wheel corners (0-4)

        # Car landmark pose from detection node
        # Format: "x,y,wheelbase,track,n_wheels,confidence"
        self.landmark_x       = 0.0
        self.landmark_y       = 0.0
        self.landmark_wb      = 0.0
        self.landmark_tr      = 0.0
        self.landmark_n_wheels= 0
        self.landmark_conf    = 0.0
        self.last_landmark_time = 0.0
        self.landmark_active  = False

        # Path
        self.path_x  = 0.0; self.path_y  = 0.0; self.path_z  = 0.0
        self.path_qx = 0.0; self.path_qy = 0.0; self.path_qz = 0.0
        self.path_qw = 1.0

        # ── Parameter clients ──────────────────────────────────────
        # One client for robot_stop_conf (existing)
        self.param_client = self.create_client(
            SetParameters, '/laserMapping/set_parameters')

        # ── Subscribers ───────────────────────────────────────────
        self.create_subscription(Twist,   '/cmd_vel',
                                 self.cmd_callback, 10)
        self.create_subscription(Float32, '/lidar_hole_area',
                                 self.hole_callback, 10)
        self.create_subscription(Float32, '/under_car_confidence',
                                 self.under_car_callback, 10)
        self.create_subscription(Float32, '/detected_wheel_count',
                                 self.wheel_count_callback, 10)
        self.create_subscription(String,  '/car_landmark_pose',
                                 self.car_landmark_callback, 10)
        self.create_subscription(Path,    '/path',
                                 self.path_callback, 10)

        # ── CSV logging ───────────────────────────────────────────
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(LOG_DIR, exist_ok=True)
        self.csv_filename = os.path.join(LOG_DIR, f'confidence_log_{ts}.csv')
        self.csv_file   = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'time',
            'linear_x', 'angular_z',
            'hole_area', 'hole_score',
            'under_car_conf', 'wheel_count',
            'landmark_active', 'landmark_x', 'landmark_y',
            'landmark_wb',     'landmark_tr', 'landmark_n_wheels',
            'landmark_conf',   'landmark_weight',
            'cmd_score', 'confidence',
            'path_x', 'path_y', 'path_z',
            'path_qx', 'path_qy', 'path_qz', 'path_qw',
        ])

        # ── Timer 10 Hz ───────────────────────────────────────────
        # per-tick status line (was a terminal INFO) -> ros2 topic echo /confidence_status
        self.status_pub = self.create_publisher(String, '/confidence_status', 10)

        self.timer = self.create_timer(0.1, self.update_confidence)

        self.get_logger().info(
            f'ConfidenceConverter v2 ready  CSV→{self.csv_filename}')

    # ── Parameter helpers ─────────────────────────────────────────

    def _set_params(self, params_dict: dict):
        """
        Send multiple parameters to laserMapping in one request.
        params_dict: {name: value}  value can be int, float, bool, str.
        """
        if not self.param_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().warn("laserMapping param service unavailable")
            return

        req = SetParameters.Request()
        for name, value in params_dict.items():
            p = Parameter()
            p.name = name
            if isinstance(value, bool):
                p.value = ParameterValue(
                    type=ParameterType.PARAMETER_BOOL,
                    bool_value=value)
            elif isinstance(value, int):
                p.value = ParameterValue(
                    type=ParameterType.PARAMETER_INTEGER,
                    integer_value=value)
            elif isinstance(value, float):
                p.value = ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE,
                    double_value=value)
            req.parameters.append(p)
        self.param_client.call_async(req)

    # ── Callbacks ─────────────────────────────────────────────────

    def cmd_callback(self, msg):
        self.linear_vel  = msg.linear.x
        self.angular_vel = msg.angular.z

    def hole_callback(self, msg):
        self.hole_area      = float(msg.data)
        self.last_hole_time = time.time()

    def under_car_callback(self, msg):
        self.under_car_conf = float(msg.data)   # 0.0–1.0

    def wheel_count_callback(self, msg):
        self.wheel_count = int(msg.data)        # 0–4

    def car_landmark_callback(self, msg):
        """
        Parse  "x,y,wheelbase,track,n_wheels,confidence"
        Published by underneath_detection when conf >= 0.40
        """
        try:
            parts = msg.data.split(',')
            if len(parts) < 6:
                return
            self.landmark_x        = float(parts[0])
            self.landmark_y        = float(parts[1])
            self.landmark_wb       = float(parts[2])
            self.landmark_tr       = float(parts[3])
            self.landmark_n_wheels = int(parts[4])
            self.landmark_conf     = float(parts[5])
            self.last_landmark_time = time.time()
            self.landmark_active   = True
        except Exception as e:
            self.get_logger().warn(f'car_landmark parse error: {e}')

    def path_callback(self, msg):
        if not msg.poses:
            return
        pose = msg.poses[-1].pose
        self.path_x  = pose.position.x
        self.path_y  = pose.position.y
        self.path_z  = pose.position.z
        self.path_qx = pose.orientation.x
        self.path_qy = pose.orientation.y
        self.path_qz = pose.orientation.z
        self.path_qw = pose.orientation.w

    # ── Main confidence loop ──────────────────────────────────────

    def update_confidence(self):
        now = time.time()

        # ── 1. CMD score (35%) ────────────────────────────────────
        # Stopped = high score (want to suppress when stationary under car)
        motion_norm = max(
            min(abs(self.linear_vel)  / 1.0, 1.0),
            min(abs(self.angular_vel) / 1.0, 1.0))
        cmd_score = 35.0 * (1.0 - motion_norm)

        # ── 2. LiDAR hole score (30%) ─────────────────────────────
        hole_clamped = max(1000.0, min(3000.0, self.hole_area))
        hole_norm    = (hole_clamped - 1000.0) / 2000.0
        target_hole  = hole_norm * 30.0

        if (now - self.last_hole_time) < 0.5:
            self.hole_score_memory = target_hole
        else:
            # Decay rate scaled down from the original 0.5/tick (at the old
            # 35-pt ceiling) to 30/35 of that, so a stale reading still
            # takes the same ~7s (70 ticks @ 10Hz) to fully decay from the
            # new 30-pt ceiling instead of decaying faster just because the
            # ceiling dropped.
            self.hole_score_memory = max(0.0, self.hole_score_memory - 0.5 * (30.0/35.0))
        hole_score = self.hole_score_memory

        # ── 3. Under-car detection score (35%) ────────────────────
        # Uses our new detection node's confidence:
        #   under_car_conf (0–1) from cross-check of BEV + side views
        #   Boosted by confirmed wheel count
        wheel_bonus    = self.wheel_count / 4.0        # 0–1
        detection_conf = self.under_car_conf * (0.7 + 0.3 * wheel_bonus)
        detection_score = detection_conf * 35.0

        # ── Final confidence ──────────────────────────────────────
        confidence = int(max(0, min(100,
            cmd_score + hole_score + detection_score)))

        # ── Landmark weight for residual correction ───────────────
        # Expires after 2s if no new landmark message
        lm_timeout = 2.0
        if (now - self.last_landmark_time) > lm_timeout:
            self.landmark_active = False

        # Weight: how strongly to pull state toward landmark
        # Scales with: detection conf × wheel count confirmation × proximity
        # Max 100.0 → used in iESKF landmark correction
        if self.landmark_active:
            dist     = math.hypot(self.landmark_x, self.landmark_y)
            prox     = max(0.1, 1.0 - dist / 3.0)   # closer = higher weight
            lm_weight = (self.landmark_conf *
                         (0.5 + 0.5 * wheel_bonus) *
                         prox * 100.0)
            lm_weight = float(max(5.0, min(100.0, lm_weight)))
        else:
            lm_weight = 0.0

        # ── Send all parameters to the SLAM backend at once ───────
        params = {
            # IMU suppression (existing)
            "mapping.robot_stop_conf": confidence,

            # Landmark correction (new)
            "mapping.landmark_detected": self.landmark_active,
            "mapping.landmark_pos_x":   float(self.landmark_x),
            "mapping.landmark_pos_y":   float(self.landmark_y),
            "mapping.landmark_weight":  lm_weight,
        }
        self._set_params(params)

        # ── CSV log ───────────────────────────────────────────────
        self.csv_writer.writerow([
            now,
            self.linear_vel, self.angular_vel,
            self.hole_area,  hole_score,
            self.under_car_conf, self.wheel_count,
            self.landmark_active,
            self.landmark_x,  self.landmark_y,
            self.landmark_wb, self.landmark_tr,
            self.landmark_n_wheels, self.landmark_conf,
            lm_weight,
            cmd_score, confidence,
            self.path_x, self.path_y, self.path_z,
            self.path_qx, self.path_qy, self.path_qz, self.path_qw,
        ])
        self.csv_file.flush()

        # ── Status topic ──────────────────────────────────────────
        lm_str = (f'LM({self.landmark_x:.2f},{self.landmark_y:.2f}) '
                  f'w={lm_weight:.1f} '
                  if self.landmark_active else 'LM=off ')
        self.status_pub.publish(String(data=
            f'CONF={confidence:3d}  '
            f'cmd={cmd_score:.0f} hole={hole_score:.0f} '
            f'det={detection_score:.0f}  '
            f'{lm_str}'
            f'wheels={self.wheel_count}/4'))

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ConfidenceConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()