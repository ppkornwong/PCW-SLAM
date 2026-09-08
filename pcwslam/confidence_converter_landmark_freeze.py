#!/usr/bin/env python3
"""
confidence_convert_3_landmark_freeze.py

New standalone node. Does NOT modify confidence_convert_2.py -- but it
duplicates that file's cmd/hole/detection scoring logic (unchanged) because
this is meant to be RUN INSTEAD OF confidence_convert_2.py, not alongside
it.

*** IMPORTANT: run this INSTEAD of confidence_convert_2.py, never both at
*** the same time. Both nodes call SetParameters on the same
*** /laserMapping/set_parameters service for the same parameter names
*** (mapping.robot_stop_conf, mapping.landmark_detected, landmark_pos_x/y,
*** landmark_weight). Running both would have them overwrite each other's
*** values every ~100ms, causing the landmark correction to flap.

WHY THIS EXISTS
---------------
confidence_convert_2.py only ever sources the landmark from
/car_landmark_pose, which underneath_detection_6.py gates behind its own
cross-checked confidence crossing 0.40. Under heavy occlusion (transmitter
blocking half the ring) that confidence can sit at 0.00 for an entire pass,
so confidence_convert_2.py sends landmark_detected=False /
landmark_weight=0.0 to the SLAM the whole time -- the "2 gate" correction
mechanism (robot_stop_conf + landmark_weight) never engages exactly when
it's needed most.

chassis_freeze_overlay_node.py fits its own rectangle directly from
/dynamic_points (bypassing that gate) and, once frozen, keeps publishing a
pose re-projected via odometry on /frozen_landmark_pose
("x,y,wb,tr,n_wheels,lock_quality,age_s") for as long as the pass lasts.

This node prefers /car_landmark_pose when it's fresh (live detection is
more trustworthy than an aging frozen snapshot), and falls back to
/frozen_landmark_pose otherwise, with the resulting landmark_weight
DECAYED by how long it's been since the freeze happened
(landmark_decay_halflife_s param) -- so a long occluded pass doesn't lean
indefinitely on one snapshot taken at freeze time.

CSV logging mirrors confidence_convert_2.py's columns, with an added
`landmark_source` column ("live" / "frozen" / "none").
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from std_msgs.msg import Float32, String

from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

import time
import csv
import math
from datetime import datetime


class ConfidenceConverterWithFreeze(Node):

    def __init__(self):
        super().__init__('confidence_converter_landmark_freeze')

        self.declare_parameter('landmark_decay_halflife_s', 20.0)
        self.landmark_decay_halflife_s = self.get_parameter('landmark_decay_halflife_s').value

        # /under_car_confidence (underneath_detection_8_color_fix.py) is
        # itself squared/tanh-compressed on the way out, so in practice it
        # rarely clears ~0.7-0.75 even when everything is detected as well
        # as it ever gets (observed ceiling in test logs: ~0.69). Feeding
        # that raw into a linear 0..1 -> 0..35pt map means detection_score
        # can never reach its own max, which was the main reason overall
        # confidence couldn't climb near 100 even parked with 4/4 wheels.
        # Rescale the realistic input range to fill the full output range
        # instead of assuming raw 0..1 is achievable.
        self.declare_parameter('det_conf_ceiling', 0.75)
        self.det_conf_ceiling = self.get_parameter('det_conf_ceiling').value

        # Tick-to-tick smoothing (EMA) applied to the noisy raw sensor
        # inputs (under_car_conf, hole_norm) before scoring, so an
        # unchanged physical state doesn't swing confidence by 50 points
        # between 100ms ticks just from lidar/vision noise.
        self.declare_parameter('sensor_smoothing_tau_s', 0.4)
        self.sensor_smoothing_tau_s = self.get_parameter('sensor_smoothing_tau_s').value
        self.under_car_conf_ema = 0.0
        self.hole_norm_ema = 0.0

        # Hysteresis band for landmark_active, replacing the old hard
        # "lm_weight < 5.0 -> off" cutoff, which could flap on/off tick to
        # tick when the raw weight hovered right at the boundary.
        self.LM_WEIGHT_ON = 6.0
        self.LM_WEIGHT_OFF = 3.0
        self.landmark_active_prev = False

        # ── State variables (unchanged from confidence_convert_2.py) ──────
        self.linear_vel = 0.0
        self.angular_vel = 0.0

        self.hole_area = 0.0
        self.hole_score_memory = 0.0
        self.last_hole_time = time.time()

        self.under_car_conf = 0.0
        self.wheel_count = 0

        # Live landmark, from /car_landmark_pose (same as confidence_convert_2.py)
        self.live_x = 0.0
        self.live_y = 0.0
        self.live_wb = 0.0
        self.live_tr = 0.0
        self.live_n_wheels = 0
        self.live_conf = 0.0
        self.last_live_time = 0.0

        # Frozen landmark, from /frozen_landmark_pose (new)
        self.frozen_x = 0.0
        self.frozen_y = 0.0
        self.frozen_wb = 0.0
        self.frozen_tr = 0.0
        self.frozen_n_wheels = 0
        self.frozen_lock_quality = 0.0
        self.frozen_age_s = 0.0
        self.last_frozen_time = 0.0

        # Path
        self.path_x = 0.0; self.path_y = 0.0; self.path_z = 0.0
        self.path_qx = 0.0; self.path_qy = 0.0; self.path_qz = 0.0
        self.path_qw = 1.0

        # ── Parameter client ───────────────────────────────────────────
        self.param_client = self.create_client(
            SetParameters, '/laserMapping/set_parameters')

        # ── Subscribers ────────────────────────────────────────────────
        self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        self.create_subscription(Float32, '/lidar_hole_area', self.hole_callback, 10)
        self.create_subscription(Float32, '/under_car_confidence', self.under_car_callback, 10)
        self.create_subscription(Float32, '/detected_wheel_count', self.wheel_count_callback, 10)
        self.create_subscription(String, '/car_landmark_pose', self.live_landmark_callback, 10)
        self.create_subscription(String, '/frozen_landmark_pose', self.frozen_landmark_callback, 10)
        self.create_subscription(Path, '/path', self.path_callback, 10)

        # ── CSV logging ────────────────────────────────────────────────
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f'confidence_log_freeze_{ts}.csv'
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'time',
            'linear_x', 'angular_z',
            'hole_area', 'hole_score',
            'under_car_conf', 'wheel_count',
            'landmark_source', 'landmark_x', 'landmark_y',
            'landmark_wb', 'landmark_tr', 'landmark_n_wheels',
            'landmark_conf', 'landmark_weight',
            'cmd_score', 'confidence',
            'path_x', 'path_y', 'path_z',
            'path_qx', 'path_qy', 'path_qz', 'path_qw',
        ])

        self.timer = self.create_timer(0.1, self.update_confidence)

        self.get_logger().info(
            f'ConfidenceConverterWithFreeze ready  CSV->{self.csv_filename}  '
            f'(run instead of confidence_convert_2.py, not alongside it)')

    # ── Parameter helper (identical to confidence_convert_2.py) ──────────

    def _set_params(self, params_dict: dict):
        # wait_for_service(timeout_sec=0.2) BLOCKS this same single-threaded
        # executor for up to 200ms -- called every tick of a 10Hz timer, so
        # if the service is ever momentarily unavailable (plausible on a
        # real robot during startup races or a laserMapping restart, less
        # likely once a bag test's laserMapping has already been up for a
        # while), this alone can make the node fall behind its own period
        # and delay every other callback sharing this thread (subscriptions,
        # this same timer). service_is_ready() is a non-blocking check --
        # skip this tick and just try again next tick 100ms later instead.
        if not self.param_client.service_is_ready():
            self.get_logger().warn("laserMapping param service unavailable",
                                    throttle_duration_sec=5.0)
            return

        req = SetParameters.Request()
        for name, value in params_dict.items():
            p = Parameter()
            p.name = name
            if isinstance(value, bool):
                p.value = ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=value)
            elif isinstance(value, int):
                p.value = ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=value)
            elif isinstance(value, float):
                p.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=value)
            req.parameters.append(p)
        self.param_client.call_async(req)

    # ── Callbacks ──────────────────────────────────────────────────────

    def cmd_callback(self, msg):
        self.linear_vel = msg.linear.x
        self.angular_vel = msg.angular.z

    def hole_callback(self, msg):
        self.hole_area = float(msg.data)
        self.last_hole_time = time.time()

    def under_car_callback(self, msg):
        self.under_car_conf = float(msg.data)

    def wheel_count_callback(self, msg):
        self.wheel_count = int(msg.data)

    def live_landmark_callback(self, msg):
        """"x,y,wheelbase,track,n_wheels,confidence" -- underneath_detection_6.py, conf>=0.40 gated."""
        try:
            parts = msg.data.split(',')
            if len(parts) < 6:
                return
            self.live_x = float(parts[0])
            self.live_y = float(parts[1])
            self.live_wb = float(parts[2])
            self.live_tr = float(parts[3])
            self.live_n_wheels = int(parts[4])
            self.live_conf = float(parts[5])
            self.last_live_time = time.time()
        except (ValueError, IndexError) as e:
            self.get_logger().warn(f'car_landmark parse error: {e}')

    def frozen_landmark_callback(self, msg):
        """"x,y,wb,tr,n_wheels,lock_quality,age_s" -- chassis_freeze_overlay_node.py."""
        try:
            parts = msg.data.split(',')
            if len(parts) < 7:
                return
            self.frozen_x = float(parts[0])
            self.frozen_y = float(parts[1])
            self.frozen_wb = float(parts[2])
            self.frozen_tr = float(parts[3])
            self.frozen_n_wheels = int(parts[4])
            self.frozen_lock_quality = float(parts[5])
            self.frozen_age_s = float(parts[6])
            self.last_frozen_time = time.time()
        except (ValueError, IndexError) as e:
            self.get_logger().warn(f'frozen_landmark parse error: {e}')

    def path_callback(self, msg):
        if not msg.poses:
            return
        pose = msg.poses[-1].pose
        self.path_x = pose.position.x
        self.path_y = pose.position.y
        self.path_z = pose.position.z
        self.path_qx = pose.orientation.x
        self.path_qy = pose.orientation.y
        self.path_qz = pose.orientation.z
        self.path_qw = pose.orientation.w

    # ── Landmark source selection ─────────────────────────────────────

    def _select_landmark(self, now: float):
        """Prefer live (fresh, upstream-confirmed) detection; fall back to
        the frozen/occlusion-robust one, decaying its trust with age since
        freeze. Returns (source, x, y, wb, tr, n_wheels, conf, decay)."""
        live_fresh = (now - self.last_live_time) <= 2.0
        if live_fresh:
            return ('live', self.live_x, self.live_y, self.live_wb, self.live_tr,
                    self.live_n_wheels, self.live_conf, 1.0)

        frozen_fresh = (now - self.last_frozen_time) <= 1.0
        if frozen_fresh:
            decay = 0.5 ** (self.frozen_age_s / self.landmark_decay_halflife_s)
            return ('frozen', self.frozen_x, self.frozen_y, self.frozen_wb, self.frozen_tr,
                    self.frozen_n_wheels, self.frozen_lock_quality, decay)

        return ('none', 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)

    # ── Main confidence loop (cmd/hole/detection scoring unchanged from
    #    confidence_convert_2.py; only the landmark source changed) ──────

    def update_confidence(self):
        now = time.time()

        motion_norm = max(
            min(abs(self.linear_vel) / 1.0, 1.0),
            min(abs(self.angular_vel) / 1.0, 1.0))
        cmd_score = 35.0 * (1.0 - motion_norm)

        # EMA smoothing constant for this tick (fixed 0.1s timer period).
        ema_alpha = 1.0 - math.exp(-0.1 / max(self.sensor_smoothing_tau_s, 1e-3))

        hole_clamped = max(1000.0, min(3000.0, self.hole_area))
        hole_norm = (hole_clamped - 1000.0) / 2000.0
        self.hole_norm_ema += ema_alpha * (hole_norm - self.hole_norm_ema)
        # Blind-spot/hole weight: 30% (was 35%, then briefly folded into a
        # 50:50 cmd-vs-rest split -- reverted). The hole signal turns out to
        # become reliable once the robot has been sitting under the car for
        # a while, so it still deserves real weight, just less than cmd and
        # detection individually now get.
        target_hole = self.hole_norm_ema * 30.0

        if (now - self.last_hole_time) < 0.5:
            self.hole_score_memory = target_hole
        else:
            # decay step scaled down to match the new 30-point max (was
            # 0.5 against a 35-point max), so it still takes ~the same
            # ~7s to fully decay after the hole signal goes stale
            self.hole_score_memory = max(0.0, self.hole_score_memory - 0.5 * (30.0 / 35.0))
        hole_score = self.hole_score_memory

        self.under_car_conf_ema += ema_alpha * (self.under_car_conf - self.under_car_conf_ema)
        # Rescale the realistic 0..det_conf_ceiling input range to fill the
        # full 0..1 output range -- see det_conf_ceiling declaration above.
        det_conf_scaled = min(1.0, self.under_car_conf_ema / max(self.det_conf_ceiling, 1e-3))
        wheel_bonus = self.wheel_count / 4.0
        detection_conf = det_conf_scaled * (0.7 + 0.3 * wheel_bonus)
        # Detection weight: 35% (was 30%).
        detection_score = detection_conf * 35.0

        # robot_stop_conf feeds the SLAM's IMU-propagation suppression
        # (mapping.robot_stop_conf -> alpha=1-conf/100 damps gyro/accel and
        # effective dt, see laserMapping.cpp). Weights: cmd 35% / hole 30%
        # / detection 35% (sums to 100), a direct three-way split -- the
        # earlier 50:50 cmd-vs-(hole+detection) rescale is reverted per
        # feedback that the hole signal is trustworthy enough, once settled
        # under the car, to keep its own explicit weight rather than being
        # folded into a combined bucket. Landmark residual correction below
        # is untouched and remains independent of this value.
        confidence = int(max(0, min(100, cmd_score + hole_score + detection_score)))

        source, lx, ly, lwb, ltr, ln_wheels, lconf, decay = self._select_landmark(now)
        landmark_active = source != 'none'

        if landmark_active:
            dist = math.hypot(lx, ly)
            prox = max(0.1, 1.0 - dist / 3.0)
            lm_wheel_bonus = ln_wheels / 4.0
            lm_weight = (lconf * (0.5 + 0.5 * lm_wheel_bonus) * prox * 100.0 * decay)
            lm_weight = float(max(0.0, min(100.0, lm_weight)))
            # Schmitt trigger instead of a single hard cutoff: needs to
            # clear LM_WEIGHT_ON to turn on, but only drops back off once
            # it falls below the lower LM_WEIGHT_OFF -- avoids flapping
            # landmark_active on/off tick to tick when the raw weight
            # hovers right at the boundary.
            threshold = self.LM_WEIGHT_OFF if self.landmark_active_prev else self.LM_WEIGHT_ON
            if lm_weight < threshold:
                landmark_active = False
                lm_weight = 0.0
        else:
            lm_weight = 0.0
        self.landmark_active_prev = landmark_active

        params = {
            "mapping.robot_stop_conf": confidence,
            "mapping.landmark_detected": landmark_active,
            "mapping.landmark_pos_x": float(lx),
            "mapping.landmark_pos_y": float(ly),
            "mapping.landmark_weight": lm_weight,
        }
        self._set_params(params)

        self.csv_writer.writerow([
            now,
            self.linear_vel, self.angular_vel,
            self.hole_area, hole_score,
            self.under_car_conf, self.wheel_count,
            source, lx, ly, lwb, ltr, ln_wheels, lconf, lm_weight,
            cmd_score, confidence,
            self.path_x, self.path_y, self.path_z,
            self.path_qx, self.path_qy, self.path_qz, self.path_qw,
        ])
        self.csv_file.flush()

        self.get_logger().info(
            f'CONF={confidence:3d}  cmd35={cmd_score:.0f} hole30={hole_score:.0f} '
            f'det35={detection_score:.0f}  '
            f'LM[{source}]({lx:.2f},{ly:.2f}) w={lm_weight:.1f} decay={decay:.2f}  '
            f'wheels={self.wheel_count}/4')

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ConfidenceConverterWithFreeze()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
