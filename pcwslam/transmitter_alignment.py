#!/usr/bin/env python3
"""
transmitter_alignment_node.py

New, standalone node. Does not modify underneath_detection_8_color_fix.py's
detection logic -- only consumes the new /car_bbox_dims topic it now
publishes (cx, cy, wb, tr of the detected chassis rectangle, un-gated by
the separate conf_score>=0.40 check /car_landmark_pose uses, since this
just needs the footprint, not the full cross-checked confidence).

LABEL CONVENTION (confirmed): the point on the ROBOT (odom + offset) is
TX; the point detected UNDERNEATH THE CAR (bbox center + offset) is RX.

WHAT THIS DOES
--------------
1. Estimates the CAR's RX position as the bounding box CENTER (+ a
   configurable offset, since it's very unlikely to sit exactly at the
   geometric center -- that offset defaults to (0,0) and is meant to be
   corrected once you've physically measured where it actually sits
   relative to the chassis rectangle's center).

2. Estimates the ROBOT's own TX position as robot odom (/path) plus a
   second configurable fixed mechanical offset (where it sits relative to
   base_link) -- also defaults to (0,0), same "fix after you measure it"
   idea.

3. Compares the two: publishes and visualizes the misalignment between
   them, in the robot's own forward/lateral axes (more directly actionable
   for docking than world-frame x/y).

4. Finds the LOWEST z point within z_sample_radius_m of the CAR's detected
   (RX) (x,y) (from /dynamic_points) -- not the whole detected footprint,
   which needs the box's true orientation (unavailable, see below) to
   reliably contain any points at all. RX is the actual point under the
   car where contact happens, so this only becomes available once a car
   has been detected (unlike sampling at the robot's own always-known TX
   point). height_differ = that height minus robot_offset_z_m (the
   robot's own lift-mechanism height, since that's what would need to
   move): negative means the lowest point is below the robot's current
   height (robot needs to move UP), positive means it's already above.
   Since the robot's own lift actuator has no
   position feedback (confirmed: the CAN motor scripts in this workspace
   are write-only, and the linear actuator's micro-ROS firmware only
   takes up/down commands, not a measured position), this is reported as
   a measured height difference, not a claimed push distance.

ORIENTATION
-----------
/car_bbox_dims now carries angle_rad as a 5th field (previously it didn't
-- the same gap /car_landmark_pose has, since the rectangle's orientation
is used internally by underneath_detection_8_color_fix.py's
_CHASSIS_SMOOTHER but was being dropped before publishing anywhere).
detect_bev() converts the fit's pixel-space angle into robot-frame yaw by
mapping a second point along the rect's axis through the exact same
px->world formulas already used for the center (car_x/car_y), rather than
hand-deriving the pixel-to-metric rotation relationship -- verified
empirically against synthetic rotated rectangles run through the real
create_intensity_bev/_find_car_rect_intensity pipeline, angle recovered to
within ~0.15deg across 0-120deg (mod the rectangle's inherent 180deg
ambiguity). This node applies that rotation to both the drawn bbox outline
and to car_off (RX's offset from bbox center, which is defined in the
CAR's own frame and needs to rotate with it, not stay robot-frame-fixed).
Backward compatible: falls back to angle_rad=0 if a 4-field message ever
arrives (e.g. an older publisher).

TOPICS
------
Subscribes: /car_bbox_dims (String "cx,cy,wb,tr,angle_rad"), /path (nav_msgs/Path),
            /dynamic_points (PointCloud2, BEST_EFFORT), /cmd_vel (Twist)
Publishes:  /transmitter_alignment
            (String "x_differ_m,y_differ_m,dist_m,lowest_z_m,height_differ_m,bbox_age_s,lowest_z_source,z_frozen")
            x_differ/y_differ = RX (car) position minus TX (robot) position.
            Published every tick regardless of bbox freshness -- lowest_z
            is sticky (see cloud_cb) and shouldn't be withheld just because
            the bbox it's anchored to has gone stale. lowest_z_source is
            "near" (a point actually fell within z_sample_radius_m of RX),
            "fallback" (nothing did this scan, so this is the single
            closest point by (x,y) instead -- less precise), or "none" (no
            reading has landed yet at all). z_frozen=1 once neither
            /cmd_vel nor the robot's own pose has shown any motion for
            z_settle_time_s -- lowest_z locks to a constant value at that
            point instead of continuing to drift on point-cloud noise, so
            the linear actuator has an actual still target to converge on;
            z_frozen=0 while still settling right
            after a real reposition.

RUN
---
    python3 dynamic_detect_6_occlusion_aware.py
    python3 underneath_detection_8_color_fix.py
    python3 transmitter_alignment_node.py

Once you've physically measured the real offsets, set them via parameters
rather than editing this file, e.g.:
    ros2 run ... transmitter_alignment_node.py --ros-args \\
        -p car_offset_x_m:=0.12 -p car_offset_y_m:=-0.03 \\
        -p robot_offset_x_m:=0.20 -p robot_offset_y_m:=0.0
"""

import time
import math
import queue
import csv
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import Twist


# Reject any /car_bbox_dims reading with wb below this -- a real car's
# wheelbase doesn't read under 2m; a box that small is noise/a bad fit.
MIN_PLAUSIBLE_WB_M = 2.0

# /cmd_vel is never exactly 0.0 at rest (float noise from the teleop/nav
# stack), so treat anything under this as "not being commanded to move"
# rather than requiring bit-exact zero.
CMD_VEL_EPS = 0.02


def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def world_to_robot(points_xy: np.ndarray, pose_xy: np.ndarray, yaw: float) -> np.ndarray:
    """Same convention as every other node in this pipeline
    (underneath_detection's shift_cloud_to_robot, chassis_freeze_overlay_node,
    landmark_cache_node): robot at origin, facing +X."""
    return (points_xy - pose_xy) @ rot2d(-yaw).T


class TransmitterAlignmentNode(Node):
    def __init__(self):
        super().__init__("transmitter_alignment_node")

        # Named by physical location (robot / car), not by RF role, so the
        # TX/RX label convention can't get tangled up with the parameter
        # names themselves.
        # Measured RX offset from the bbox center, in the CAR'S OWN frame
        # (x along wheelbase/length, y along track/width -- see the
        # detect_bev local_corners convention: half=[wb/2, tr/2]).
        #   x: receiver splits the wheelbase rear:front = 0.45:0.55 (i.e.
        #      shifted toward the rear of center), measured against a
        #      2.75m wheelbase: 0.45*2.75 - 2.75/2 = 1.2375 - 1.375 =
        #      -0.1375m. Negative = toward the back, matching "back" being
        #      the low/negative end of local x (same convention as the
        #      prior 125cm-from-back measurement).
        #      CAVEAT: /car_bbox_dims' angle_rad has a documented 180deg
        #      fit ambiguity (see this file's ORIENTATION docstring), so
        #      "back" can silently flip to "front" between passes -- this
        #      offset only stays correct as long as that ambiguity happens
        #      to resolve the same way it did when this was measured. If
        #      RX ever shows up at the wrong end of the car in the
        #      display, that's almost certainly why.
        #   y: confirmed against the live display -- the receiver sits on
        #      the centerline, no lateral offset. (Earlier guess of tr/2
        #      put it visibly off to the left; that was wrong, not just a
        #      sign flip.)
        self.declare_parameter("car_offset_x_m", -0.1375)  # car's RX offset from bbox center
        self.declare_parameter("car_offset_y_m", 0.0)
        self.declare_parameter("robot_offset_x_m", -0.30)   # robot's TX offset from odom
        self.declare_parameter("robot_offset_y_m", 0.01)
        self.declare_parameter("robot_offset_z_m", 0.06)  # rough guess -- refine once measured
        self.declare_parameter("z_sample_radius_m", 0.15)
        # lowest_z is a min()/nearest-point extreme-value statistic
        # recomputed fresh from a noisy, resampled point cloud every scan
        # -- while the robot sits still that made it visibly jitter tick
        # to tick even though the car underneath isn't moving, which is no
        # target a linear actuator can settle on. EMA-smooth it instead of
        # publishing the raw per-scan value directly; still fully
        # self-corrects (large dt after a gap -> alpha->1 -> snaps to the
        # fresh reading) rather than latching permanently onto one outlier.
        self.declare_parameter("z_smoothing_tau_s", 0.5)
        # Even smoothed, an EMA never actually stops moving -- it just
        # moves slower, so it's still not a "still" target for the linear
        # actuator to converge on. While the robot's own LATCHED pose (see
        # path_cb's jitter gate) hasn't changed -- i.e. the robot genuinely
        # isn't moving, so the car underneath it isn't either -- hold
        # lowest_z rock-still instead of continuing to chase per-scan
        # point-cloud noise. Only resume updating once the robot's pose
        # actually changes (a real reposition, not jitter). This window is
        # how long to keep EMA-converging right after such a move before
        # locking -- long enough to settle past scan noise, short enough
        # not to feel laggy.
        self.declare_parameter("z_settle_time_s", 1.0)
        self.declare_parameter("bbox_timeout_s", 1.0)
        self.declare_parameter("pose_latch_trans_m", 0.10)
        self.declare_parameter("pose_latch_rot_rad", 0.05)
        self.declare_parameter("canvas_px", 500)
        self.declare_parameter("scale_px_per_m", 80.0)
        self.declare_parameter("display_window", True)

        gp = self.get_parameter
        self.car_off = np.array([gp("car_offset_x_m").value,
                                  gp("car_offset_y_m").value])
        self.robot_off = np.array([gp("robot_offset_x_m").value,
                                    gp("robot_offset_y_m").value])
        self.robot_off_z = gp("robot_offset_z_m").value
        self.z_sample_radius_m = gp("z_sample_radius_m").value
        self.z_smoothing_tau_s = gp("z_smoothing_tau_s").value
        self.z_settle_time_s = gp("z_settle_time_s").value
        self.bbox_timeout_s = gp("bbox_timeout_s").value
        self.pose_latch_trans = gp("pose_latch_trans_m").value
        self.pose_latch_rot = gp("pose_latch_rot_rad").value
        self.canvas_px = int(gp("canvas_px").value)
        self.scale = gp("scale_px_per_m").value
        self.display_window = bool(gp("display_window").value)

        self.pose: Optional[Tuple[float, float, float]] = None

        self.bbox_cxcy: Optional[np.ndarray] = None
        self.bbox_wb = 0.0
        self.bbox_tr = 0.0
        self.bbox_angle_rad = 0.0
        self.last_bbox_time = 0.0

        self.lowest_z: Optional[float] = None
        self.lowest_z_time = 0.0
        self.lowest_z_source: Optional[str] = None   # "near" or "fallback"
        self.z_frozen = False
        # "Last time any motion signal fired" rather than a single pose
        # comparison -- fed by BOTH /cmd_vel (instant: fires the moment
        # the robot is commanded to move, no SLAM lag) and the latched
        # pose actually changing (catches real motion even if a cmd_vel
        # message was momentarily missed/zero). Freezing only kicks in
        # once z_settle_time_s has passed with NEITHER signal firing.
        self._last_motion_time = time.time()
        self.linear_vel = 0.0
        self.angular_vel = 0.0

        qos_be = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(String, "/car_bbox_dims", self.bbox_cb, 10)
        self.create_subscription(Path, "/path", self.path_cb, 10)
        self.create_subscription(PointCloud2, "/dynamic_points", self.cloud_cb, qos_be)
        self.create_subscription(Twist, "/cmd_vel", self.cmd_vel_cb, 10)

        self.pub_align = self.create_publisher(String, "/transmitter_alignment", 10)

        # CSV log -- every tick, not just when a bbox is fresh/available,
        # so the file captures the full timeline (including gaps/dropouts)
        # rather than only "success" rows -- needed to compute things like
        # detection uptime %, not just the values during good detections.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"transmitter_alignment_log_{ts}.csv"
        self.csv_file = open(self.csv_filename, "w", newline="")
        self.csv_file.write(
            f"# car_offset_m={tuple(self.car_off)} robot_offset_m={tuple(self.robot_off)} "
            f"robot_offset_z_m={self.robot_off_z} z_sample_radius_m={self.z_sample_radius_m} "
            f"min_plausible_wb_m={MIN_PLAUSIBLE_WB_M} bbox_timeout_s={self.bbox_timeout_s}\n")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "time", "elapsed_s",
            "pose_x_m", "pose_y_m", "pose_yaw_rad",
            "bbox_present", "bbox_fresh", "bbox_age_s",
            "bbox_x_m", "bbox_y_m", "bbox_wb_m", "bbox_tr_m", "bbox_angle_rad",
            "rx_x_m", "rx_y_m", "tx_x_m", "tx_y_m",
            "x_differ_m", "y_differ_m", "dist_m",
            "lowest_z_present", "lowest_z_m", "lowest_z_source", "z_frozen", "lowest_z_age_s", "height_differ_m",
        ])
        self._log_t_start = time.time()

        self._display_queue: "queue.Queue" = queue.Queue(maxsize=1)
        if self.display_window:
            self.create_timer(1.0 / 30.0, self._display_tick)
        self.create_timer(0.1, self._compute_and_publish)

        self.get_logger().info(
            "transmitter_alignment_node up  "
            f"car_offset(RX)={tuple(self.car_off)}  robot_offset(TX)={tuple(self.robot_off)}  "
            "(defaults are (0,0) -- set via params once you've measured the real offsets)  "
            f"CSV->{self.csv_filename}")

    # ---- subscriptions ----

    def path_cb(self, msg: Path):
        if not msg.poses:
            return
        p = msg.poses[-1].pose
        yaw = yaw_from_quat(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
        new_pose = (p.position.x, p.position.y, yaw)
        if self.pose is None:
            self.pose = new_pose
        else:
            dx = new_pose[0] - self.pose[0]
            dy = new_pose[1] - self.pose[1]
            dyaw = abs(math.atan2(math.sin(new_pose[2] - self.pose[2]),
                                   math.cos(new_pose[2] - self.pose[2])))
            if math.hypot(dx, dy) >= self.pose_latch_trans or dyaw >= self.pose_latch_rot:
                self.pose = new_pose
                self._last_motion_time = time.time()   # real (non-jitter) move -- see cloud_cb

    def cmd_vel_cb(self, msg: Twist):
        self.linear_vel = msg.linear.x
        self.angular_vel = msg.angular.z
        if abs(self.linear_vel) > CMD_VEL_EPS or abs(self.angular_vel) > CMD_VEL_EPS:
            self._last_motion_time = time.time()

    def bbox_cb(self, msg: String):
        try:
            parts = [float(v) for v in msg.data.split(",")]
            cx, cy, wb, tr = parts[:4]
            # angle_rad is the 5th field as of underneath_detection_8_color_fix.py's
            # /car_bbox_dims update; fall back to 0 (axis-aligned) if an
            # older publisher without it is somehow still running.
            angle_rad = parts[4] if len(parts) > 4 else 0.0
        except (ValueError, IndexError):
            return
        # Plausibility gate: reject anything with wb < MIN_PLAUSIBLE_WB_M
        # outright -- don't update state, don't plot, don't publish off of
        # it. A real car's wheelbase doesn't read under 2m; a box that
        # small is noise/a bad fit, not a smaller car.
        if wb < MIN_PLAUSIBLE_WB_M:
            return
        self.bbox_cxcy = np.array([cx, cy])
        self.bbox_wb = wb
        self.bbox_tr = tr
        self.bbox_angle_rad = angle_rad
        self.last_bbox_time = time.time()

    def cloud_cb(self, msg: PointCloud2):
        # Sample near the CAR's detected (RX) (x,y), not the whole car
        # footprint and not the robot's own TX point -- RX is the actual
        # point under the car where contact happens, so that's the
        # meaningful place to measure hang height. Trade-off vs sampling at
        # the always-known TX point: this now needs a detected bbox first
        # (self.bbox_cxcy is not None), so there's nothing to sample until
        # a car has actually been detected.
        if self.pose is None or self.bbox_cxcy is None:
            return
        points = self._cloud_to_xyzi(msg)
        if points.shape[0] == 0:
            return
        # /dynamic_points is in the SLAM WORLD frame (same as /cloud_registered
        # everywhere else in this pipeline) -- RX's position is robot-relative,
        # so points have to be transformed into the robot frame first. This
        # was the actual bug behind "lowest near rx never updates" before:
        # world coordinates were being compared directly against a
        # robot-relative offset, which only ever matches by coincidence
        # (robot sitting at world origin with zero yaw).
        robot_xy = world_to_robot(points[:, :2], np.array(self.pose[:2]), self.pose[2])
        rx_point = self.bbox_cxcy + self.car_off
        dx = robot_xy[:, 0] - rx_point[0]
        dy = robot_xy[:, 1] - rx_point[1]
        dist = np.hypot(dx, dy)
        near = dist <= self.z_sample_radius_m
        if np.any(near):
            # Confident reading: lowest point within the tight sample
            # radius -- most representative of the actual contact point
            # directly under RX.
            raw_z = float(points[near, 2].min())
            self.lowest_z_source = "near"
        else:
            # Nothing fell inside z_sample_radius_m this scan (occlusion,
            # RX offset not dialed in yet, a sparse return right at the
            # edge of the blind cone, etc.) -- rather than leave lowest_z
            # stuck at n/a, fall back to the single closest point in the
            # whole cloud by (x,y) distance to RX and use ITS z. A rough
            # reading from the nearest available ground truth beats no
            # reading at all; lowest_z_source flags which kind this is so
            # consumers/logs can tell them apart.
            idx = int(np.argmin(dist))
            raw_z = float(points[idx, 2])
            self.lowest_z_source = "fallback"

        # See z_settle_time_s / _last_motion_time declarations: hold
        # lowest_z rock-still once BOTH /cmd_vel and the latched pose have
        # been quiet for a full settle window, instead of letting an EMA
        # drift forever on point-cloud noise. cmd_vel is checked first
        # because it's the more direct, lag-free "is the robot actually
        # being commanded to move" signal (SLAM pose can take a moment to
        # confirm a move the robot's already making); the pose-change
        # check in path_cb backstops it in case a cmd_vel message is ever
        # missed while the robot is still genuinely in motion.
        now_t = time.time()
        if now_t - self._last_motion_time < self.z_settle_time_s:
            # Actively moving, or still within the settle window after the
            # last motion -- keep EMA-converging past scan-to-scan noise.
            if self.lowest_z is None:
                self.lowest_z = raw_z
            else:
                dt = max(now_t - self.lowest_z_time, 0.0)
                alpha = 1.0 - math.exp(-dt / max(self.z_smoothing_tau_s, 1e-3))
                self.lowest_z += alpha * (raw_z - self.lowest_z)
            self.lowest_z_time = now_t
            self.z_frozen = False
        else:
            # Settled: no motion signal has fired since the window
            # elapsed -- this is the value the linear actuator should
            # target. Leave it (and lowest_z_time) untouched so it stays
            # genuinely constant; lowest_z_age_s growing is the visible
            # signal that it's frozen, not stale.
            self.z_frozen = True

    # ---- main computation ----

    def _compute_and_publish(self):
        now = time.time()

        pose_x = self.pose[0] if self.pose is not None else float("nan")
        pose_y = self.pose[1] if self.pose is not None else float("nan")
        pose_yaw = self.pose[2] if self.pose is not None else float("nan")

        bbox_present = self.bbox_cxcy is not None
        bbox_fresh = bbox_present and (now - self.last_bbox_time) <= self.bbox_timeout_s
        bbox_age = (now - self.last_bbox_time) if bbox_present else float("nan")

        if bbox_present:
            # car_off is where RX sits relative to the bbox center IN THE
            # CAR'S OWN frame (e.g. "10cm toward the front along the car's
            # length"), so it has to rotate with the detected box
            # orientation, not stay fixed in the robot's frame -- otherwise
            # it would point in a fixed robot-relative direction regardless
            # of which way the car is actually facing.
            car_point_robot = self.bbox_cxcy + rot2d(self.bbox_angle_rad) @ self.car_off  # RX
            robot_point_robot = self.robot_off                                            # TX
            misalign = car_point_robot - robot_point_robot        # RX - TX
            x_differ, y_differ = float(misalign[0]), float(misalign[1])
            misalign_dist = float(np.hypot(x_differ, y_differ))
        else:
            car_point_robot = None
            x_differ = y_differ = misalign_dist = float("nan")

        # No freshness timeout on lowest_z (unlike bbox_fresh above): once a
        # height reading near RX has ever been captured, keep showing/
        # logging it rather than blanking to nan between updates -- the car
        # underside isn't moving frame to frame while docked, so the last
        # good reading stays valid and this should have data all the time
        # once the first reading lands.
        lz_present = self.lowest_z is not None
        lz = self.lowest_z if lz_present else float("nan")
        lz_age = (now - self.lowest_z_time) if lz_present else float("nan")
        height_differ = (self.lowest_z - self.robot_off_z) if lz_present else float("nan")

        self.csv_writer.writerow([
            f"{now:.3f}", f"{now - self._log_t_start:.2f}",
            f"{pose_x:.4f}", f"{pose_y:.4f}", f"{pose_yaw:.4f}",
            int(bbox_present), int(bbox_fresh), f"{bbox_age:.2f}",
            f"{self.bbox_cxcy[0]:.4f}" if bbox_present else "",
            f"{self.bbox_cxcy[1]:.4f}" if bbox_present else "",
            f"{self.bbox_wb:.4f}" if bbox_present else "",
            f"{self.bbox_tr:.4f}" if bbox_present else "",
            f"{self.bbox_angle_rad:.5f}" if bbox_present else "",
            f"{car_point_robot[0]:.4f}" if car_point_robot is not None else "",
            f"{car_point_robot[1]:.4f}" if car_point_robot is not None else "",
            f"{self.robot_off[0]:.4f}", f"{self.robot_off[1]:.4f}",
            f"{x_differ:.4f}", f"{y_differ:.4f}", f"{misalign_dist:.4f}",
            int(lz_present), f"{lz:.4f}" if lz_present else "",
            self.lowest_z_source if lz_present else "",
            int(self.z_frozen) if lz_present else "",
            f"{lz_age:.2f}" if lz_present else "",
            f"{height_differ:.4f}" if lz_present else "",
        ])
        self.csv_file.flush()

        # Always publish, even when bbox isn't fresh (or never arrived) --
        # lowest_z has its own sticky/fallback logic in cloud_cb and
        # shouldn't be held back from consumers just because the bbox
        # reading it was anchored to has since gone stale; x_differ/
        # y_differ/misalign_dist/bbox_age simply read as nan in that case,
        # same as they already do in the CSV above.
        #
        # "x_differ,y_differ,dist,lowest_z,height_differ,bbox_age,lowest_z_source,z_frozen" --
        # lowest_z is measured near RX (car); height_differ = lowest_z -
        # robot_offset_z_m (the robot's own lift-mechanism height):
        # negative means the lowest point is BELOW the robot's current
        # height (robot needs to move UP to reach it), positive means it's
        # already above. lowest_z_source is "near" (a real sample fell
        # within z_sample_radius_m of RX) or "fallback" (nothing did --
        # this is the single closest point by (x,y) instead, less precise)
        # or "none" if no reading has ever landed. z_frozen=1 means the
        # robot's pose hasn't changed in over z_settle_time_s -- lowest_z
        # is holding rock-still at its settled value (the intended target
        # for the linear actuator); z_frozen=0 means it's still actively
        # converging (robot just moved, or hasn't settled yet).
        msg = String(data=f"{x_differ:.4f},{y_differ:.4f},"
                          f"{misalign_dist:.4f},{lz:.4f},{height_differ:.4f},"
                          f"{bbox_age:.2f},{self.lowest_z_source or 'none'},"
                          f"{int(self.z_frozen)}")
        self.pub_align.publish(msg)

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

    # ---- visualization ----

    def _to_px(self, x_robot: float, y_robot: float) -> Tuple[int, int]:
        """Same axis convention as underneath_detection_8_color_fix.py's BEV
        panels (create_bev_dilated/create_intensity_bev): robot forward
        (+X) is drawn UP, robot left (+Y) is drawn LEFT. Everything this
        node tracks (bbox_cxcy, car_off, robot_off) is already
        robot-relative, so this takes robot-frame coordinates directly --
        no world-frame round trip needed, and critically no dependence on
        robot heading, so the picture doesn't rotate around as the robot
        turns the way a world-frame-oriented view would."""
        S = self.canvas_px
        u = int(round(-y_robot * self.scale + S / 2))
        v = int(round(S / 2 - x_robot * self.scale))
        return u, v

    def _display_tick(self):
        frame = self._draw_frame()
        if frame is None:
            return
        try:
            self._display_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._display_queue.put_nowait(frame)
        except queue.Full:
            pass
        try:
            f = self._display_queue.get_nowait()
            cv2.imshow("Transmitter Alignment", f)
            cv2.waitKey(1)
        except queue.Empty:
            pass

    def _draw_frame(self) -> Optional[np.ndarray]:
        img = np.zeros((self.canvas_px, self.canvas_px, 3), dtype=np.uint8)

        # robot marker + heading arrow -- always at the robot-frame origin
        # facing "up" (local +X), matching the BEV panel's own robot
        # marker. No pose/yaw needed for this anymore: everything drawn
        # here is already robot-relative, so this view doesn't rotate
        # around as the robot turns the way the old world-frame version did.
        c_px = self._to_px(0.0, 0.0)
        # cv2.drawMarker(img, c_px, (60, 220, 60), cv2.MARKER_TRIANGLE_UP, 14, 2)
        # cv2.arrowedLine(img, c_px, self._to_px(0.3, 0.0), (60, 220, 60), 1, tipLength=0.3)

        # Legend block, fixed in the top-right corner -- like a plot legend,
        # not attached to the markers at all, so it never collides with
        # anything on the plot regardless of where TX/RX end up.
        # TX (robot) keeps the (255,180,0) swatch, RX (car) keeps
        # (255,0,220) -- same colors/positions as before, only the text
        # pairing changed per the confirmed convention (robot=TX, car=RX).
        leg_x = self.canvas_px - 130
        cv2.rectangle(img, (leg_x, 10), (leg_x + 14, 22), (255, 180, 0), -1)
        cv2.putText(img, "TX (robot)", (leg_x + 20, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 180, 0), 1, cv2.LINE_AA)
        cv2.rectangle(img, (leg_x, 30), (leg_x + 14, 42), (255, 0, 220), -1)
        cv2.putText(img, "RX (car)", (leg_x + 20, 41),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 0, 220), 1, cv2.LINE_AA)

        # robot's TX marker (robot + fixed offset) -- plain marker, no
        # attached label, per the legend above.
        robot_pt_px = self._to_px(*self.robot_off)
        cv2.drawMarker(img, robot_pt_px, (255, 180, 0), cv2.MARKER_DIAMOND, 12, 2)

        now = time.time()
        bbox_fresh = self.bbox_cxcy is not None and (now - self.last_bbox_time) <= self.bbox_timeout_s

        if bbox_fresh:
            R_box = rot2d(self.bbox_angle_rad)
            rx_xy = self.bbox_cxcy + R_box @ self.car_off

            # bbox footprint outline, drawn at its actual detected
            # orientation (angle_rad from /car_bbox_dims, verified against
            # underneath_detection_8_color_fix.py's own rect fit to within
            # ~0.15deg empirically) -- same green as the chassis rectangle
            # in underneath_detection's own surround view (0,220,80), for
            # visual consistency between windows.
            half = np.array([self.bbox_wb / 2.0, self.bbox_tr / 2.0])
            local_corners = np.array([[-half[0], -half[1]], [half[0], -half[1]],
                                       [half[0], half[1]], [-half[0], half[1]]])
            corner_pts = [self.bbox_cxcy + R_box @ lc for lc in local_corners]
            pts = np.array([self._to_px(*p) for p in corner_pts], dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 220, 80), 2)

            car_pt_px = self._to_px(*rx_xy)
            cv2.drawMarker(img, car_pt_px, (255, 0, 220), cv2.MARKER_DIAMOND, 12, 2)

            cv2.line(img, robot_pt_px, car_pt_px, (200, 200, 200), 1)

            misalign = rx_xy - self.robot_off
            misalign_dist = float(np.hypot(misalign[0], misalign[1]))
            cv2.putText(img, f"x differ={misalign[0]:+.3f}m  y differ={misalign[1]:+.3f}m  "
                              f"dist={misalign_dist:.3f}m",
                        (10, self.canvas_px - 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"bbox: wb={self.bbox_wb:.2f}m tr={self.bbox_tr:.2f}m",
                        (10, self.canvas_px - 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (0, 220, 80), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "no recent /car_bbox_dims", (10, self.canvas_px - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 255), 1, cv2.LINE_AA)

        # No freshness gate -- once a reading near RX has ever landed, keep
        # showing it (see the matching change in _compute_and_publish for
        # why: the underside isn't moving frame to frame while docked, so
        # this should have data all the time once the first reading comes
        # in, not flicker to n/a between updates). Unlike bbox_fresh above,
        # this deliberately keeps the last RX height reading on screen even
        # after the bbox itself goes stale.
        if self.lowest_z is not None:
            height_differ = self.lowest_z - self.robot_off_z
            src_txt = "" if self.lowest_z_source == "near" else f" [{self.lowest_z_source}]"
            state_txt = " FROZEN" if self.z_frozen else " settling"
            lz_txt = (f"lowest z near RX={self.lowest_z:+.3f}m   "
                      f"height differ={height_differ:+.3f}m{src_txt}{state_txt}")
            lz_col = (0, 255, 0) if self.z_frozen else (255, 255, 255)
        else:
            lz_txt = "lowest z near RX: n/a"
            lz_col = (255, 255, 255)
        cv2.putText(img, lz_txt, (10, self.canvas_px - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, lz_col, 1, cv2.LINE_AA)

        return img

    # ---- utility ----

    def _cloud_to_xyzi(self, msg: PointCloud2) -> np.ndarray:
        names = [f.name for f in msg.fields]
        use_intensity = "intensity" in names
        fields = ["x", "y", "z"] + (["intensity"] if use_intensity else [])
        cloud_arr = pc2.read_points(msg, field_names=fields, skip_nans=True)
        if getattr(cloud_arr, "dtype", None) is not None and cloud_arr.dtype.names is not None:
            cols = [cloud_arr[f].astype(np.float64) for f in fields]
            arr = np.stack(cols, axis=-1)
        else:
            arr = np.array([list(p) for p in cloud_arr], dtype=np.float64)
        if arr.shape[0] == 0:
            return np.zeros((0, 4))
        if not use_intensity:
            arr = np.hstack([arr, np.zeros((arr.shape[0], 1))])
        return arr


def main(args=None):
    rclpy.init(args=args)
    node = TransmitterAlignmentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
