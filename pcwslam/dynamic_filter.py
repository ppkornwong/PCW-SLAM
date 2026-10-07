#!/usr/bin/env python3
# ================================================================
# DynamicFilterOcclusionAware
#
# Short-window BEV diff, robot-following (same algorithm as
# ws/dynamic_detect_1_person_follow.py), publishing the CLEANED cloud.
#
#   - every scan is moved into the robot frame (pose = x, y, yaw), so
#     the BEV window follows/rotates with the robot
#   - current image   = last DIFF_WINDOW_SEC seconds of points
#   - reference image = the DIFF_WINDOW_SEC window as it was
#                       DIFF_DELAY_SEC ago, re-rendered at the CURRENT
#                       pose so static objects land on identical pixels
#   - "gone" pixels (bright in ref, dark now) -> grid cells -> blobs;
#     every point in a blob column is removed
#
# Kept from the earlier occlusion-aware version: gone pixels only count
# where the sensor actually saw this frame (ray-cast observed mask), and
# with no fresh pose the whole cloud passes through untouched.
#
# Outputs (unchanged for downstream): output_topic = cloud with movers
# removed, movers_topic = the removed points.
# ================================================================

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import numpy as np
import cv2
import time
import math
from collections import deque

from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header, String


# ================================================================
# PARAMETERS  (BEV block identical to dynamic_detect_5.py)
# ================================================================
BEV_SIZE   = 600
BEV_SCALE  = 100.0
BEV_X_MIN  = -3.0;  BEV_X_MAX = 8.0
BEV_Y_MIN  = -5.0;  BEV_Y_MAX = 5.0
Z_MIN      =  0.0;  Z_MAX     = 0.8

DIFF_WINDOW_SEC = 3.0   # points older than this expire from the diff image
DIFF_DELAY_SEC  = 2.0   # reference = that window as it was this long ago

INT_SCALE = 255.0

DIFF_THRESH      = 20
GRID_N           = 60
CELL_PX          = BEV_SIZE // GRID_N
CELL_ACTIVE_FRAC = 0.008
MIN_CELL_PIXELS  = max(int(CELL_PX * CELL_PX * CELL_ACTIVE_FRAC), 3)

BLOB_MIN_PX  = CELL_PX * CELL_PX
BLOB_MAX_PX  = CELL_PX * CELL_PX * 6
BLOB_MAX_ASP = 3.5

CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

TRACK_MAX_DIST_M = 1.5
TRACK_LOST_SEC   = 1.0

TOPIC = "/cloud_registered"

assert BEV_SIZE % GRID_N == 0, "BEV_SIZE must be divisible by GRID_N"


# ================================================================
# POINTCLOUD I/O  (identical to dynamic_detect_5.py)
# ================================================================
def read_points_fast(msg):
    n   = msg.width * msg.height
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
    fm  = {f.name: f.offset for f in msg.fields}
    x   = buf[:, fm['x']:fm['x']+4].copy().view(np.float32).reshape(-1)
    y   = buf[:, fm['y']:fm['y']+4].copy().view(np.float32).reshape(-1)
    z   = buf[:, fm['z']:fm['z']+4].copy().view(np.float32).reshape(-1)
    I   = buf[:, fm['intensity']:fm['intensity']+4].copy().view(np.float32).reshape(-1) \
          if 'intensity' in fm else np.full(n, 128.0, dtype=np.float32)
    c   = np.column_stack([x, y, z, I])
    return c[np.isfinite(c).all(axis=1)]


def cloud_to_pc2(cloud, frame_id, stamp):
    msg = PointCloud2(); msg.header = Header()
    msg.header.stamp = stamp; msg.header.frame_id = frame_id
    msg.height = 1; msg.width = max(len(cloud), 1)
    msg.is_dense = False; msg.is_bigendian = False
    fields = []
    for i, name in enumerate(['x', 'y', 'z', 'intensity']):
        f = PointField(); f.name = name; f.offset = i * 4
        f.datatype = PointField.FLOAT32; f.count = 1; fields.append(f)
    msg.fields = fields; msg.point_step = 16; msg.row_step = 16 * msg.width
    msg.data = cloud.astype(np.float32).tobytes() if len(cloud) > 0 else bytes(16)
    return msg


# ================================================================
# GEOMETRY HELPERS  (identical to dynamic_detect_5.py)
# ================================================================
def z_filter(cloud):
    return cloud[(cloud[:, 2] >= Z_MIN) & (cloud[:, 2] <= Z_MAX)]


def yaw_from_quat(q):
    return float(math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


def to_robot(cloud, pose):
    """World-frame points -> robot frame (pose = x, y, yaw)."""
    if len(cloud) == 0:
        return cloud
    x0, y0, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx = cloud[:, 0] - x0
    dy = cloud[:, 1] - y0
    out = cloud.copy()
    out[:, 0] =  c * dx + s * dy
    out[:, 1] = -s * dx + c * dy
    return out


def cloud_to_uv(cloud):
    X, Y = cloud[:, 0], cloud[:, 1]
    u = ((-Y - BEV_Y_MIN) / (BEV_Y_MAX - BEV_Y_MIN) * BEV_SIZE).astype(np.int32)
    v = ((BEV_X_MAX - X)  / (BEV_X_MAX - BEV_X_MIN) * BEV_SIZE).astype(np.int32)
    valid = (u >= 0) & (u < BEV_SIZE) & (v >= 0) & (v < BEV_SIZE)
    return u, v, valid


def px_to_m(u_px, v_px):
    x_m =  BEV_X_MAX - (v_px / BEV_SIZE) * (BEV_X_MAX - BEV_X_MIN)
    y_m = -((u_px / BEV_SIZE) * (BEV_Y_MAX - BEV_Y_MIN) + BEV_Y_MIN)
    return float(x_m), float(y_m)


def build_bev_gray(cloud):
    if len(cloud) == 0:
        return np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.uint8)

    u, v, valid = cloud_to_uv(cloud)
    I = cloud[:, 3][valid]
    flat = v[valid] * BEV_SIZE + u[valid]

    npx = BEV_SIZE * BEV_SIZE
    s = np.bincount(flat, weights=I.astype(np.float64), minlength=npx)
    c = np.bincount(flat, minlength=npx)
    avg = (s / np.maximum(c, 1)).reshape(BEV_SIZE, BEV_SIZE)

    return np.clip(avg / INT_SCALE * 255.0, 0, 255).astype(np.uint8)


def render_gray(cloud_robot, dk):
    """BEV -> dilate -> blur -> JET -> gray, exactly the image the diff
    thresholds (DIFF_THRESH etc.) were tuned on."""
    img8 = cv2.dilate(build_bev_gray(cloud_robot), dk, iterations=1)
    img8 = cv2.GaussianBlur(img8, (5, 5), 0)
    return cv2.cvtColor(cv2.applyColorMap(img8, cv2.COLORMAP_JET), cv2.COLOR_BGR2GRAY)


# ================================================================
# VISIBILITY / SHADOW
# ================================================================
def azimuth_first_return(points_xy, sensor_xy, n_az_bins):
    """Nearest-return range per azimuth bin, as seen from sensor_xy.
    Bins with no returns get +inf (= nothing known along that ray)."""
    d = points_xy - sensor_xy
    r = np.hypot(d[:, 0], d[:, 1])
    theta = np.arctan2(d[:, 1], d[:, 0])
    bins = np.clip(((theta + math.pi) / (2 * math.pi) * n_az_bins).astype(np.int32),
                   0, n_az_bins - 1)

    r_first = np.full(n_az_bins, np.inf, dtype=np.float64)
    if r.size == 0:
        return r_first

    order = np.argsort(bins, kind="stable")
    b_sorted = bins[order]
    r_sorted = r[order]
    uniq, start_idx = np.unique(b_sorted, return_index=True)
    r_first[uniq] = np.minimum.reduceat(r_sorted, start_idx)
    return r_first


class VisibilitySolver:
    """Grid-resolution observed/unknown classification. The BEV window is
    robot-centred (sensor at the origin), so cell centre positions are
    constant and are precomputed once."""

    def __init__(self, n_az_bins: int, shadow_margin_m: float):
        self.n_az_bins = n_az_bins
        self.shadow_margin_m = shadow_margin_m

        gx = np.zeros((GRID_N, GRID_N), dtype=np.float64)
        gy = np.zeros((GRID_N, GRID_N), dtype=np.float64)
        half = CELL_PX / 2.0
        for gi in range(GRID_N):
            for gj in range(GRID_N):
                x_m, y_m = px_to_m(gj * CELL_PX + half, gi * CELL_PX + half)
                gx[gi, gj] = x_m
                gy[gi, gj] = y_m
        self._gx = gx
        self._gy = gy

    def observed_grid(self, band_xy: np.ndarray, sensor_xy: np.ndarray) -> np.ndarray:
        """True where this frame actually carries information about the cell."""
        r_first = azimuth_first_return(band_xy, sensor_xy, self.n_az_bins)

        dx = self._gx - sensor_xy[0]
        dy = self._gy - sensor_xy[1]
        r = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        bins = np.clip(((theta + math.pi) / (2 * math.pi) * self.n_az_bins).astype(np.int32),
                       0, self.n_az_bins - 1)

        r_ray = r_first[bins]
        known_ray = np.isfinite(r_ray)
        return known_ray & (r <= r_ray + self.shadow_margin_m)


def upsample_grid(mask_grid: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(mask_grid, CELL_PX, axis=0), CELL_PX, axis=1)


# ================================================================
# FRAME DIFF
# ================================================================
def bev_frame_diff(cur_gray, ref_gray, observed_px):
    """Disappeared-pixel diff (ref bright, current dark), masked to observed
    pixels only. Returns (labels, detections); labels > 0 is the blob mask,
    and each size-valid detection carries its label for optional per-blob
    gating after tracking."""
    gone = np.clip(ref_gray.astype(np.int16) - cur_gray.astype(np.int16),
                   0, 255).astype(np.uint8)
    gone_bin = (gone > DIFF_THRESH) & observed_px

    cells = gone_bin.reshape(GRID_N, CELL_PX, GRID_N, CELL_PX).sum(axis=(1, 3))
    active = cells >= MIN_CELL_PIXELS
    grid_mask = upsample_grid(active).astype(np.uint8) * 255

    blob_mask = cv2.morphologyEx(grid_mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        blob_mask, connectivity=8)

    detections = []
    for lbl in range(1, n_labels):
        area_px = int(stats[lbl, cv2.CC_STAT_AREA])
        if not (BLOB_MIN_PX <= area_px <= BLOB_MAX_PX):
            continue
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if bw <= 0 or bh <= 0:
            continue
        if max(bw, bh) / max(min(bw, bh), 1) > BLOB_MAX_ASP:
            continue

        cx_px = float(centroids[lbl, 0]); cy_px = float(centroids[lbl, 1])
        x_m, y_m = px_to_m(cx_px, cy_px)
        x0 = int(stats[lbl, cv2.CC_STAT_LEFT])
        y0 = int(stats[lbl, cv2.CC_STAT_TOP])
        detections.append({
            "lbl": lbl,
            "cx_px": cx_px, "cy_px": cy_px,
            "x_m": x_m, "y_m": y_m,
            "area_px": area_px,
            "box": (x0, y0, x0 + bw, y0 + bh),
            "track": None,
        })

    return labels, detections


# ================================================================
# PERSON TRACKER
#   Same nearest-centroid logic as dynamic_detect_5.PersonTracker, plus
#   a confirmed_moving flag. dynamic_detect_5 computes vx/vy but never
#   gates removal on them; here they can actually be required.
# ================================================================
class PersonTracker:
    def __init__(self, move_speed_min: float, move_confirm_frames: int):
        self._tracks = {}
        self._next_id = 0
        self.move_speed_min = move_speed_min
        self.move_confirm_frames = move_confirm_frames

    def _bump_motion(self, trk):
        speed = math.hypot(trk["vx"], trk["vy"])
        if speed >= self.move_speed_min:
            trk["moving_frames"] += 1
        else:
            trk["moving_frames"] = 0
        if trk["moving_frames"] >= self.move_confirm_frames:
            trk["confirmed_moving"] = True

    def update(self, detections, t):
        max_d = TRACK_MAX_DIST_M * BEV_SCALE
        tids = list(self._tracks.keys())
        matched = set()

        if tids and detections:
            tp = np.array([[self._tracks[i]["cx_px"],
                            self._tracks[i]["cy_px"]] for i in tids])
            dp = np.array([[d["cx_px"], d["cy_px"]] for d in detections])
            dists = np.sqrt(((tp[:, None, :] - dp[None, :, :]) ** 2).sum(2))
            while dists.size:
                ti, di = np.unravel_index(np.argmin(dists), dists.shape)
                if dists[ti, di] > max_d:
                    break
                tid = tids[ti]; det = detections[di]; trk = self._tracks[tid]
                dt = t - trk["last_seen"]
                if dt > 0:
                    trk["vx"] = 0.7 * trk["vx"] + 0.3 * (det["x_m"] - trk["x_m"]) / dt
                    trk["vy"] = 0.7 * trk["vy"] + 0.3 * (det["y_m"] - trk["y_m"]) / dt
                trk.update({"cx_px": det["cx_px"], "cy_px": det["cy_px"],
                            "x_m": det["x_m"], "y_m": det["y_m"],
                            "area_px": det["area_px"], "box": det["box"],
                            "last_seen": t})
                trk["history"].append((det["x_m"], det["y_m"]))
                self._bump_motion(trk)
                det["track"] = trk
                matched.add(di)
                dists[ti, :] = np.inf
                dists[:, di] = np.inf

        for di, det in enumerate(detections):
            if di not in matched:
                tid = self._next_id; self._next_id += 1
                trk = {
                    "id": tid, "cx_px": det["cx_px"], "cy_px": det["cy_px"],
                    "x_m": det["x_m"], "y_m": det["y_m"],
                    "vx": 0.0, "vy": 0.0,
                    "area_px": det["area_px"], "box": det["box"],
                    "last_seen": t, "history": deque(maxlen=40),
                    "moving_frames": 0, "confirmed_moving": False,
                }
                self._tracks[tid] = trk
                det["track"] = trk

        for tid in [i for i, tr in self._tracks.items()
                    if t - tr["last_seen"] > TRACK_LOST_SEC]:
            del self._tracks[tid]

        return list(self._tracks.values())


# ================================================================
# MAIN NODE
# ================================================================
class DynamicFilterOcclusionAware(Node):

    def __init__(self):
        super().__init__("dynamic_filter_occlusion_aware")

        self.declare_parameter("output_topic", "/dynamic_points")
        self.declare_parameter("movers_topic", "/moving_points")
        self.declare_parameter("odom_topic", "/aft_mapped_to_init")
        self.declare_parameter("az_bins", 720)
        self.declare_parameter("shadow_margin_m", 0.15)
        self.declare_parameter("pose_timeout_s", 1.0)
        # off = remove every diff blob (like dynamic_detect_1_person_follow);
        # on  = only blobs whose track is confirmed moving
        self.declare_parameter("require_moving_track", False)
        self.declare_parameter("move_speed_min_mps", 0.15)
        self.declare_parameter("move_confirm_frames", 2)
        self.declare_parameter("publish_status", True)

        gp = self.get_parameter
        out_topic = gp("output_topic").value
        movers_topic = gp("movers_topic").value
        odom_topic = gp("odom_topic").value
        self.pose_timeout_s = gp("pose_timeout_s").value
        self.require_moving_track = bool(gp("require_moving_track").value)
        self.publish_status = bool(gp("publish_status").value)

        self._vis = VisibilitySolver(int(gp("az_bins").value),
                                      float(gp("shadow_margin_m").value))
        self._tracker = PersonTracker(float(gp("move_speed_min_mps").value),
                                       int(gp("move_confirm_frames").value))

        self._dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # (t, world-frame z-banded scan); long enough to rebuild the reference
        self._hist = deque()

        self._pose = None            # (x, y, yaw) in world
        self._last_pose_time = 0.0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(PointCloud2, TOPIC, self._cb, sensor_qos)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.create_subscription(Path, "/path", self._path_cb, 10)

        self._dyn_pub = self.create_publisher(PointCloud2, out_topic, 10)
        self._movers_pub = self.create_publisher(PointCloud2, movers_topic, 10)
        self._status_pub = self.create_publisher(String, "/dynamic_filter_status", 10) \
            if self.publish_status else None

        self._stats = {"passthrough_no_pose": 0, "removed_total": 0}

        self.get_logger().info(
            f"DynamicFilterOcclusionAware  in={TOPIC}  out={out_topic}  "
            f"odom={odom_topic}  window={DIFF_WINDOW_SEC}s delay={DIFF_DELAY_SEC}s  "
            f"require_moving_track={self.require_moving_track}")

    # ─────────────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self._pose = (p.x, p.y, yaw_from_quat(msg.pose.pose.orientation))
        self._last_pose_time = time.time()

    def _path_cb(self, msg: Path):
        # Fallback pose source; /aft_mapped_to_init is preferred (much
        # higher rate). Only used when odom has not arrived recently.
        if not msg.poses:
            return
        if (time.time() - self._last_pose_time) <= self.pose_timeout_s:
            return
        pose = msg.poses[-1].pose
        self._pose = (pose.position.x, pose.position.y,
                      yaw_from_quat(pose.orientation))
        self._last_pose_time = time.time()

    # ─────────────────────────────────────────────────────────────
    def _cb(self, msg: PointCloud2):
        t0 = time.time()
        none = np.zeros((0, 4), np.float32)

        cloud_full = read_points_fast(msg)
        if len(cloud_full) == 0:
            self._publish(cloud_full, none, msg)
            return

        # FAIL SAFE: without a pose we cannot render the robot-frame BEV or
        # tell shadow from free space, so nothing may be removed.
        pose = self._pose
        if pose is None or (t0 - self._last_pose_time) > self.pose_timeout_s:
            self._stats["passthrough_no_pose"] += 1
            self._publish(cloud_full, none, msg)
            self._publish_status(len(cloud_full), len(cloud_full), 0, 0,
                                 f"NO_POSE dt={int((time.time()-t0)*1000)}ms")
            return

        band = z_filter(cloud_full)

        self._hist.append((t0, band))
        keep_s = DIFF_WINDOW_SEC + DIFF_DELAY_SEC + 1.0
        while self._hist and self._hist[0][0] < t0 - keep_s:
            self._hist.popleft()

        # Not enough history yet to have a reference from DIFF_DELAY_SEC ago.
        if self._hist[0][0] > t0 - DIFF_DELAY_SEC:
            self._publish(cloud_full, none, msg)
            return

        # Current image: the last DIFF_WINDOW_SEC of points.
        cur_world = np.vstack([c for (t, c) in self._hist
                               if t >= t0 - DIFF_WINDOW_SEC])
        cur_g = render_gray(to_robot(cur_world, pose), self._dk)

        # Reference: the same-length window ending DIFF_DELAY_SEC ago,
        # re-rendered at the CURRENT pose so static things line up.
        target = t0 - DIFF_DELAY_SEC
        t_end = min(self._hist, key=lambda x: abs(x[0] - target))[0]
        ref_world = np.vstack([c for (t, c) in self._hist
                               if t_end - DIFF_WINDOW_SEC <= t <= t_end])
        ref_g = render_gray(to_robot(ref_world, pose), self._dk)

        # Sensor sits at the origin of the robot-frame BEV.
        band_r = to_robot(band, pose)
        observed_grid = self._vis.observed_grid(band_r[:, :2], np.zeros(2))
        observed_px = upsample_grid(observed_grid)

        labels, detections = bev_frame_diff(cur_g, ref_g, observed_px)
        tracks = self._tracker.update(detections, t0)

        if self.require_moving_track:
            keep = [d["lbl"] for d in detections
                    if d["track"] is not None and d["track"]["confirmed_moving"]]
            removal_mask = np.isin(labels, keep)
        else:
            removal_mask = labels > 0

        if removal_mask.any():
            u, v, valid = cloud_to_uv(to_robot(cloud_full, pose))
            in_dyn = np.zeros(len(cloud_full), dtype=bool)
            in_dyn[valid] = removal_mask[v[valid], u[valid]]
            cleaned = cloud_full[~in_dyn]
            movers  = cloud_full[in_dyn]
        else:
            cleaned = cloud_full
            movers  = none

        self._stats["removed_total"] += len(movers)
        self._publish(cleaned, movers, msg)

        shadow_frac = 1.0 - float(observed_grid.mean())
        n_moving = sum(1 for t in tracks if t["confirmed_moving"])
        self._publish_status(len(cloud_full), len(cleaned), len(movers), n_moving,
                             f"shadow={shadow_frac:.2f} dt={int((time.time()-t0)*1000)}ms")

    # ─────────────────────────────────────────────────────────────
    def _publish_status(self, n_in, n_kept, n_removed, n_moving, extra):
        if self._status_pub is not None:
            self._status_pub.publish(String(
                data=f"in={n_in} kept={n_kept} removed={n_removed} "
                     f"movers={n_moving} {extra}"))

    def _publish(self, cleaned, movers, src_msg):
        self._dyn_pub.publish(
            cloud_to_pc2(cleaned,
                         frame_id=src_msg.header.frame_id,
                         stamp=src_msg.header.stamp))
        self._movers_pub.publish(
            cloud_to_pc2(movers,
                         frame_id=src_msg.header.frame_id,
                         stamp=src_msg.header.stamp))


# ================================================================
# MAIN
# ================================================================
def main(args=None):
    rclpy.init(args=args)
    node = DynamicFilterOcclusionAware()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
