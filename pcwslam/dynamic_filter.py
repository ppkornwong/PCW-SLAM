#!/usr/bin/env python3
# ================================================================
# DynamicFilterOcclusionAware
#
# NEW FILE. Does not modify dynamic_detect_5.py.
# *** RUN THIS INSTEAD OF dynamic_detect_5.py, NOT ALONGSIDE IT ***
# (both publish /dynamic_points by default; two publishers on the
#  same topic would interleave two different clouds downstream).
#
# ----------------------------------------------------------------
# WHY THIS EXISTS
# ----------------------------------------------------------------
# dynamic_detect_5.py:186-192 documents its core premise as:
#
#     "/cloud_registered is an accumulating SLAM output, so:
#        - 'disappeared' pixels = something WAS there and moved -> signal"
#
# That premise does not hold for this SLAM fork. In
# laserMapping.cpp, publish_frame_world() (lines 521-577) rebuilds the
# /cloud_registered message from scratch every scan out of
# feats_down_world -- it publishes ONLY the current scan's points
# transformed to world frame. There is no accumulation on the C++ side.
# (The dense "map" seen in rviz is rviz's own Decay Time display
# holding a window of past per-scan messages, not a persisted cloud.)
#
# Consequence: "disappeared" does NOT uniquely mean "moved". A pixel
# goes dark for ANY reason the current scan lacks a return there:
#     - occlusion  (the 3cm transmitter under the car)  <-- the problem
#     - the surface leaving the sensor FOV as the robot moves
#     - range dropout / grazing incidence
#     - ordinary scan-pattern variation
#
# dynamic_detect_5's two-EMA diff, 9x9 dilation and PERSIST_K=3 gate
# all defend against *random* flicker. Occlusion is *systematic and
# persistent*, so it defeats every one of them: the fast EMA
# (alpha=0.35) collapses in ~3 frames while the slow reference
# (alpha=0.05) holds the stale brightness for ~2s, the diff lights up,
# and the persistence counter only climbs because a shadow -- unlike
# flicker -- never blinks off. BLOB_MAX_PX = CELL_PX^2 * 24 = 2400 px^2
# is ~0.74 m^2 at this BEV scale, i.e. person-sized by construction, so
# the occlusion shadow is then accepted as a person and its points are
# removed COLUMN-WISE AT ALL HEIGHTS (dynamic_detect_5.py:394-399),
# taking the car undercarriage with it -- which is exactly the data
# underneath_detection_6.py needs to fit its chassis rectangle.
#
# The SLAM itself never subscribes to /dynamic_points (it consumes the
# raw lidar topic and PUBLISHES /cloud_registered), which is why the
# mapping result looks unaffected while the detection pipeline starves.
#
# ----------------------------------------------------------------
# THE FIX: free space vs unknown space
# ----------------------------------------------------------------
# Absence of a return is only evidence of change if the sensor actually
# looked there. This node separates the two cases by ray casting from
# the sensor origin, the standard occupancy-grid free/unknown split:
#
#   for each azimuth bin, r_first = range of the nearest return
#     r <  r_first - margin : ray passed through  -> FREE      (observed)
#     |r - r_first| <= margin : the surface itself -> OCCUPIED (observed)
#     r >  r_first + margin : behind an obstacle  -> SHADOW    (UNKNOWN)
#     no returns in that bin at all               -> UNKNOWN
#
# Disappearance is only believed where the pixel is OBSERVED this
# frame. In shadow the reference image is frozen (not decayed) and the
# persistence counter is held, so occluded geometry is never mistaken
# for departed geometry.
#
# This keeps genuine person removal intact: when someone walks away,
# the rays that used to stop on them now reach the background, so
# r_first grows and their old location becomes observed-free -- a real
# disappearance, and still removed. When the transmitter blocks the
# undercarriage, those rays stop early, the region behind is unknown,
# and nothing is removed.
#
# Secondary gate (require_moving_track, default on): a blob is only
# removed if the tracker has confirmed it actually moving. Note that
# dynamic_detect_5.py already computes per-track vx/vy but never uses
# them for removal -- the velocities feed only its log line.
#
# Re-baselining: when a pixel returns from shadow to observed, its
# reference is reset to the current frame rather than diffed against a
# value from before the occlusion, so a long occluded stretch does not
# produce one large false "disappearance" the moment visibility
# returns.
#
# Fails SAFE: with no pose yet, or no returns to raycast, the full
# cloud is passed through unfiltered. Removing nothing is always
# preferable here to removing the car.
#
# ----------------------------------------------------------------
# BEV geometry, topic names, image constants and blob thresholds are
# kept byte-identical to dynamic_detect_5.py so downstream consumers
# (underneath_detection_6.py et al) see exactly the same conventions.
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

REF_ALPHA  = 0.05
FAST_ALPHA = 0.35

INT_SCALE = 255.0

DIFF_THRESH      = 20
GRID_N           = 60
CELL_PX          = BEV_SIZE // GRID_N
CELL_ACTIVE_FRAC = 0.08
MIN_CELL_PIXELS  = max(int(CELL_PX * CELL_PX * CELL_ACTIVE_FRAC), 3)
PERSIST_K        = 3

BLOB_MIN_PX  = CELL_PX * CELL_PX
BLOB_MAX_PX  = CELL_PX * CELL_PX * 24
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


# ================================================================
# VISIBILITY / SHADOW  (new -- the actual fix)
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
    fixed in world coordinates, so cell centre positions are constant and
    are precomputed once."""

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
    """Same structure as dynamic_detect_5.bev_frame_diff, with the
    disappearance evidence masked to observed pixels only. Returns
    (labels, grid_mask, detections); each detection carries its label so
    the caller can accept/reject per blob after tracking."""
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

    return labels, grid_mask, detections


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
        self.declare_parameter("require_moving_track", True)
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

        self._dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        self._ref     = None
        self._fast    = None
        self._persist = np.zeros((BEV_SIZE, BEV_SIZE), np.uint8)
        self._was_observed = np.zeros((BEV_SIZE, BEV_SIZE), bool)

        self._sensor_xy = None
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

        self._last_log = 0.0
        self._stats = {"passthrough_no_pose": 0, "removed_total": 0}

        self.get_logger().info(
            f"DynamicFilterOcclusionAware  in={TOPIC}  out={out_topic}  "
            f"odom={odom_topic}  az_bins={gp('az_bins').value}  "
            f"require_moving_track={self.require_moving_track}  "
            f"(run INSTEAD OF dynamic_detect_5.py)")

    # ─────────────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self._sensor_xy = np.array([p.x, p.y], dtype=np.float64)
        self._last_pose_time = time.time()

    def _path_cb(self, msg: Path):
        # Fallback pose source; /aft_mapped_to_init is preferred (much
        # higher rate). Only used when odom has not arrived recently.
        if not msg.poses:
            return
        if (time.time() - self._last_pose_time) <= self.pose_timeout_s:
            return
        p = msg.poses[-1].pose.position
        self._sensor_xy = np.array([p.x, p.y], dtype=np.float64)
        self._last_pose_time = time.time()

    # ─────────────────────────────────────────────────────────────
    def _cb(self, msg: PointCloud2):
        t0 = time.time()

        cloud_full = read_points_fast(msg)
        if len(cloud_full) == 0:
            self._publish(cloud_full, np.zeros((0, 4), np.float32), msg)
            return

        # FAIL SAFE: without a pose we cannot tell shadow from free space,
        # so nothing may be removed. Passing the whole cloud through is
        # always preferable to deleting the car.
        pose_fresh = (self._sensor_xy is not None and
                      (t0 - self._last_pose_time) <= self.pose_timeout_s)
        if not pose_fresh:
            self._stats["passthrough_no_pose"] += 1
            self._publish(cloud_full, np.zeros((0, 4), np.float32), msg)
            self._maybe_log(t0, len(cloud_full), len(cloud_full), 0, 0, "NO_POSE")
            return

        band = z_filter(cloud_full)

        cur = build_bev_gray(band)
        cur = cv2.dilate(cur, self._dk, iterations=1)
        cur = cv2.GaussianBlur(cur, (5, 5), 0)

        observed_grid = self._vis.observed_grid(band[:, :2], self._sensor_xy)
        observed_px = upsample_grid(observed_grid)
        observed_u8 = observed_px.astype(np.uint8)

        if self._ref is None:
            self._ref  = cur.astype(np.float32)
            self._fast = cur.astype(np.float32)
            self._was_observed = observed_px.copy()
            self._publish(cloud_full, np.zeros((0, 4), np.float32), msg)
            return

        # Re-baseline pixels that just came back from shadow: comparing a
        # fresh observation against a reference from before the occlusion
        # would manufacture one large false disappearance at the moment
        # visibility returns.
        reentered = observed_px & (~self._was_observed)
        if reentered.any():
            self._ref[reentered] = cur[reentered]
            self._fast[reentered] = cur[reentered]
            self._persist[reentered] = 0
        self._was_observed = observed_px.copy()

        # EMAs advance only where the scene was actually observed; in
        # shadow both images hold their last known value.
        cv2.accumulateWeighted(cur, self._fast, FAST_ALPHA, mask=observed_u8)
        fast_u8 = self._fast.astype(np.uint8)
        ref_u8  = self._ref.astype(np.uint8)

        labels, grid_mask, detections = bev_frame_diff(fast_u8, ref_u8, observed_px)

        cv2.accumulateWeighted(cur, self._ref, REF_ALPHA, mask=observed_u8)

        # Persistence counter also only advances where observed, so a
        # static shadow cannot accumulate its way past PERSIST_K.
        self._persist = np.where(observed_px & (grid_mask > 0),
                                 np.minimum(self._persist + 1, 250),
                                 np.where(observed_px, 0, self._persist)
                                 ).astype(np.uint8)

        tracks = self._tracker.update(detections, t0)

        if self.require_moving_track:
            keep = [d["lbl"] for d in detections
                    if d["track"] is not None and d["track"]["confirmed_moving"]]
        else:
            keep = [d["lbl"] for d in detections]

        if keep:
            accepted_mask = np.isin(labels, keep)
        else:
            accepted_mask = np.zeros((BEV_SIZE, BEV_SIZE), dtype=bool)

        removal_gate = accepted_mask & (self._persist >= PERSIST_K) & observed_px

        if removal_gate.any():
            u, v, valid = cloud_to_uv(cloud_full)
            in_dyn = np.zeros(len(cloud_full), dtype=bool)
            in_dyn[valid] = removal_gate[v[valid], u[valid]]
            cleaned = cloud_full[~in_dyn]
            movers  = cloud_full[in_dyn]
        else:
            cleaned = cloud_full
            movers  = np.zeros((0, 4), dtype=np.float32)

        self._stats["removed_total"] += len(movers)
        self._publish(cleaned, movers, msg)

        shadow_frac = 1.0 - float(observed_grid.mean())
        n_moving = sum(1 for t in tracks if t["confirmed_moving"])
        self._maybe_log(t0, len(cloud_full), len(cleaned), len(movers),
                        n_moving, f"shadow={shadow_frac:.2f}")

    # ─────────────────────────────────────────────────────────────
    def _maybe_log(self, t0, n_in, n_kept, n_removed, n_moving, extra):
        if self._status_pub is not None:
            self._status_pub.publish(String(
                data=f"in={n_in} kept={n_kept} removed={n_removed} "
                     f"movers={n_moving} {extra}"))
        if t0 - self._last_log > 5.0:
            self._last_log = t0
            self.get_logger().info(
                f"in={n_in} kept={n_kept} removed={n_removed} "
                f"movers={n_moving} {extra} "
                f"dt={int((time.time()-t0)*1000)}ms")

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
