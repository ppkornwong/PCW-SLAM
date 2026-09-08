import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import numpy as np
import cv2
import time
import threading
import queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Path
from std_msgs.msg import String
from std_msgs.msg import Float32

# ================================================================
# FAST POINTCLOUD READER
# ================================================================
def read_points_fast(msg):
    point_step = msg.point_step
    n_points   = msg.width * msg.height
    buf        = np.frombuffer(msg.data, dtype=np.uint8).reshape(n_points, point_step)
    field_map  = {f.name: f.offset for f in msg.fields}

    x = buf[:, field_map['x']:field_map['x']+4].copy().view(np.float32).reshape(-1)
    y = buf[:, field_map['y']:field_map['y']+4].copy().view(np.float32).reshape(-1)
    z = buf[:, field_map['z']:field_map['z']+4].copy().view(np.float32).reshape(-1)

    if 'intensity' in field_map:
        intensity = buf[:, field_map['intensity']:field_map['intensity']+4].copy()\
                        .view(np.float32).reshape(-1)
    else:
        intensity = np.full(n_points, 128.0, dtype=np.float32)

    cloud = np.column_stack([x, y, z, intensity])
    valid = np.isfinite(cloud).all(axis=1)
    return cloud[valid]


# ================================================================
# DEPTH BUFFER  –  keep nearest point per pixel
# ================================================================
def _depth_and_intensity(flat_idx, dist, intensity, n_pixels):
    order  = np.argsort(dist)
    idx_s  = flat_idx[order];  dist_s = dist[order];  int_s = intensity[order]
    _, uniq = np.unique(idx_s, return_index=True)
    depth_buf = np.full(n_pixels, np.inf, dtype=np.float32)
    int_buf   = np.zeros(n_pixels, dtype=np.float32)
    depth_buf[idx_s[uniq]] = dist_s[uniq]
    int_buf  [idx_s[uniq]] = int_s [uniq]
    return depth_buf, int_buf


# ================================================================
# PERSPECTIVE PROJECTION
# ================================================================
def perspective_view(X, Y, Z, intensity,
                     width=480, height=240,
                     fx=120, fy=100,
                     max_dist=15.0,
                     dilate_kernel=None,
                     label="", label_color=(255, 255, 255)):
    cx_img = width  // 2
    cy_img = int(height * 0.55)

    mask = X > 0.1
    Xf, Yf, Zf, If = X[mask], Y[mask], Z[mask], intensity[mask]
    if Xf.size == 0:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        if label:
            cv2.putText(img, label, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 1, cv2.LINE_AA)
        return img, np.full((height,width),np.inf,np.float32), np.zeros((height,width),np.float32)

    u = (Yf / Xf * fx + cx_img).astype(np.int32)
    v = (-Zf / Xf * fy + cy_img).astype(np.int32)

    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, If = u[valid], v[valid], If[valid]
    dist = np.sqrt(Xf[valid]**2 + Yf[valid]**2 + Zf[valid]**2)

    flat_idx = v * width + u
    n_pixels = height * width
    depth_buf, int_buf = _depth_and_intensity(flat_idx, dist, If, n_pixels)

    depth_map  = depth_buf.reshape(height, width)
    int_map    = int_buf.reshape(height, width)
    empty_mask = ~np.isfinite(depth_map)

    depth_norm = 1.0 - np.clip(depth_map / max_dist, 0, 1)
    depth_norm[empty_mask] = 0.0
    int_norm = np.clip(int_map / 255.0, 0, 1)

    blended = (0.5 * depth_norm + 0.5 * int_norm)
    img8 = (blended * 255).astype(np.uint8)

    if dilate_kernel is not None:
        occupied = (~empty_mask).astype(np.uint8) * 255
        img8     = cv2.dilate(img8,     dilate_kernel, iterations=1)
        occupied = cv2.dilate(occupied, dilate_kernel, iterations=1)
    else:
        occupied = (~empty_mask).astype(np.uint8) * 255

    img_color = cv2.applyColorMap(img8, cv2.COLORMAP_JET)
    img_color[occupied == 0] = 0

    if label:
        cv2.putText(img_color, label, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 1, cv2.LINE_AA)
    return img_color, depth_map, int_map


# ================================================================
# FOUR DIRECTIONAL VIEWS
# ================================================================
def four_views(cloud, proj_width=480, proj_height=240,
               dilate_kernel=None, camera_height=0.5):
    X0 = cloud[:, 0];  Y0 = cloud[:, 1]
    Z  = cloud[:, 2] - camera_height
    I  = cloud[:, 3]

    front = perspective_view( X0,  Y0, Z, I, proj_width, proj_height,
                              dilate_kernel=dilate_kernel,
                              label="FRONT", label_color=(0, 255, 0))
    rear  = perspective_view(-X0, -Y0, Z, I, proj_width, proj_height,
                              dilate_kernel=dilate_kernel,
                              label="REAR",  label_color=(0, 80, 255))
    right = perspective_view( Y0, -X0, Z, I, proj_width, proj_height,
                              dilate_kernel=dilate_kernel,
                              label="RIGHT", label_color=(0, 255, 255))
    left  = perspective_view(-Y0,  X0, Z, I, proj_width, proj_height,
                              dilate_kernel=dilate_kernel,
                              label="LEFT",  label_color=(255, 220, 0))
    return front, rear, left, right


# ================================================================
# BEV  –  dilated
# ================================================================
def create_bev_dilated(cloud, bev_size=400, scale=100.0,
                       dilate_kernel=None, max_z=2.0):
    cloud = cloud[cloud[:, 2] < max_z]
    if len(cloud) == 0:
        return None

    X, Y, I = cloud[:, 0], cloud[:, 1], cloud[:, 3]
    roi     = (X > -3.0) & (X < 8.0) & (np.abs(Y) < 5.0)
    X, Y, I = X[roi], Y[roi], I[roi]
    if len(X) == 0:
        return None

    S  = bev_size
    u  = (-Y * scale + S / 2).astype(np.int32)
    v  = (S / 2 - X * scale).astype(np.int32)
    valid = (u >= 0) & (u < S) & (v >= 0) & (v < S)
    u, v, I = u[valid], v[valid], I[valid]

    flat    = v * S + u
    pixels  = S * S
    img_sum = np.bincount(flat, weights=I.astype(np.float64), minlength=pixels)
    img_cnt = np.bincount(flat, minlength=pixels)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg = np.where(img_cnt > 0, img_sum / img_cnt, 0.0).reshape(S, S)

    # Percentile-clipped contrast stretch instead of raw max-normalization.
    # avg/avg.max() lets a single bright outlier pixel set the whole scale,
    # crushing every ordinary-intensity return into the low (blue, in the
    # JET colormap) end -- which is why the car blob barely differed from
    # background. Clipping the top/bottom 2% before stretching spreads the
    # real mid-range values across the full 0-255 span instead.
    occ = (img_cnt > 0).reshape(S, S)
    if occ.any():
        vals = avg[occ]
        lo = float(np.percentile(vals, 2))
        hi = float(np.percentile(vals, 98))
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        img8 = np.clip((avg - lo) / (hi - lo), 0.0, 1.0)
        img8 = (img8 * 255).astype(np.uint8)
        img8[~occ] = 0
    else:
        img8 = np.zeros((S, S), dtype=np.uint8)

    if dilate_kernel is not None:
        img8 = cv2.dilate(img8, dilate_kernel, iterations=1)

    img8  = cv2.GaussianBlur(img8, (5, 5), 0)
    color = cv2.applyColorMap(img8, cv2.COLORMAP_JET)
    _draw_robot_marker(color, S, "BEV DILATED")
    return color


# ================================================================
# BLIND-SPOT VIEW  –  raw occupancy, no BEV photometric pipeline
#
# create_bev_dilated exists to make the car LOOK GOOD (z-band clip,
# percentile contrast stretch, Gaussian blur, JET colormap) -- none of
# that is needed for the one thing this panel has to answer: did we get
# a return near a given pixel, yes or no. A light dilation IS still
# applied though, and it's not optional cosmetics here: individual LiDAR
# returns along a scan line are naturally spaced a few pixels apart, so
# on a bare undilated occupancy image almost the ENTIRE background reads
# as "empty" too -- _detect_hole's contour test (a compact dark blob
# enclosed by occupied pixels) can't tell the deliberate blind-spot gap
# apart from ordinary inter-point space without first closing those
# small gaps into a continuous surface. That's why the hole showed up on
# the BEV panel (which already dilates) but not here.
# ================================================================
def build_blind_spot_view(cloud, bev_size=400, scale=100.0, dilate_kernel=None):
    """Plain white-on-black occupancy render of a raw point cloud, no
    height band, no ROI crop, no intensity blending, no colormap -- just
    dilation, to feed _detect_hole with an unambiguous "was this area hit
    at all" image."""
    img = np.zeros((bev_size, bev_size), dtype=np.uint8)
    if len(cloud) > 0:
        X, Y = cloud[:, 0], cloud[:, 1]
        S = bev_size
        u = (-Y * scale + S / 2).astype(np.int32)
        v = (S / 2 - X * scale).astype(np.int32)
        valid = (u >= 0) & (u < S) & (v >= 0) & (v < S)
        img[v[valid], u[valid]] = 255

        if dilate_kernel is not None:
            img = cv2.dilate(img, dilate_kernel, iterations=1)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# ================================================================
# Z-BAND BEV
# ================================================================
Z_CAR_LOW  = 0.08
Z_CAR_HIGH = 0.55


def create_bev_zband(cloud, bev_size=400, scale=100.0,
                     dilate_kernel=None):
    band = cloud[(cloud[:, 2] >= Z_CAR_LOW) & (cloud[:, 2] <= Z_CAR_HIGH)]
    if len(band) < 10:
        return None, None

    X, Y = band[:, 0], band[:, 1]
    roi = (X > -3.0) & (X < 4.0) & (np.abs(Y) < 3.0)
    X, Y = X[roi], Y[roi]
    if len(X) < 10:
        return None, None

    S  = bev_size
    u  = (-Y * scale + S/2).astype(np.int32)
    v  = (S/2 - X * scale).astype(np.int32)
    valid = (u >= 0) & (u < S) & (v >= 0) & (v < S)
    u, v = u[valid], v[valid]

    flat    = v * S + u
    density = np.bincount(flat, minlength=S*S).reshape(S, S).astype(np.float32)

    raw = density.copy()
    if density.max() > 0:
        d8 = (density / density.max() * 255).astype(np.uint8)
    else:
        d8 = np.zeros((S, S), np.uint8)

    if dilate_kernel is not None:
        d8 = cv2.dilate(d8, dilate_kernel, iterations=1)

    return raw, d8


# ================================================================
# INTENSITY-STRATIFIED BEV  —  3 MATERIAL BANDS  (corrected direction)
#
# Field observation on the real car (JET colormap check):
#   battery casing (metal)  = BLUE   -> LOWEST  intensity
#   underbody frame (plastic)= YELLOW -> MID     intensity
#   tyres (rubber)           = ORANGE -> HIGHEST intensity
#
# Physically plausible: polished metal reflects the beam specularly
# AWAY from the receiver (weak return), while matte plastic/rubber
# scatter diffusely straight back (strong return).
#
# Bands are adaptive tertiles of the local intensity distribution so
# the split survives sensor/gain changes; fixed fallbacks otherwise.
# ================================================================
INT_METAL_MAX_FB  = 45     # fallback upper bound of metal (LOW) band
INT_RUBBER_MIN_FB = 100    # fallback lower bound of rubber (HIGH) band
INT_ROI_M         = 2.5
INT_Z_MIN         = -0.05  # tyre contact patch / low underbody can sit
INT_Z_MAX         = 1.00   # just below 0; include it. Upper raised so a
                           # taller vehicle underside is still captured.


def create_intensity_bev(cloud, bev_size=400, scale=100.0,
                         dilate_kernel=None):
    """Returns metal_gray, plastic_gray, rubber_gray, combined(BGR)."""
    S = bev_size

    r = INT_ROI_M
    mask = ((cloud[:,0] > -r) & (cloud[:,0] < r) & (np.abs(cloud[:,1]) < r)
            & (cloud[:,2] > INT_Z_MIN) & (cloud[:,2] < INT_Z_MAX))
    local = cloud[mask]
    if len(local) < 10:
        empty = np.zeros((S,S), np.uint8)
        return empty, empty.copy(), empty.copy(), np.zeros((S,S,3),np.uint8)

    X, Y, Z, I = local[:,0], local[:,1], local[:,2], local[:,3]
    u = (-Y * scale + S/2).astype(np.int32)
    v = (S/2 - X * scale).astype(np.int32)
    in_frame = (u>=0)&(u<S)&(v>=0)&(v<S)
    u,v,I = u[in_frame], v[in_frame], I[in_frame]
    flat = v*S + u

    if len(I) > 20:
        i_lo = float(np.percentile(I, 33))   # metal | plastic split
        i_hi = float(np.percentile(I, 66))   # plastic | rubber split
        # Guard: if the intensity distribution is nearly uniform (a car
        # with only frame returns and no distinct tyre/battery bands), the
        # tertile split would manufacture a spurious "rubber" band from the
        # top third of essentially-identical values -> false wheel confirms.
        # When the high tertile isn't meaningfully brighter than the mid,
        # fall back to absolute thresholds so the rubber band stays empty
        # unless genuinely high-intensity returns exist.
        if (i_hi - i_lo) < 12.0:
            i_lo = INT_METAL_MAX_FB
            i_hi = INT_RUBBER_MIN_FB
    else:
        i_lo = INT_METAL_MAX_FB
        i_hi = INT_RUBBER_MIN_FB

    metal_mask   = I <= i_lo                 # battery casing  (LOW)
    plastic_mask = (I > i_lo) & (I < i_hi)   # frame panels    (MID)
    rubber_mask  = I >= i_hi                 # tyres           (HIGH)

    def occ(m):
        if m.sum() == 0:
            return np.zeros((S,S), np.float32)
        return np.bincount(flat[m], minlength=S*S)\
                 .reshape(S,S).astype(np.float32)

    metal_occ   = occ(metal_mask)
    plastic_occ = occ(plastic_mask)
    rubber_occ  = occ(rubber_mask)

    # Per-band normalization (each band to its own max). The rectangle fit
    # needs each band's SHAPE crisp; the wheel-confirm test was reworked to
    # use peak + area + a 0.6x plastic ratio (not absolute cross-band
    # means), so it no longer needs a shared scale to work.
    def to_u8(arr):
        m = arr.max()
        if m > 0:
            return np.clip(arr / m * 255.0, 0, 255).astype(np.uint8)
        return np.zeros_like(arr, dtype=np.uint8)

    metal_gray   = to_u8(metal_occ)
    plastic_gray = to_u8(plastic_occ)
    rubber_gray  = to_u8(rubber_occ)

    if dilate_kernel is not None:
        metal_gray   = cv2.dilate(metal_gray,   dilate_kernel)
        plastic_gray = cv2.dilate(plastic_gray, dilate_kernel)
        rubber_gray  = cv2.dilate(rubber_gray,  dilate_kernel)

    # debug composite: previously additive B=metal/G=plastic/R=rubber
    # channel-blend. Any pixel with points from more than one band mixed
    # into a secondary color (yellow/cyan/magenta), so wherever bands sit
    # close together spatially the whole area washed into similar-looking
    # blended hues instead of three visually distinct materials.
    #
    # Fixed: winner-take-all categorical coloring. Each pixel gets the
    # single bold, maximally-distinct color of whichever band is strongest
    # there (no blending possible), with brightness scaled by that band's
    # own normalized strength so faint vs. strong detections still read,
    # while the hue itself never mixes.
    stack = np.stack([metal_gray, plastic_gray, rubber_gray], axis=-1).astype(np.float32)
    winner = np.argmax(stack, axis=-1)
    strength = stack.max(axis=-1) / 255.0
    has_signal = stack.max(axis=-1) > 0

    # BGR, chosen for max hue separation from each other and from the
    # (0,220,80) green used elsewhere to draw the fitted chassis rectangle.
    INT_BEV_PALETTE = np.array([
        [  0, 140, 255],   # metal   (battery, low intensity)  -> orange
        [255, 180,   0],   # plastic (frame,   mid intensity)  -> sky blue
        [255,   0, 220],   # rubber  (tyres,   high intensity) -> magenta
    ], dtype=np.float32)

    combined = (INT_BEV_PALETTE[winner] * strength[..., None]).astype(np.uint8)
    combined[~has_signal] = 0

    cv2.putText(combined, "INT-BEV  metal=orange plastic=blue rubber=magenta",
                (5,14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200,200,200), 1)
    _draw_robot_marker(combined, S)

    # Diagnostic histogram: shows the RAW intensity distribution this frame
    # (before any metal/plastic/rubber labeling) with the actual i_lo/i_hi
    # split lines drawn on it. A percentile split always produces three
    # equally-populated bands whether or not the real distribution has
    # three separated clusters -- if the split lines fall in the middle of
    # one blob instead of a real gap, or the rubber band keeps landing
    # empty because the fallback threshold (INT_RUBBER_MIN_FB) is above
    # anything this sensor/scene actually returns, this makes it visible
    # directly instead of having to infer it from the coloring alone.
    _draw_intensity_hist_inset(combined, I, i_lo, i_hi)

    return metal_gray, plastic_gray, rubber_gray, combined


def _draw_intensity_hist_inset(img, I, i_lo, i_hi, x0=8, y0=140, w=160, h=70):
    if I.size == 0:
        return
    hist, _ = np.histogram(I, bins=40, range=(0, 255))
    hist_norm = hist.astype(np.float32)
    peak = hist_norm.max()
    if peak > 0:
        hist_norm = hist_norm / peak

    cv2.rectangle(img, (x0-3, y0-16), (x0+w+3, y0+h+3), (25,25,25), -1)
    cv2.putText(img, "raw intensity histogram", (x0, y0-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180,180,180), 1, cv2.LINE_AA)

    bar_w = w / len(hist_norm)
    for i, v in enumerate(hist_norm):
        bh = int(v * h)
        x = int(x0 + i*bar_w)
        cv2.rectangle(img, (x, y0+h-bh), (int(x+bar_w)-1, y0+h), (150,150,150), -1)

    def xpos(val):
        return int(x0 + np.clip(val, 0, 255)/255.0 * w)

    lo_x, hi_x = xpos(i_lo), xpos(i_hi)
    cv2.line(img, (lo_x, y0), (lo_x, y0+h), (0,140,255), 1)    # metal|plastic, orange
    cv2.line(img, (hi_x, y0), (hi_x, y0+h), (255,0,220), 1)    # plastic|rubber, magenta
    cv2.putText(img, f"lo={i_lo:.0f}", (max(0,lo_x-14), y0+h+13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.26, (0,140,255), 1, cv2.LINE_AA)
    cv2.putText(img, f"hi={i_hi:.0f}", (max(0,hi_x-4), y0+h+13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.26, (255,0,220), 1, cv2.LINE_AA)


def _fit_rect_robust(mask_gray, hole_cx, hole_cy, scale,
                     inner_x_m=0.35, inner_y_m=0.35,
                     outer_x_m=1.9,  outer_y_m=1.5, thresh=15,
                     min_pts=30):
    """PCA orientation + trimmed-percentile extents on a rectangular ring
    around the hole.

    Why not convexHull + minAreaRect (old method): a SINGLE stray ground
    pixel in the ring becomes a hull vertex and rotates/inflates the whole
    rectangle — that's the misalignment seen on the BEV panel. PCA uses
    ALL pixels to vote on the orientation, and the 2..98 percentile
    extents ignore outliers, so the rect hugs the actual pixel mass."""
    ys, xs = np.where(mask_gray > thresh)
    if len(xs) < min_pts:
        return False, None

    dxm = (xs - hole_cx) / scale
    dym = (ys - hole_cy) / scale
    ring = ((np.abs(dxm) < outer_x_m) & (np.abs(dym) < outer_y_m) &
            ((np.abs(dxm) > inner_x_m) | (np.abs(dym) > inner_y_m)))
    xs, ys = xs[ring], ys[ring]
    if len(xs) < min_pts:
        return False, None

    pts  = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0)
    p    = pts - mean
    cov  = np.cov(p.T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]      # PCA initial guess
    angle0 = float(np.degrees(np.arctan2(major[1], major[0])))

    # Refine: PCA of a rectangle RING gets pulled a few degrees toward
    # the diagonal by the dense tyre blobs at the corners. Sweep +-20 deg
    # around the PCA guess and keep the angle whose trimmed bounding box
    # has minimum area (rotating-calipers on robust extents).
    best = None
    for da in np.arange(-20.0, 20.5, 1.0):
        th = np.radians(angle0 + da)
        ca, sa = np.cos(th), np.sin(th)
        a = p[:,0]*ca + p[:,1]*sa
        b = -p[:,0]*sa + p[:,1]*ca
        a0, a1 = np.percentile(a, [2, 98])
        b0, b1 = np.percentile(b, [2, 98])
        area = (a1-a0) * (b1-b0)
        if best is None or area < best[0]:
            best = (area, angle0+da, a0, a1, b0, b1)

    _, angle, a0, a1, b0, b1 = best
    th = np.radians(angle)
    major = np.array([np.cos(th),  np.sin(th)])
    minor = np.array([-np.sin(th), np.cos(th)])
    L  = float(a1 - a0)
    Wd = float(b1 - b0)
    if Wd > L:                                   # keep long side = major
        L, Wd = Wd, L
        major, minor = minor, major
        angle += 90.0
        a0, a1, b0, b1 = b0, b1, a0, a1
    center = mean + ((a0+a1)/2.0)*major + ((b0+b1)/2.0)*minor
    rect = ((float(center[0]), float(center[1])), (L, Wd), float(angle))
    return True, rect


class RectSmoother:
    """EMA over rect center/size/orientation across frames.
    Orientation is averaged on the doubled-angle circle (a rectangle's
    long axis is ambiguous mod 180 deg, so 179 -> 1 must NOT average to 90)."""
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self._s    = None
        self.n_updates = 0   # consecutive updates since last reset() --
                              # used elsewhere as a "has this rect actually
                              # converged" gate, since the EMA itself has no
                              # notion of stability, only recency.

    def reset(self):
        self._s = None
        self.n_updates = 0

    def update(self, rect):
        (cx, cy), (rw, rh), ang = rect
        if rh > rw:                       # canonical: long side first
            rw, rh = rh, rw
            ang += 90.0
        a2  = np.radians(2.0 * ang)
        vec = np.array([np.cos(a2), np.sin(a2)])
        self.n_updates += 1
        if self._s is None:
            self._s = [cx, cy, rw, rh, vec]
        else:
            s, al = self._s, self.alpha
            s[0] += al*(cx - s[0]);  s[1] += al*(cy - s[1])
            s[2] += al*(rw - s[2]);  s[3] += al*(rh - s[3])
            s[4]  = (1-al)*s[4] + al*vec
            n = np.linalg.norm(s[4])
            if n > 1e-6:
                s[4] = s[4] / n
        cx, cy, rw, rh, vec = self._s
        ang_s = float(np.degrees(np.arctan2(vec[1], vec[0])) / 2.0)
        return ((float(cx), float(cy)), (float(rw), float(rh)), ang_s)


# persists across frames so the drawn rect settles instead of jittering
_CHASSIS_SMOOTHER = RectSmoother(alpha=0.25)


def _find_car_rect_intensity(metal_gray, plastic_gray, rubber_gray,
                             hole_cx, hole_cy, scale, annotated=None):
    """Corrected material logic:
         chassis rect  <- PERIMETER band (plastic frame + rubber tyres,
                          both HIGH-ish intensity) — traces the true
                          outer footprint
         battery rect  <- METAL band (LOW intensity, inboard near hole)
         wheel corners <- chassis rect corners, confirmed where RUBBER
                          dominates PLASTIC (both are high-intensity, so
                          the old rubber-vs-metal check told us nothing)"""
    h, w = metal_gray.shape

    # ── 1. chassis footprint from the perimeter band ─────────────
    perimeter = cv2.max(plastic_gray, rubber_gray)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    perimeter = cv2.morphologyEx(perimeter, cv2.MORPH_CLOSE, k)
    # undo the per-band dilation so the fit hugs the true footprint
    perimeter = cv2.erode(perimeter,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))

    ok, rect = _fit_rect_robust(perimeter, hole_cx, hole_cy, scale,
                                inner_x_m=0.35, inner_y_m=0.35,
                                outer_x_m=1.9,  outer_y_m=1.5)
    if not ok:
        # last resort: fit whatever the metal band gives (battery-sized,
        # will be clamped up to CAR_WB_MIN/TR_MIN below)
        ok, rect = _fit_rect_robust(metal_gray, hole_cx, hole_cy, scale,
                                    inner_x_m=0.10, inner_y_m=0.10,
                                    outer_x_m=1.5,  outer_y_m=1.2,
                                    thresh=10, min_pts=15)
        if not ok:
            return False, None, {}

    (cx,cy),(bw,bh),angle = rect
    wb_m = max(bw,bh)/scale
    tr_m = min(bw,bh)/scale
    wb_c = float(np.clip(wb_m, CAR_WB_MIN, CAR_WB_MAX))
    tr_c = float(np.clip(tr_m, CAR_TR_MIN, CAR_TR_MAX))
    rect = ((cx,cy),
            (wb_c*scale if bw>=bh else tr_c*scale,
             wb_c*scale if bh>=bw else tr_c*scale), angle)

    # temporal smoothing: rect converges onto the car over ~10 frames
    rect = _CHASSIS_SMOOTHER.update(rect)

    # ── 2. battery footprint from the metal (LOW) band ───────────
    bat_ok, bat_rect = _fit_rect_robust(metal_gray, hole_cx, hole_cy, scale,
                                        inner_x_m=0.0, inner_y_m=0.0,
                                        outer_x_m=1.2, outer_y_m=1.0,
                                        thresh=10, min_pts=15)

    # ── 3. wheel corners from chassis rect, geometric + fixed intensity ──
    # A tyre almost never sits exactly on the geometric corner: the
    # rectangle is fit to the frame/perimeter, but the rubber return is
    # a small patch that can be 10-20 cm inboard or offset. So instead
    # of averaging a patch centred on the corner (which dilutes a small
    # tyre blob with surrounding empty pixels -> mean stays under 15),
    # we take the PEAK rubber response in a search window.
    #
    # Two things changed from the old single OR'd rule
    # ((rubber_peak>30 and area>=4) or (rubber_peak>60 and rubber_peak>=
    # plastic_peak)): first, that rule let a corner "confirm" purely on
    # an absolute rubber floor with no comparison to plastic at all --
    # a corner reading rubber=63/plastic=170 (plastic clearly dominant)
    # still passed. Since each band is independently normalized to its
    # own per-frame max (create_intensity_bev's to_u8), an absolute peak
    # value isn't a real confidence measure on its own; only a genuine
    # margin over the competing band is. Second, once the chassis rect
    # itself has converged over enough frames (RectSmoother.n_updates,
    # matching the "~10 frames" convergence the smoother already
    # documents), the box geometry is itself trustworthy -- a corner is
    # only rejected then if intensity actively CONTRADICTS it (no rubber
    # response there at all), rather than needing intensity to re-win
    # the case from scratch on every single frame.
    RUBBER_DOMINANCE_MARGIN = 15.0   # rubber must clear plastic by this much
    STABLE_FRAMES_FOR_GEOM  = 10     # matches _CHASSIS_SMOOTHER's own comment
    rect_is_stable = _CHASSIS_SMOOTHER.n_updates >= STABLE_FRAMES_FOR_GEOM

    box_pts   = cv2.boxPoints(rect)
    search_px = max(6, int(0.45*scale))   # ~45 cm search window
    wheel_conf   = {}
    corner_names = ["FL","FR","RL","RR"]

    for i, pt in enumerate(box_pts):
        px, py = int(round(pt[0])), int(round(pt[1]))
        r0, r1 = max(0, py-search_px), min(h, py+search_px)
        c0, c1 = max(0, px-search_px), min(w, px+search_px)
        name = corner_names[i]
        if r1 > r0 and c1 > c0:
            rub_win = rubber_gray [r0:r1, c0:c1]
            pla_win = plastic_gray[r0:r1, c0:c1]
            met_win = metal_gray  [r0:r1, c0:c1]
            rubber_peak  = float(rub_win.max())
            rubber_area  = int((rub_win > 40).sum())   # pixels clearly lit
            plastic_peak = float(pla_win.max())
            metal_peak   = float(met_win.max())

            rubber_wins_intensity = (rubber_peak > 30 and
                                      rubber_peak >= plastic_peak + RUBBER_DOMINANCE_MARGIN)
            if rect_is_stable:
                # box geometry is already trustworthy; only reject a
                # corner where there is no rubber response there at all
                confirmed = (rubber_area >= 2) or rubber_wins_intensity
            else:
                # box not yet established -- intensity has to carry it,
                # and has to genuinely win, not just clear a floor
                confirmed = rubber_wins_intensity
            wheel_conf[name] = {
                "confirmed":     confirmed,
                "rubber_score":  rubber_peak,
                "rubber_area":   rubber_area,
                "plastic_score": plastic_peak,
                "metal_score":   metal_peak,
                "px": px, "py": py,
            }
        else:
            wheel_conf[name] = {"confirmed": False,
                                "rubber_score":0.,"rubber_area":0,
                                "plastic_score":0.,"metal_score":0.,
                                "px":px,"py":py}

    # Gate EVERYTHING on rect_is_stable together, not just wheel confirmation:
    # box drawing, wheel circles, and the (True, rect, wheel_conf) returned
    # to the caller -- which is what detect_bev() turns into a car_landmark
    # detection and, further downstream, /car_landmark_pose. Previously the
    # green/pink boxes were drawn and published as soon as ANY single-frame
    # fit succeeded, even a noisy one; nothing downstream ever saw or acted
    # on an unconfirmed fit now.
    if rect_is_stable:
        if annotated is not None:
            _dbg = " ".join(
                f"{n}:{'Y' if wheel_conf[n]['confirmed'] else 'n'}"
                f"(r{wheel_conf[n]['rubber_score']:.0f}/a{wheel_conf[n]['rubber_area']}"
                f"/p{wheel_conf[n]['plastic_score']:.0f})"
                for n in corner_names)
            cv2.putText(annotated, f"{_dbg}",
                        (5, annotated.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0,200,255), 1, cv2.LINE_AA)

            box = np.int32(cv2.boxPoints(rect))
            cv2.drawContours(annotated,[box],0,(0,220,80),2)        # chassis
            if bat_ok:
                bbox = np.int32(cv2.boxPoints(bat_rect))
                cv2.drawContours(annotated,[bbox],0,(255,0,255),1)  # battery
                (bx,by),_,_ = bat_rect
                cv2.putText(annotated,"BAT",(int(bx)+4,int(by)+4),
                            cv2.FONT_HERSHEY_SIMPLEX,0.28,(255,0,255),1,cv2.LINE_AA)
            for nm,wc in wheel_conf.items():
                col = (0,165,255) if wc["confirmed"] else (60,60,180)
                cv2.circle(annotated,(wc["px"],wc["py"]),9,col,2)
                cv2.putText(annotated,
                            f"{nm} r={wc['rubber_score']:.0f}/p={wc['plastic_score']:.0f}",
                            (wc["px"]+4,wc["py"]-4),
                            cv2.FONT_HERSHEY_SIMPLEX,0.24,col,1,cv2.LINE_AA)
        return True, rect, wheel_conf

    if annotated is not None:
        cv2.putText(annotated,
                    f"stabilizing {_CHASSIS_SMOOTHER.n_updates}/{STABLE_FRAMES_FOR_GEOM}",
                    (5, annotated.shape[0]-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0,140,255), 1, cv2.LINE_AA)
    return False, None, {}


def _draw_robot_marker(img, size, label=""):
    cx, cy = size // 2, size // 2
    cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
    cv2.arrowedLine(img, (cx, cy), (cx, cy - 20),
                    (255, 255, 255), 1, tipLength=0.35)
    if label:
        cv2.putText(img, label, (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)


# ================================================================
# DETECTION PARAMETERS
# ================================================================
BEV_SCALE = 100.0

HOLE_AREA_MIN_Q  = 400
HOLE_AREA_FULL_Q = 2500

CAR_INNER_RADIUS_M  = 0.5
CAR_OUTER_RADIUS_M  = 2.8
CAR_THRESH          = 35
CAR_WB_MIN, CAR_WB_MAX = 1.8, 2.75
CAR_TR_MIN, CAR_TR_MAX = 0.8, 2.0
CAR_CLOSE_K  = 9

# Publish floor for the fitted chassis rectangle ("green box"). RectSmoother
# EMAs the box in from whatever the first few frames find, so early on it can
# report a wheelbase well under the true ~2.7-2.75m before converging -- that
# partial/undersized box shouldn't go out to consumers as if it were a
# confirmed car detection. Below CAR_WB_MIN..CAR_WB_MAX it's still being
# clipped up to CAR_WB_MIN, so this is a separate, stricter "trust it yet"
# gate, not a re-statement of the clip range.
WB_PUBLISH_MIN_M = 2.7

ARCH_TOP_FRAC   = 0.60
ARCH_COL_WIN    = 20
ARCH_MIN_POINTS = 15


BLIND_HOLE_RADIUS_M     = 0.55   # same physical blind-cone radius the old
                                  # BEV contour test used to anchor to
BLIND_HOLE_EMPTY_FRAC   = 0.35   # trigger once >=35% of that disk is empty


def _detect_blind_spot_hole(bev_img, scale,
                            radius_m=BLIND_HOLE_RADIUS_M,
                            min_empty_frac=BLIND_HOLE_EMPTY_FRAC):
    """Hole test that doesn't require the gap to be a closed shape.

    _detect_hole() finds an enclosed contour of empty pixels via
    cv2.findContours(..., RETR_EXTERNAL) -- that only works when the ring
    of returns wraps all the way around the gap. A real blind cone is
    often a half-circle/crescent instead (part of the ring occluded, or
    simply outside this scan's azimuth coverage), so the "empty" region
    connects straight through to the surrounding background with no
    enclosing boundary at all, and the contour test never finds anything
    to measure. A LiDAR blind spot is fundamentally "no returns near the
    sensor", not "a closed loop of returns around a gap" -- so test that
    directly: what fraction of a small disk centered on the robot has no
    return, independent of the empty region's shape or connectivity."""
    h, w = bev_img.shape[:2]
    ou, ov = w // 2, h // 2
    r_px = int(radius_m * scale)

    gray = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY)
    Yg, Xg = np.ogrid[:h, :w]
    disk = (Xg - ou) ** 2 + (Yg - ov) ** 2 <= r_px ** 2

    n_disk = int(disk.sum())
    if n_disk == 0:
        return False, float(ou), float(ov), 0.

    empty = disk & (gray <= 40)
    n_empty = int(empty.sum())
    if n_empty / n_disk < min_empty_frac:
        return False, float(ou), float(ov), 0.

    ys, xs = np.where(empty)
    ecx = float(xs.mean()) if len(xs) else float(ou)
    ecy = float(ys.mean()) if len(ys) else float(ov)
    return True, ecx, ecy, float(n_empty)


# ================================================================
# FIND CAR RECTANGLE
# ================================================================
def _find_car_rect(gray_bev, hole_cx, hole_cy, scale, annotated=None,
                   metal_gray=None):
    h, w = gray_bev.shape
    ou, ov = w//2, h//2

    inner_px = int(0.30 * scale)
    outer_px = int(1.8  * scale)

    Yg, Xg = np.ogrid[:h, :w]
    dist2  = (Xg - hole_cx)**2 + (Yg - hole_cy)**2
    donut  = (dist2 >= inner_px**2) & (dist2 <= outer_px**2)

    if metal_gray is not None and metal_gray.max() > 0:
        src = metal_gray.astype(np.float32)
    else:
        src = gray_bev.astype(np.float32)
        vals = src[donut]
        if len(vals) > 20:
            thr = float(np.percentile(vals, 55))
            src = np.clip(src - thr, 0, 255)

    src_masked = src * donut.astype(np.float32)

    dy = Yg.astype(np.float32) - hole_cy
    dx = Xg.astype(np.float32) - hole_cx
    angles = np.arctan2(dy, dx)

    N_SECTORS = 8
    sector_sz = 2 * np.pi / N_SECTORS
    sector_peaks = []

    for s in range(N_SECTORS):
        a_lo = -np.pi + s * sector_sz
        a_hi = a_lo + sector_sz
        sector_mask = donut & (angles >= a_lo) & (angles < a_hi)
        patch = src_masked * sector_mask.astype(np.float32)
        if patch.max() < 5:
            continue
        py_pk, px_pk = np.unravel_index(np.argmax(patch), patch.shape)
        sector_peaks.append((int(px_pk), int(py_pk), float(patch[py_pk, px_pk])))

    if len(sector_peaks) < 4:
        bright_mask = (src_masked > src_masked[donut].mean()
                       if donut.any() else np.zeros_like(src_masked, bool))
        ys, xs = np.where(bright_mask > 0)
        if len(xs) < 10:
            if annotated is not None:
                cv2.putText(annotated, "chassis: too few peaks",
                            (10,h-12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.30, (100,100,255), 1, cv2.LINE_AA)
            return False, None
        pts  = np.column_stack([xs,ys]).astype(np.int32)
        hull = cv2.convexHull(pts.reshape(-1,1,2))
        rect = cv2.minAreaRect(hull)
    else:
        sector_peaks.sort(key=lambda p: -p[2])

        def angle_from_hole(p):
            return np.arctan2(p[1]-hole_cy, p[0]-hole_cx)
        sector_peaks_sorted = sorted(sector_peaks, key=angle_from_hole)

        peak_pts = np.array([[p[0],p[1]] for p in sector_peaks_sorted],
                            dtype=np.int32)

        hull = cv2.convexHull(peak_pts.reshape(-1,1,2))
        rect = cv2.minAreaRect(hull)

        cx_peaks = float(peak_pts[:,0].mean())
        cy_peaks = float(peak_pts[:,1].mean())

        if len(sector_peaks_sorted) >= 4:
            n = len(sector_peaks_sorted)
            step = n // 4
            corners = [sector_peaks_sorted[i*step] for i in range(4)]
            corners_sorted = sorted(corners, key=angle_from_hole)
            corner_pts = np.array([[p[0],p[1]] for p in corners_sorted],
                                  dtype=np.int32)
        else:
            corner_pts = peak_pts

        hull_c = cv2.convexHull(corner_pts.reshape(-1,1,2))
        rect   = cv2.minAreaRect(hull_c)
        (cx,cy),(bw,bh),angle = rect

        if annotated is not None:
            for px,py,bv in sector_peaks:
                cv2.circle(annotated,(px,py),5,(255,200,0),1)
            for i in range(len(corner_pts)):
                pt1 = tuple(corner_pts[i])
                pt2 = tuple(corner_pts[(i+1)%len(corner_pts)])
                cv2.line(annotated, pt1, pt2, (0,0,255), 2)
            for pt in corner_pts:
                cv2.circle(annotated, tuple(pt), 8, (0,0,255), 2)

    (cx,cy),(bw,bh),angle = rect
    wb_m = max(bw,bh)/scale
    tr_m = min(bw,bh)/scale
    wb_c = float(np.clip(wb_m, CAR_WB_MIN, CAR_WB_MAX))
    tr_c = float(np.clip(tr_m, CAR_TR_MIN, CAR_TR_MAX))
    rect = ((cx,cy),(wb_c*scale if bw>=bh else tr_c*scale,
                     wb_c*scale if bh>=bw else tr_c*scale), angle)

    dim_ok = not (wb_m > CAR_WB_MAX or tr_m > CAR_TR_MAX)
    if annotated is not None:
        lbl = f"wb={wb_c:.1f}m tr={tr_c:.1f}m"
        col = (0,200,80) if dim_ok else (0,100,255)
        if not dim_ok:
            lbl += f" SUSPECT({wb_m:.1f}x{tr_m:.1f})"
        cv2.putText(annotated, lbl, (int(cx)+6,int(cy)-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, col, 1, cv2.LINE_AA)

    return dim_ok, rect


# ================================================================
# WHEEL POSITIONS FROM RECTANGLE CORNERS
# ================================================================
def _wheels_from_rect(rect, scale, gray_bev, annotated=None):
    h, w = gray_bev.shape
    ou, ov = w//2, h//2

    box_pts = cv2.boxPoints(rect)
    (cx,cy),(bw,bh),angle = rect

    names  = ["FL","FR","RL","RR"]
    sorted_pts = sorted(box_pts, key=lambda p: (p[1]>cy, p[0]>cx))
    labels = {0:"FL", 1:"FR", 2:"RL", 3:"RR"}

    patch_px = max(4, int(0.35 * scale))
    wheels   = []

    for i, pt in enumerate(box_pts):
        px, py = int(round(pt[0])), int(round(pt[1]))

        x_m = (ov - py) / scale
        y_m = -(px - ou) / scale

        r0,r1 = max(0,py-patch_px), min(h,py+patch_px)
        c0,c1 = max(0,px-patch_px), min(w,px+patch_px)
        void_ratio = 0.
        confirmed  = False
        if r1>r0 and c1>c0:
            patch      = gray_bev[r0:r1, c0:c1]
            void_ratio = float((patch < 35).sum()) / max(patch.size,1)
            confirmed = void_ratio >= 0.50

        wheels.append({
            "name":       names[i],
            "cx_px":float(px),"cy_px":float(py),
            "x_m":float(x_m),"y_m":float(y_m),
            "void_ratio":float(void_ratio),
            "confirmed":confirmed,
        })

        if annotated is not None and 0<=px<w and 0<=py<h:
            col = (0,165,255) if confirmed else (80,80,200)
            cv2.circle(annotated, (px,py), 9, col, 2)
            cv2.rectangle(annotated,(c0,r0),(c1,r1), col, 1)
            cv2.putText(annotated,
                        f"{names[i]} {void_ratio:.2f}",
                        (px+4,py-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26, col, 1, cv2.LINE_AA)

    return wheels


# ================================================================
# SIDE-VIEW UNDER-CAR CHECK
# ================================================================
def _check_under_car(img_gray, view_name, annotated=None):
    H, W = img_gray.shape

    band_r0 = int(0.15 * H)
    band_r1 = int(0.60 * H)
    band    = img_gray[band_r0:band_r1, :]

    occupied     = (band > 20)
    col_occupied = occupied.any(axis=0)
    width_frac   = float(col_occupied.sum()) / W

    confirmed = False
    top_rows  = []
    for c in range(W):
        rows = np.where(occupied[:, c])[0]
        if len(rows) > 0:
            top_rows.append(rows.min())

    flatness = 0.
    if len(top_rows) > 10:
        flatness = 1.0 - min(float(np.std(top_rows)) / (band_r1 - band_r0), 1.0)
        confirmed = (width_frac >= 0.30 and flatness >= 0.55)

    if annotated is not None:
        col = (0, 200, 100) if confirmed else (60, 60, 60)
        cv2.rectangle(annotated, (0, band_r0), (W-1, band_r1), col, 1)
        label = (f"under-car OK w={width_frac:.2f} f={flatness:.2f}"
                 if confirmed else
                 f"w={width_frac:.2f} f={flatness:.2f}")
        cv2.putText(annotated, label, (6, band_r0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, col, 1, cv2.LINE_AA)

    return confirmed, float(width_frac), float(flatness)


# ================================================================
# BEV DETECTIONS
# ================================================================
def detect_bev(bev_dilated_img, scale=BEV_SCALE,
               metal_gray=None, plastic_gray=None, rubber_gray=None):
    annotated  = bev_dilated_img.copy()
    detections = []
    h, w = bev_dilated_img.shape[:2]
    ou, ov = w//2, h//2
    gray = cv2.cvtColor(bev_dilated_img, cv2.COLOR_BGR2GRAY)

    # Rectangle/car detection now runs FIRST and unconditionally, centered
    # on the robot origin (ou, ov) rather than a detected hole position.
    # Previously this whole block sat behind `if hole_ok:`, so whenever
    # _detect_hole's contour/size/distance heuristic failed -- which
    # happens often under occlusion, exactly when knowing "this is a car"
    # matters most -- the rectangle fit never even ran, regardless of how
    # complete the BEV blob looked. The ring-based fit only ever needed a
    # search center near the robot; the hole itself is constrained to sit
    # within HOLE_DIST_MAX_M of that same origin, so robot-origin is an
    # equally valid anchor and doesn't depend on the hole heuristic
    # succeeding.
    if (metal_gray is not None and plastic_gray is not None
            and rubber_gray is not None):
        _rect_result = _find_car_rect_intensity(
            metal_gray, plastic_gray, rubber_gray,
            ou, ov, scale, annotated)
        rect_ok, rect, int_wheel_conf = _rect_result[0], _rect_result[1], _rect_result[2] if len(_rect_result)>2 else {}
    else:
        rect_ok, rect = _find_car_rect(gray, ou, ov, scale, annotated,
                                        metal_gray=metal_gray)
        int_wheel_conf = {}

    # Hole detection no longer runs here -- moved to the blind-spot frame
    # (see _process_frame), which stays robust to the ring being partially
    # occluded instead of requiring a fully closed BEV contour.

    if rect_ok:
        (cx,cy),(bw,bh),ang = rect
        wb_m = max(bw,bh)/scale
        tr_m = min(bw,bh)/scale
        car_x = (ov-cy)/scale
        car_y = -(cx-ou)/scale
        # Convert the rect's angle (measured in pixel space by
        # _fit_rect_robust/cv2) into a world/robot-frame yaw by mapping a
        # second point along the rect's axis through the SAME px->world
        # formulas used for the center just above, rather than re-deriving
        # the rotation relationship between pixel and metric axes by hand
        # (u=-Y*scale+.., v=-X*scale+.. is not a plain coordinate swap, so
        # a hand-derived angle offset is an easy place to get subtly wrong).
        ang_rad = np.radians(ang)
        cx2, cy2 = cx + np.cos(ang_rad), cy + np.sin(ang_rad)
        car_x2 = (ov-cy2)/scale
        car_y2 = -(cx2-ou)/scale
        car_angle_rad = float(np.arctan2(car_y2-car_y, car_x2-car_x))

        if int_wheel_conf:
            n_conf = sum(1 for wc in int_wheel_conf.values()
                         if wc["confirmed"])
            for nm, wc in int_wheel_conf.items():
                px,py = wc["px"],wc["py"]
                x_m = (ov-py)/scale; y_m = -(px-ou)/scale
                detections.append({
                    "type":         "wheel_corner",
                    "distance":     float(np.hypot(x_m,y_m)),
                    "x":x_m,"y":y_m,
                    "rubber_score": wc["rubber_score"],
                    "plastic_score": wc.get("plastic_score", 0.),
                    "metal_score":  wc["metal_score"],
                    "corner":       nm,
                    "confirmed":    wc["confirmed"],
                    "view":         "bev",
                })
        else:
            wheels = _wheels_from_rect(rect, scale, gray, annotated)
            n_conf = sum(1 for wh in wheels if wh["confirmed"])
            for wh in wheels:
                detections.append({
                    "type":      "wheel_corner",
                    "distance":  float(np.hypot(wh["x_m"],wh["y_m"])),
                    "x":wh["x_m"],"y":wh["y_m"],
                    "void_ratio":wh["void_ratio"],
                    "corner":    wh["name"],
                    "confirmed": wh["confirmed"],
                    "view":      "bev",
                })

        col = (0,255,0) if n_conf>=2 else (0,180,120)
        cu,cv_ = int(ou-car_y*scale), int(ov-car_x*scale)
        cs=12
        cv2.line(annotated,(cu-cs,cv_),(cu+cs,cv_),col,2)
        cv2.line(annotated,(cu,cv_-cs),(cu,cv_+cs),col,2)
        cv2.putText(annotated,
                    f"CAR {car_x:.1f},{car_y:.1f}m wb={wb_m:.1f} tr={tr_m:.1f} w={n_conf}/4",
                    (cu+14,cv_-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, col, 1, cv2.LINE_AA)

        if (metal_gray is not None and plastic_gray is not None
                and rubber_gray is not None):
            overlay = annotated.copy()
            metal_m = metal_gray > 20            # battery (LOW int)
            overlay[metal_m, 0] = np.clip(
                overlay[metal_m, 0].astype(np.int16) + 80, 0, 255).astype(np.uint8)
            plastic_m = plastic_gray > 20        # frame (MID int)
            overlay[plastic_m, 1] = np.clip(
                overlay[plastic_m, 1].astype(np.int16) + 60, 0, 255).astype(np.uint8)
            rubber_m = rubber_gray > 20          # tyres (HIGH int)
            overlay[rubber_m, 2] = np.clip(
                overlay[rubber_m, 2].astype(np.int16) + 80, 0, 255).astype(np.uint8)
            cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
            cv2.putText(annotated,
                        "INT: blue=metal(bat) green=plastic red=rubber",
                        (5, h-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.28, (180,180,180), 1, cv2.LINE_AA)

        detections.append({
            "type":"car_landmark",
            "distance":float(np.hypot(car_x,car_y)),
            "x":float(car_x),"y":float(car_y),
            "wheelbase":float(wb_m),"track":float(tr_m),
            "n_wheels":n_conf,"corner_conf":n_conf/4.,
            "view":"bev",
            "angle_rad":car_angle_rad,
        })
    else:
        cv2.putText(annotated, "car rect fit failed",
                    (10,h-12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,(100,100,255),1,cv2.LINE_AA)

    return annotated, detections


# ================================================================
# SIDE-VIEW DETECTIONS
# ================================================================
def detect_perspective(args):
    (img_color, depth_map, int_map), view_name, hole_detected = args
    annotated  = img_color.copy()
    detections = []
    H, W = img_color.shape[:2]

    if not hole_detected:
        cv2.putText(annotated, "waiting for hole",
                    (8, H-8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.28, (40,40,40), 1, cv2.LINE_AA)
        return annotated, detections

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    confirmed, width_frac, flatness = _check_under_car(gray, view_name, annotated)

    if confirmed:
        detections.append({
            "type":     "under_car",
            "distance": float(width_frac),
            "x":        float(width_frac),
            "y":        float(flatness),
            "view":     view_name,
        })

    status = "under-car" if confirmed else "clear"
    cv2.putText(annotated, status, (8, H-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                (0,200,100) if confirmed else (60,60,60), 1, cv2.LINE_AA)

    return annotated, detections


# ================================================================
# CROSS-CHECK
# ================================================================
def cross_check_detections(bev_dets, side_dets_by_view):
  import numpy as np

def cross_check_detections(bev_dets, side_dets_by_view):
    """
    Compute an overall confidence score from all detection sources.

    Contributions
    -------------
    Hole detection      : 30%
    Car rectangle       : 40%
    Wheel detection     : 20%
    Side-view evidence  : 10%

    Quality is graded instead of binary. A nonlinear mapping prevents
    confidence from saturating at 1.0 unless every detector is extremely
    reliable.
    """

    hole_dets = [d for d in bev_dets if d["type"] == "lidar_hole"]
    car_dets  = [d for d in bev_dets if d["type"] == "car_landmark"]

    n_wheels = sum(
        1 for d in bev_dets
        if d["type"] == "wheel_corner" and d.get("confirmed", False)
    )

    under_cnt = sum(
        1
        for view in side_dets_by_view.values()
        for d in view
        if d["type"] == "under_car"
    )

    raw_score = 0.0
    evidence = []

    # -------------------------------------------------------------
    # Hole quality
    # -------------------------------------------------------------
    hole_q = 0.0
    if hole_dets:
        h_area = max(d.get("area_px", 0.0) for d in hole_dets)

        hole_q = np.clip(
            (h_area - HOLE_AREA_MIN_Q) /
            max(HOLE_AREA_FULL_Q - HOLE_AREA_MIN_Q, 1e-6),
            0.0,
            1.0,
        )

        # nonlinear scaling
        hole_q = hole_q ** 2

        raw_score += 0.30 * hole_q
        evidence.append(
            f"Hole={h_area:.0f}px (q={hole_q:.2f})"
        )

    # -------------------------------------------------------------
    # Car rectangle quality
    # -------------------------------------------------------------
    fit_q = 0.0
    if car_dets:
        fit_q = max(
            d.get("fit_quality", 0.0)
            for d in car_dets
        )

        fit_q = np.clip(fit_q, 0.0, 1.0)

        # nonlinear scaling
        fit_q = fit_q ** 2

        raw_score += 0.40 * fit_q
        evidence.append(
            f"Rect(q={fit_q:.2f})"
        )

    # -------------------------------------------------------------
    # Wheel confidence
    # -------------------------------------------------------------
    wheel_lookup = {
        0: 0.00,
        1: 0.15,
        2: 0.45,
        3: 0.75,
        4: 1.00,
    }

    wheel_q = wheel_lookup.get(min(n_wheels, 4), 1.0)

    raw_score += 0.20 * wheel_q

    evidence.append(
        f"Wheels={n_wheels}/4 (q={wheel_q:.2f})"
    )

    # -------------------------------------------------------------
    # Side-view evidence
    # -------------------------------------------------------------
    side_q = np.clip(under_cnt / 4.0, 0.0, 1.0)

    # slightly nonlinear
    side_q = side_q ** 1.5

    raw_score += 0.10 * side_q

    evidence.append(
        f"Sides={under_cnt} (q={side_q:.2f})"
    )

    # -------------------------------------------------------------
    # Final confidence
    # -------------------------------------------------------------
    confidence = np.tanh(1.5 * raw_score)

    evidence.append(
        f"Raw={raw_score:.2f}"
    )

    evidence.append(
        f"Conf={confidence:.2f}"
    )

    return float(confidence), evidence

    return min(score,1.0), evidence


# ================================================================
# MAIN NODE  — freeze-proof version
# ================================================================
class IntensityLandmarkNode(Node):

    def __init__(self):
        super().__init__("intensity_landmark_node")

        self.proj_width  = 320
        self.proj_height = 160
        self.bev_size    = 400
        self.bev_scale   = BEV_SCALE * (400.0/600.0)

        point_radius = 3
        k = point_radius * 2 - 1
        self._dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) \
                              if point_radius > 1 else None

        # Short rolling buffer of raw /dynamic_points scans, time-windowed
        # (not spatially-persistent like the voxel layer below). Feeds the
        # blind-spot panel: a single scan alone is sparse enough to look
        # like isolated "moving points" rather than a solid surface, but
        # the full persistent/accumulated layer is built from minutes of
        # scans re-projected into the current pose and can back-fill the
        # sensor's own blind cone with stale pre-maneuver data (see
        # build_blind_spot_view()'s docstring). A few seconds is enough to
        # densify a single scan into a proper-looking layer while staying
        # short enough that it still reflects the CURRENT occlusion state
        # once the robot has settled after a maneuver.
        BLIND_BUFFER_SECONDS = 2.0
        self.persistence     = BLIND_BUFFER_SECONDS
        self.buffer_lock     = threading.Lock()
        self.buffer          = deque()

        # FIX (long-run hollowing): a persistent voxel layer that remembers
        # static structure. The 5 s rolling buffer alone caused the
        # accumulated image to slowly erode over ~15 min: whenever the
        # dynamic filter over-removed a spot for a few frames, that region
        # was absent from every scan in the 5 s window, so once the good
        # observations aged out there was nothing left holding it. This
        # layer keeps a last-seen point per voxel across many scans.
        #
        # Eviction is SPATIAL, not time-based: a voxel is dropped once its
        # position (re-projected into the CURRENT robot-centric frame) falls
        # outside the same ROI create_bev_dilated() already renders
        # (X in (-3,8), Y in (-5,5), i.e. literally "out of frame"), not
        # after a fixed number of seconds unseen. The previous policy
        # (PERSIST_VOX_SECONDS=90) evicted a voxel the instant it went 90 s
        # without being re-observed -- which is exactly what happens to the
        # car underside behind the transmitter's permanent blind spot, so
        # it visually vanished from the display after 90 s even though nothing
        # moved and it was still sitting right there in frame. Spatial
        # eviction means a point that stays in view is never dropped no
        # matter how long the robot sits still, and a point genuinely
        # leaves the store only once the robot has actually driven far
        # enough that it would no longer be drawn anyway.
        self.PERSIST_VOX_SIZE       = 0.05   # 5 cm voxels
        self.PERSIST_FRAME_MARGIN_M = 0.5    # slack beyond the BEV ROI edge
                                              # so points don't flicker in/out
                                              # right at the boundary from
                                              # ordinary pose jitter
        self.persist_vox_lock    = threading.Lock()
        # array-based store (vectorized, ~4 ms/scan): parallel arrays of
        # packed voxel keys and their [t_last, x, y, z, i] rows.
        self._pv_keys = np.empty((0,), np.int64)
        self._pv_data = np.empty((0, 5), np.float32)

        self.MAX_POINTS      = 12000
        # latch thresholds: must be ABOVE under-car SLAM jitter, below
        # real driving motion. 10 cm / ~3 deg works well in practice.
        self.min_translation = 0.10
        self.min_rotation    = 0.05
        self.display_every_n = 8
        self._frame_counter  = 0
        self._hole_detected  = False

        self.current_pose       = None
        self.last_accepted_pose = None
        self._pose_lock         = threading.Lock()

        # FIX 2: Replace threading.Event with a bounded Queue for display frames.
        # maxsize=1 means the display loop always gets the LATEST frame and never
        # blocks the compute thread waiting for the GUI to drain a queue.
        self._display_queue = queue.Queue(maxsize=1)
        self._shutdown      = threading.Event()

        # FIX 3: Increase worker count to match the 4 perspective views so
        # ThreadPoolExecutor.map() can always run all 4 in parallel.
        # Also add a timeout guard so a stalled worker doesn't freeze the node.
        self._proj_pool = ThreadPoolExecutor(max_workers=4)
        self._PROJ_TIMEOUT = 2.0   # seconds — drop frame if workers stall

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(Path,        "/path",           self.path_callback,       10)

        # self.create_subscription(PointCloud2, "/cloud_registered", self.pointcloud_callback, sensor_qos)
        self.create_subscription(PointCloud2, "/dynamic_points", self.pointcloud_callback, sensor_qos)
        self.landmark_pub     = self.create_publisher(String,  "/intensity_landmarks",  10)
        self.lidar_hole_pub   = self.create_publisher(Float32, "/lidar_hole_area",      10)
        self.car_landmark_pub = self.create_publisher(String,  "/car_landmark_pose",    10)
        self.wheel_count_pub  = self.create_publisher(Float32, "/detected_wheel_count", 10)
        self.confidence_pub   = self.create_publisher(Float32, "/under_car_confidence", 10)
        # "cx,cy,wb,tr,angle_rad" -- deliberately NOT gated by conf_score>=0.40 like
        # /car_landmark_pose is. detect_bev() already only produces a
        # car_landmark detection once the rect has passed its own stability
        # gate (rect_is_stable in _find_car_rect_intensity), so this is
        # available sooner/more often for consumers that just need the
        # chassis footprint (e.g. transmitter-alignment estimation) and
        # don't need the full cross-checked confidence score.
        self.car_bbox_pub     = self.create_publisher(String,  "/car_bbox_dims",        10)

        # FIX 4: Display loop runs on the MAIN thread (via a ROS timer) to keep
        # cv2.imshow on the thread that created the window. On most platforms
        # (especially Linux/X11) calling imshow from a non-main thread eventually
        # deadlocks the X11 event loop. We poll at 30 Hz; if no frame is ready
        # the call is a fast no-op.
        self._display_timer = self.create_timer(1.0 / 30.0, self._display_tick)

        # Compute thread — keeps pointcloud_callback non-blocking
        self._compute_thread = threading.Thread(
            target=self._compute_loop, daemon=True)
        self._compute_queue = queue.Queue(maxsize=2)  # FIX 5: bounded compute queue
        self._compute_thread.start()

        self.get_logger().info("IntensityLandmarkNode ready (freeze-proof build)")

    # ─────────────────────────────────────────────────────────────
    # FIX 4 cont.: display runs as a ROS timer on the spinning thread
    def _display_tick(self):
        try:
            frame = self._display_queue.get_nowait()
        except queue.Empty:
            return
        # cv2.imshow/waitKey talk to the X11/GTK display connection -- a
        # transient hiccup there (window closed, VNC/SSH X-forwarding
        # drop) used to raise straight out of this ROS timer callback,
        # through rclpy.spin(), killing the whole node (and, under a
        # launch respawn, wiping all in-process state -- the buffer and
        # persistent voxel layer -- on restart). Detection doesn't
        # actually depend on the window, so a display error should never
        # take the node down with it.
        try:
            if frame is not None:
                cv2.imshow("Landmark Detection – Surround View", frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"display tick error: {e}", throttle_duration_sec=5.0)

    # ─────────────────────────────────────────────────────────────
    def path_callback(self, msg):
        # POSE LATCH — fixes the constant flip/rotate of the display.
        # Under a car the lidar sees a degenerate scene (ceiling + a few
        # pillars), so the SLAM's yaw estimate jitters. The old gate had
        # two bugs:
        #   1. min(abs(y1-y0), 2*pi) is NOT an angle wrap — near the
        #      +/-180 deg boundary the raw difference is ~6.28 rad, so it
        #      always passed the gate and the pose updated every message,
        #      re-spinning the whole rendered map.
        #   2. min_rotation = 0.01 rad (0.57 deg) is smaller than normal
        #      under-car jitter, so jitter passed the gate anyway.
        # Fix: proper shortest-angle wrap + thresholds jitter can't reach.
        # Real driving (>10 cm or >~3 deg) still unlocks immediately.
        #
        # Runs directly on the ROS spin thread (it's a subscription
        # callback, not behind _compute_loop's own try/except) -- an
        # unhandled exception here would propagate out of rclpy.spin()
        # and take the whole node down, so guard it the same way.
        try:
            if not msg.poses:
                return
            new_pose = msg.poses[-1].pose
            with self._pose_lock:
                if self.last_accepted_pose is None:
                    self.current_pose = self.last_accepted_pose = new_pose
                    return
                dx   = new_pose.position.x - self.last_accepted_pose.position.x
                dy   = new_pose.position.y - self.last_accepted_pose.position.y
                dist = np.hypot(dx, dy)
                def yaw(p):
                    q = p.orientation
                    return np.arctan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
                d    = yaw(new_pose) - yaw(self.last_accepted_pose)
                dyaw = abs(np.arctan2(np.sin(d), np.cos(d)))   # proper wrap
                if dist < self.min_translation and dyaw < self.min_rotation:
                    return
                self.current_pose = self.last_accepted_pose = new_pose
        except Exception as e:
            self.get_logger().error(f"path_callback error: {e}", throttle_duration_sec=5.0)

    # ─────────────────────────────────────────────────────────────
    def shift_cloud_to_robot(self, cloud):
        with self._pose_lock:
            pose = self.current_pose
        if pose is None:
            return cloud
        cx, cy = pose.position.x, pose.position.y
        q   = pose.orientation
        yaw = np.arctan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        tx  = cloud[:, 0] - cx;  ty = cloud[:, 1] - cy
        c, s = np.cos(-yaw), np.sin(-yaw)
        out       = cloud.copy()
        out[:, 0] = tx*c - ty*s
        out[:, 1] = tx*s + ty*c
        return out

    # ─────────────────────────────────────────────────────────────
    # FIX 5: pointcloud_callback just enqueues raw points — no heavy work here.
    # If the compute queue is full we drop the frame rather than blocking.
    def pointcloud_callback(self, msg):
        # Also runs on the ROS spin thread like path_callback above -- same
        # reasoning applies: an unhandled exception here would kill the
        # whole node (and wipe self.buffer / the persistent voxel layer on
        # restart), so don't let a single malformed/edge-case scan do that.
        try:
            t0     = time.time()
            points = read_points_fast(msg)
            if len(points) == 0:
                return

            # Accumulate into the time-stamped buffer (lock protects deque)
            with self.buffer_lock:
                self.buffer.append((t0, points))
                cutoff = t0 - self.persistence
                while self.buffer and self.buffer[0][0] < cutoff:
                    self.buffer.popleft()

            # Update the persistent voxel layer: stamp every observed voxel with
            # the current time and position. Voxels that keep being seen stay
            # refreshed; a voxel is only dropped once it falls out of the
            # currently-rendered frame (see the spatial prune in
            # _update_persist_vox), not after a fixed time unseen -- so a spot
            # that was briefly over-removed refills the instant it's seen again,
            # and a point that stays in view is never evicted just for sitting
            # still, while the map still can't grow without bound since the
            # in-frame region itself is spatially finite.
            self._update_persist_vox(points, t0)

            self._frame_counter += 1
            if self._frame_counter % self.display_every_n != 0:
                return

            # Snapshot for compute thread. Use the PERSISTENT layer as the
            # accumulation source (not just the 5 s buffer), so momentary
            # over-removals don't leave lasting holes.
            buf_snapshot = self._persist_snapshot(t0)
            if not buf_snapshot:
                return

            # Windowed (BLIND_BUFFER_SECONDS) snapshot for the blind-spot
            # panel: denser than any one scan, but still short enough to
            # reflect current occlusion rather than the whole persistent pass.
            with self.buffer_lock:
                blind_points = (np.vstack([p for (_, p) in self.buffer])
                                 if self.buffer else points)

            try:
                self._compute_queue.put_nowait((t0, buf_snapshot, blind_points))
            except queue.Full:
                self.get_logger().debug("compute queue full — dropping frame")
        except Exception as e:
            self.get_logger().error(f"pointcloud_callback error: {e}", throttle_duration_sec=5.0)

    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _pv_pack(k):
        # pack int voxel (i,j,k) into one int64 (21 bits each, signed-safe
        # for the ranges seen under a vehicle; ample for a local map)
        return (k[:, 0] << 42) ^ ((k[:, 1] & 0x1FFFFF) << 21) ^ (k[:, 2] & 0x1FFFFF)

    def _update_persist_vox(self, points, t0):
        if len(points) == 0:
            return
        vs = self.PERSIST_VOX_SIZE
        k = np.floor(points[:, :3] / vs).astype(np.int64)
        pk = self._pv_pack(k)
        # one representative row per occupied voxel this scan
        upk, fi = np.unique(pk, return_index=True)
        inten = (points[fi, 3] if points.shape[1] > 3
                 else np.zeros(len(fi), np.float32))
        newdata = np.column_stack([
            np.full(len(fi), t0, np.float32),
            points[fi, 0], points[fi, 1], points[fi, 2], inten
        ]).astype(np.float32)

        with self.persist_vox_lock:
            if self._pv_keys.size:
                # refresh existing voxels: drop their old rows, append new
                keep = ~np.isin(self._pv_keys, upk)
                self._pv_keys = np.concatenate([self._pv_keys[keep], upk])
                self._pv_data = np.concatenate([self._pv_data[keep], newdata])
            else:
                self._pv_keys = upk
                self._pv_data = newdata
            # prune voxels that have fallen OUT OF FRAME (occasionally):
            # re-project each voxel's world (x,y) into the current
            # robot-centric frame using the same transform as
            # shift_cloud_to_robot(), and keep it only while it still
            # falls within the ROI create_bev_dilated() actually renders
            # (X in (-3,8), Y in (-5,5), padded by PERSIST_FRAME_MARGIN_M).
            # Without a pose yet, skip pruning entirely rather than guess.
            if self._frame_counter % 20 == 0 and self._pv_keys.size:
                with self._pose_lock:
                    pose = self.current_pose
                if pose is not None:
                    cx, cy = pose.position.x, pose.position.y
                    q   = pose.orientation
                    yaw = np.arctan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
                    tx  = self._pv_data[:, 1] - cx
                    ty  = self._pv_data[:, 2] - cy
                    c, s = np.cos(-yaw), np.sin(-yaw)
                    local_x = tx*c - ty*s
                    local_y = tx*s + ty*c
                    m = self.PERSIST_FRAME_MARGIN_M
                    in_frame = ((local_x > -3.0 - m) & (local_x < 8.0 + m) &
                                (np.abs(local_y) < 5.0 + m))
                    self._pv_keys = self._pv_keys[in_frame]
                    self._pv_data = self._pv_data[in_frame]

    def _persist_snapshot(self, t0):
        """Return the persistent layer as a [(t0, points)] buffer matching
        the shape the compute path expects.

        FIX (wheels vanishing under cap): _pv_keys/_pv_data accumulate in
        ascending packed-voxel-key order (np.unique sorts on refresh), and
        that key is a linear function of (x_vox, y_vox, z_vox) -- so key
        order tracks spatial position. arr[-MAX_POINTS:] used to slice off
        the LOW end of that key range every time the store grew past the
        cap, which deterministically evicts whichever geometric extreme of
        the car sits there first -- and the wheels, sitting at the four
        corners of the footprint, are exactly the extremes; the flat
        underbody/battery near the median key was untouched. That's why
        the body stayed visible while the wheels dropped out after a
        while under the car. An unbiased random subsample costs the same
        O(N) as the slice but doesn't structurally favor the center over
        the corners."""
        with self.persist_vox_lock:
            if self._pv_data.shape[0] == 0:
                return []
            arr = self._pv_data[:, 1:5].copy()
        if len(arr) > self.MAX_POINTS:
            idx = np.random.choice(len(arr), self.MAX_POINTS, replace=False)
            arr = arr[idx]
        return [(t0, arr)]

    # ─────────────────────────────────────────────────────────────
    def _compute_loop(self):
        """Heavy processing runs here, off the ROS callback thread."""
        while not self._shutdown.is_set():
            try:
                t0, buf_snapshot, live_points = self._compute_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._process_frame(t0, buf_snapshot, live_points)
            except Exception as e:
                self.get_logger().error(f"compute error: {e}")

    # ─────────────────────────────────────────────────────────────
    def _process_frame(self, t0, buf_snapshot, live_points):
        cloud = np.vstack([p for (_, p) in buf_snapshot])

        if len(cloud) > self.MAX_POINTS:
            idx   = np.random.choice(len(cloud), self.MAX_POINTS, replace=False)
            cloud = cloud[idx]

        cloud = self.shift_cloud_to_robot(cloud)

        # ── Hole detection, from the blind-spot frame ────────────────────
        # Moved off the accumulated BEV: the old BEV contour test
        # (cv2.findContours(..., RETR_EXTERNAL)) only fires when the ring
        # of returns closes all the way around the gap, which fails
        # exactly when the transmitter occludes part of the ring -- the
        # most common real case, and the one where knowing "there's a
        # hole" matters most. _detect_blind_spot_hole() tests the fraction
        # of empty pixels in a disk around the robot instead, so it still
        # fires on a half-occluded crescent gap. It's built from
        # self.buffer's short rolling window of the CURRENT live scan
        # (see build_blind_spot_view's docstring), not the multi-minute
        # accumulated BEV, so it also reflects the current occlusion state
        # rather than a stale pre-maneuver one.
        live_cloud = self.shift_cloud_to_robot(live_points)
        blind_bev  = build_blind_spot_view(live_cloud, self.bev_size, self.bev_scale,
                                            self._dilate_kernel)
        blind_hole = _detect_blind_spot_hole(blind_bev, self.bev_scale)
        hole_ok    = blind_hole[0]
        hole_area_px = blind_hole[3] if hole_ok else 0.0
        if hole_ok:
            _, bhx, bhy, barea = blind_hole
            cv2.circle(blind_bev, (int(bhx), int(bhy)), 8, (0,0,255), 2)
            cv2.putText(blind_bev, f"HOLE {barea:.0f}px", (int(bhx)+6, int(bhy)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0,0,255), 1, cv2.LINE_AA)
        _draw_robot_marker(blind_bev, self.bev_size, "BLIND SPOT")
        self._hole_detected = hole_ok

        # BEV (all heights)
        bev_dilated = create_bev_dilated(
            cloud, self.bev_size, self.bev_scale, self._dilate_kernel)

        if hole_ok:
            metal_gray, plastic_gray, rubber_gray, int_bev_color = \
                create_intensity_bev(
                    cloud, self.bev_size, self.bev_scale, self._dilate_kernel)
        else:
            metal_gray = plastic_gray = rubber_gray = None
            int_bev_color = np.zeros((self.bev_size, self.bev_size, 3), np.uint8)
            cv2.putText(int_bev_color, "INT-BEV: active under car only",
                        (10, self.bev_size//2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (60,60,60), 1, cv2.LINE_AA)

        if bev_dilated is not None:
            bev_d_ann, bev_dets = detect_bev(
                bev_dilated, self.bev_scale,
                metal_gray=metal_gray, plastic_gray=plastic_gray,
                rubber_gray=rubber_gray)
        else:
            bev_d_ann     = np.zeros((self.bev_size, self.bev_size, 3), np.uint8)
            bev_dets      = []
            int_bev_color = np.zeros((self.bev_size, self.bev_size, 3), np.uint8)

        # Hole result (from the blind-spot frame, computed above) still
        # feeds cross_check_detections' hole_q the same way a BEV
        # "lidar_hole" detection used to.
        if hole_ok:
            bev_dets.append({
                "type": "lidar_hole", "distance": float(hole_area_px),
                "x": 0., "y": 0., "area_px": float(hole_area_px),
                "view": "blind_spot",
            })

        W_pv, H_pv = self.proj_width, self.proj_height
        _blank = (np.zeros((H_pv,W_pv,3),np.uint8),
                  np.full((H_pv,W_pv),np.inf,np.float32),
                  np.zeros((H_pv,W_pv),np.float32))

        if hole_ok:
            front_t, rear_t, left_t, right_t = four_views(
                cloud, W_pv, H_pv, self._dilate_kernel)
        else:
            front_t = rear_t = left_t = right_t = _blank

        # FIX 6: Use submit+result with per-future timeouts instead of map().
        # This prevents a single stuck worker from hanging the whole call forever.
        futures = {
            name: self._proj_pool.submit(detect_perspective, args)
            for name, args in [
                ("front", (front_t, "front", hole_ok)),
                ("rear",  (rear_t,  "rear",  hole_ok)),
                ("left",  (left_t,  "left",  hole_ok)),
                ("right", (right_t, "right", hole_ok)),
            ]
        }

        results = {}
        for name, fut in futures.items():
            try:
                results[name] = fut.result(timeout=self._PROJ_TIMEOUT)
            except FuturesTimeout:
                self.get_logger().warn(f"{name} view timed out — using blank")
                H_pv, W_pv = self.proj_height, self.proj_width
                results[name] = (np.zeros((H_pv, W_pv, 3), np.uint8), [])
            except Exception as e:
                self.get_logger().warn(f"{name} view error: {e}")
                H_pv, W_pv = self.proj_height, self.proj_width
                results[name] = (np.zeros((H_pv, W_pv, 3), np.uint8), [])

        front_ann, front_det = results["front"]
        rear_ann,  rear_det  = results["rear"]
        left_ann,  left_det  = results["left"]
        right_ann, right_det = results["right"]

        side_by_view = {"rear":rear_det,"left":left_det,"right":right_det}
        conf_score, evidence = cross_check_detections(bev_dets, side_by_view)

        # Publish
        msg_hole = Float32()
        msg_hole.data = float(hole_area_px)
        self.lidar_hole_pub.publish(msg_hole)

        all_dets = bev_dets + front_det + rear_det + left_det + right_det
        if all_dets:
            parts = [f"{d['type']},{d['distance']:.2f},{d['x']:.2f},{d['y']:.2f},{d['view']}"
                     for d in all_dets]
            out = String(); out.data = ";".join(parts)
            self.landmark_pub.publish(out)

        sc = Float32(); sc.data = float(conf_score)
        self.confidence_pub.publish(sc)

        wc = Float32()
        wc.data = float(sum(1 for d in bev_dets
                            if d["type"] == "wheel_corner" and d.get("confirmed")))
        self.wheel_count_pub.publish(wc)

        # Throttled per-corner diagnostic: shows the peak rubber/plastic
        # score at each rectangle corner and whether it confirmed, so the
        # wheel threshold can be tuned from real numbers instead of guessed.
        corners = [d for d in bev_dets if d["type"] == "wheel_corner"]
        if corners:
            now = time.time()
            if now - getattr(self, "_wheel_log_t", 0.0) > 2.0:
                self._wheel_log_t = now
                dbg = " ".join(
                    f"{d.get('corner','?')}:"
                    f"{'Y' if d.get('confirmed') else 'n'}"
                    f"(r{d.get('rubber_score',0):.0f}"
                    f"/p{d.get('plastic_score',0):.0f})"
                    for d in corners)
                self.get_logger().info(f"[WHEELS] {int(wc.data)}/4  {dbg}")

        side_conf = sum(1 for v in side_by_view.values()
                        for d in v if d["type"] == "under_car")
        if side_conf > 0:
            self.get_logger().debug(f"[SIDE] {side_conf}/4 views confirm under-car")

        car_dets = [d for d in bev_dets if d["type"]=="car_landmark"]
        # Don't publish the green box until it's grown to a trustworthy size
        # -- see WB_PUBLISH_MIN_M. Applies to both publishers below: a
        # too-small box is equally misleading whether or not conf_score has
        # crossed its own gate.
        car_dets = [d for d in car_dets
                    if d.get('wheelbase', 0.) >= WB_PUBLISH_MIN_M]
        if car_dets:
            bd = car_dets[0]
            bbox_msg = String()
            bbox_msg.data = (f"{bd['x']:.4f},{bd['y']:.4f},"
                              f"{bd.get('wheelbase',0.):.4f},{bd.get('track',0.):.4f},"
                              f"{bd.get('angle_rad',0.):.5f}")
            self.car_bbox_pub.publish(bbox_msg)

        if car_dets and conf_score >= 0.40:
            c = car_dets[0]
            cm = String()
            cm.data = (f"{c['x']:.4f},{c['y']:.4f},"
                       f"{c.get('wheelbase',0.):.4f},{c.get('track',0.):.4f},"
                       f"{c.get('n_wheels',0)},{conf_score:.3f}")
            self.car_landmark_pub.publish(cm)
            self.get_logger().info(
                f"[CAR] x={c['x']:.2f} y={c['y']:.2f} "
                f"wb={c.get('wheelbase',0):.2f} conf={conf_score:.2f} {evidence}")

        # Compose surround view image
        cc  = (0, int(conf_score*255), int((1-conf_score)*180))
        W_v = self.proj_width
        H_v = self.proj_height

        def add_label(img, title, sub="", title_col=(200,200,200)):
            out = img.copy()
            h, w = out.shape[:2]
            cv2.rectangle(out, (0,0), (w,22), (15,15,15), -1)
            cv2.putText(out, title, (5,15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, title_col, 1, cv2.LINE_AA)
            cv2.putText(out, f"{w}x{h}px", (w-72, h-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (90,90,90), 1, cv2.LINE_AA)
            if sub:
                cv2.putText(out, sub, (5, h-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, cc, 1, cv2.LINE_AA)
            return out

        if hole_ok:
            front_img, _, _ = front_t
        else:
            front_img = np.zeros((H_v, W_v, 3), np.uint8)
            cv2.putText(front_img, "no hole detected",
                        (8, H_v//2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (50,50,50), 1, cv2.LINE_AA)

        front_lbl = add_label(front_img,  "FRONT", sub=f"fov: {W_v}px wide",
                              title_col=(0,255,0))
        right_lbl = add_label(right_ann,  "RIGHT",  title_col=(0,255,255))
        left_lbl  = add_label(left_ann,   "LEFT",   title_col=(255,220,0))
        rear_lbl  = add_label(rear_ann,   "REAR",   title_col=(80,80,255))

        grid_h  = H_v * 2
        top_row = np.hstack([front_lbl, right_lbl])
        bot_row = np.hstack([left_lbl,  rear_lbl])
        grid_4  = np.vstack([top_row, bot_row])
        cv2.line(grid_4,(W_v,0),(W_v,grid_h),(80,80,80),1)
        cv2.line(grid_4,(0,H_v),(W_v*2,H_v),(80,80,80),1)

        m_per_px = 1.0 / (self.bev_scale / 100.0)
        bev_panel = cv2.resize(bev_d_ann, (grid_h, grid_h))
        bev_panel = add_label(bev_panel,
                              f"BEV  conf={conf_score:.2f}",
                              sub=f"1px={m_per_px:.2f}m | " + " | ".join(evidence),
                              title_col=(200,200,200))

        blind_panel = cv2.resize(blind_bev, (grid_h, grid_h))
        blind_panel = add_label(blind_panel,
                                "BLIND SPOT (live)",
                                sub=("hole" if blind_hole[0] else "no hole"),
                                title_col=(0,0,255))

        surround = np.hstack([grid_4, bev_panel, blind_panel])

        # FIX 7: Put frame into bounded queue (non-blocking).
        # Drop the oldest frame if the display hasn't consumed it yet —
        # this keeps the compute thread moving and prevents memory pileup.
        try:
            self._display_queue.put_nowait(surround)
        except queue.Full:
            try:
                self._display_queue.get_nowait()   # evict stale frame
            except queue.Empty:
                pass
            try:
                self._display_queue.put_nowait(surround)
            except queue.Full:
                pass

        self.get_logger().debug(
            f"render {(time.time()-t0)*1000:.1f}ms  "
            f"buf={len(self.buffer)} pts={len(cloud)}")

    # ─────────────────────────────────────────────────────────────
    def destroy_node(self):
        self._shutdown.set()
        self._proj_pool.shutdown(wait=False)
        super().destroy_node()


# ================================================================
# MAIN
# ================================================================
def main(args=None):
    rclpy.init(args=args)
    node = IntensityLandmarkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()