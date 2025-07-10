import sys
import os
import pygame
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from segment_anything import sam_model_registry, SamPredictor
import datetime
import logging
import cv2
import pygame.gfxdraw
import math
import csv
from scipy.ndimage import rotate as ndimage_rotate

# Declaration of working slop:
# This project was vibe-coded with Github Copilot. 
# 
# If you want to try to maintain this code, don't. 
# Save your sanity and rewrite instead.


# Configuration
IMAGE_DIR = "images"  # directory with equirectangular images
SAM_MODEL_TYPE = "vit_h"  # model type: vit_h, vit_l, vit_b
SAM_WEIGHTS_PATH = "sam_vit_h_4b8939.pth"  # SAM weights file
RESULTS_DIR = "results"
CACHE_DIR = "cache"
SVF_DIR = "svf"  # Add this line for SVF output directory

# UI Layout constants (single source of truth)
LEFT_PANEL_WIDTH = 420  # wider UI
STATUS_BAR_HEIGHT = 32
MODERN_FONT_NAME = pygame.font.match_font('segoeui,arial,liberationsans,sansserif') or None
MODERN_FONT_ANTIALIAS = True
PANEL_BG = (38, 38, 42)
PANEL_BORDER = (60, 60, 70)
PANEL_HEADING = (220, 220, 230)
PANEL_TEXT = (210, 210, 210)
PANEL_SUBTLE = (140, 140, 150)
PANEL_DIVIDER = (80, 80, 90)
STATUS_BG = (28, 28, 30)
STATUS_TEXT = (160, 160, 170)

# Logging initialization
def init_logging(name, console_level=logging.INFO, file_level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    # Remove existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    fh = logging.FileHandler(os.path.join(RESULTS_DIR, f"{name}.log"))
    fh.setLevel(file_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

# Initialize Pygame
def main():
    logger = init_logging('svf', console_level=logging.WARN, file_level=logging.DEBUG)
    logger.info('Starting SAM Segmenter')
    pygame.init()
    pygame.font.init()
    temp_font = pygame.font.SysFont(None, 48)

    # Prepare cache dir
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    # Prepare SVF dir
    if not os.path.exists(SVF_DIR):
        os.makedirs(SVF_DIR)

    # Get first image in directory, prefer test.jpg if present
    files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]
    if not files:
        logger.error(f"No images found in {IMAGE_DIR}")
        print(f"No images found in {IMAGE_DIR}")
        return
    # Prefer 'test.jpg' (case-insensitive), else first file
    preferred = next((f for f in files if f.lower() == "test.jpg"), None)
    image_file = preferred if preferred else files[0]
    image_idx = files.index(image_file)
    image_path = os.path.join(IMAGE_DIR, image_file)
    img_pil = Image.open(image_path).convert("RGB")
    img = np.array(img_pil)
    h_img, w_img, _ = img.shape
    h = int(h_img)
    w = int(w_img)
    logger.debug(f"Set h={h}, w={w} (image dimensions)")

    # Now update window size based on image and create main window
    window_w = int(LEFT_PANEL_WIDTH + min(int(w * 1.0), 1600))
    window_h = int(min(int(h * 1.0), 900) + STATUS_BAR_HEIGHT)
    img_area_w = int(window_w - LEFT_PANEL_WIDTH)
    img_area_h = int(window_h - STATUS_BAR_HEIGHT)
    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption(f"SAM Segmenter - {image_file}")
    # Draw loading message in main window (now correct size)
    screen.fill((30, 30, 30))
    loading_text = temp_font.render("Loading model...", True, (255, 255, 255))
    screen.blit(loading_text, (LEFT_PANEL_WIDTH + img_area_w // 2 - 180, img_area_h // 2 - 24))
    pygame.display.flip()
    for _ in range(10):
        pygame.event.pump()
        pygame.time.wait(10)

    # --- Model loading in background thread ---
    import time
    from threading import Thread
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_loading_status = {'status': 'Loading model...', 'done': False, 'success': False, 'error': None, 'start_time': time.time(), 'total_time': None}
    model_loaded = [None]
    def load_model_bg():
        try:
            model_loading_status['status'] = f"Loading model to {'GPU' if device == 'cuda' else 'CPU'}..."
            sam_local = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_WEIGHTS_PATH)
            sam_local.to(device)
            predictor_local = SamPredictor(sam_local)
            model_loaded[0] = (sam_local, predictor_local)
            model_loading_status['done'] = True
            model_loading_status['success'] = True
            model_loading_status['total_time'] = time.time() - model_loading_status['start_time']
            model_loading_status['status'] = f"Model loaded to {device} in {model_loading_status['total_time']:.1f}s"
        except Exception as e:
            model_loading_status['done'] = True
            model_loading_status['success'] = False
            model_loading_status['error'] = str(e)
            model_loading_status['status'] = f"Model load failed: {e}"
    t = Thread(target=load_model_bg)
    t.start()

    # --- Window and area sizes (remove nav bar height) ---
    window_w = int(LEFT_PANEL_WIDTH + min(int(w * 1.0), 1600))
    window_h = int(min(int(h * 1.0), 900) + STATUS_BAR_HEIGHT)
    img_area_w = int(window_w - LEFT_PANEL_WIDTH)
    img_area_h = int(window_h - STATUS_BAR_HEIGHT)
    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption(f"SAM Segmenter - {image_file}")

    # Cached rects for partial redraws (remove nav bar)
    left_panel_rect = pygame.Rect(0, 0, LEFT_PANEL_WIDTH, window_h)
    image_area_rect = pygame.Rect(LEFT_PANEL_WIDTH, 0, img_area_w, img_area_h)
    status_bar_rect = pygame.Rect(0, window_h - STATUS_BAR_HEIGHT, window_w, STATUS_BAR_HEIGHT)

    # Set up window for half-size image, but now with left panel and status bar
    # Window size: left panel + image area, status bar at bottom
    window_w = int(LEFT_PANEL_WIDTH + min(int(w * 1.0), 1600))  # use 1.0 for max possible width
    window_h = int(min(int(h * 1.0), 900) + STATUS_BAR_HEIGHT)
    img_area_w = int(window_w - LEFT_PANEL_WIDTH)
    img_area_h = int(window_h - STATUS_BAR_HEIGHT)
    logger.debug(f"UI window_w={window_w}, window_h={window_h}, img_area_w={img_area_w}, img_area_h={img_area_h}")

    # --- Zoom limits (must be after img_area_w/h, w, h are known) ---
    MIN_SCALE = float(max(img_area_w / w, img_area_h / h, 0.2))  # Never let image be smaller than area
    MAX_SCALE = 1.5  # Lower max zoom for performance

    base_scale = MIN_SCALE  # Start at lowest possible zoom
    scale_factor = base_scale
    w_ui, h_ui = int(w * scale_factor), int(h * scale_factor)
    img_ui = np.array(Image.fromarray(img).resize((w_ui, h_ui), Image.LANCZOS))
    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption(f"SAM Segmenter - {files[0]}")

    # Viewport for scrolling and zooming
    offset_x, offset_y = 0, 0
    dragging = False
    zoom_center = (img_area_w // 2, (img_area_h) // 2)
    logger.info(f"Image size: {w_ui}x{h_ui}, Window size: {window_w}x{window_h}")

    # Prepare a cached surface for fast blitting during drag
    img_ui_surf = pygame.surfarray.make_surface(img_ui.swapaxes(0, 1))

    def update_ui_image():
        nonlocal img_ui, w_ui, h_ui, img_ui_surf
        w_ui, h_ui = int(w * scale_factor), int(h * scale_factor)
        img_ui = np.array(Image.fromarray(img).resize((w_ui, h_ui), Image.LANCZOS))
        img_ui_surf = pygame.surfarray.make_surface(img_ui.swapaxes(0, 1))
        logger.debug(f"update_ui_image: w_ui={w_ui}, h_ui={h_ui}, scale_factor={scale_factor}")

    def clamp_offsets():
        nonlocal offset_x, offset_y
        max_offset_x = max(0, w_ui - img_area_w)
        max_offset_y = max(0, h_ui - (img_area_h))
        offset_x = min(max(0, offset_x), max_offset_x)
        offset_y = min(max(0, offset_y), max_offset_y)

    # Show loading while model loads
    screen.fill((30, 30, 30))
    loading_text = temp_font.render("Loading model...", True, (255, 255, 255))
    screen.blit(loading_text, (LEFT_PANEL_WIDTH + img_area_w // 2 - 180, img_area_h // 2 - 24))
    pygame.display.flip()
    for _ in range(10):
        pygame.event.pump()
        pygame.time.wait(10)

    pygame.font.init()
    font = pygame.font.Font(MODERN_FONT_NAME, 28)
    panel_font = pygame.font.Font(MODERN_FONT_NAME, 22)
    panel_font_subtle = pygame.font.Font(MODERN_FONT_NAME, 18)
    panel_heading_font = pygame.font.Font(MODERN_FONT_NAME, 32)
    status_font = pygame.font.Font(MODERN_FONT_NAME, 20)
    # All non-status texts for the left panel
    left_panel_texts = [
        "Left click: add point (new area)",
        "Right click: remove point",
        "R: reset",
        "ESC x3: quit",
        "Pan: middle mouse or Ctrl+left drag",
        "Arrow keys: scroll",
        "+/- or Ctrl+wheel: zoom",
        "1,2,3,4: Select amount"
    ]
    status_text = "Add points to segment."

    # --- Click types for weighted mask ---
    CLICK_TYPES = [
        {"name": "Purple", "color": (128, 0, 128), "weight": 1.0},
        {"name": "Red",    "color": (220, 40, 40),  "weight": 0.75},
        {"name": "Orange", "color": (255, 140, 0),  "weight": 0.5},
        {"name": "Yellow", "color": (255, 220, 40), "weight": 0.25},
    ]
    DIRECTION_TOOL = {"name": "W", "color": (255, 80, 200), "weight": None, "is_direction": True}
    active_click_type = 0  # Index into CLICK_TYPES or 4 for direction tool
    w_point = None  # (x, y) for W point in image coordinates

    # --- Drawing direction points and labels ---
    def draw_direction_points_func(screen, w_point, w_label_color=(255,80,200)):
        if w_point is None:
            return
        px, py = w_point
        h_img, w_img = h, w
        def wrap_x(x):
            return x % w_img
        # Correct mapping: clicked=W, right=N, opposite=E, left=S (for equirectangular)
        points = [
            (px, py, 'W'),  # Clicked = West
            (wrap_x(px + w_img//4), py, 'N'),  # Right = North
            (wrap_x(px + w_img//2), py, 'E'),  # Opposite = East
            (wrap_x(px + 3*w_img//4), py, 'S'),  # Left = South
        ]
        for x, y, label in points:
            sx = int(x * scale_factor) - offset_x
            sy = int(y * scale_factor) - offset_y
            if 0 <= sx < img_area_w and 0 <= sy < img_area_h:
                pygame.draw.circle(screen, w_label_color, (LEFT_PANEL_WIDTH + sx, sy), 10)
                font = pygame.font.Font(MODERN_FONT_NAME, 18)
                label_surf = font.render(label, True, w_label_color)
                label_rect = label_surf.get_rect(center=(LEFT_PANEL_WIDTH + sx, sy - 18))
                screen.blit(label_surf, label_rect)

    def draw_direction_points_hemisphere(screen, w_point, out_size, crop_n, w_label_color=(255,80,200)):
        """
        Draw W, N, E, S as lines from the outer edge inward by ~20° zenith angle, with a circle and label at the inner end.
        The direction points are mirrored over the horizon (azimuth φ → φ+π) to project ground directions onto the sky.
        """
        if w_point is None:
            return
        px, py = w_point
        h_img, w_img = h - 2*crop_n, w
        def wrap_x(x):
            return x % w_img
        # Now: W, N, E, S (clicked, right, opposite, left)
        points = [
            (px, py, 'W'),
            (wrap_x(px + w_img//4), py, 'N'),
            (wrap_x(px + w_img//2), py, 'E'),
            (wrap_x(px + 3*w_img//4), py, 'S'),
        ]
        out_r = out_size // 2
        theta_inward = math.radians(20)
        theta_outer = math.pi / 2  # 90° zenith (edge)
        theta_inner = theta_outer - theta_inward  # ~70° zenith
        for x, y, label in points:
            py_cropped = y - crop_n
            if 0 <= py_cropped < h_img:
                phi_eq = 2 * math.pi * (x / w_img)
                phi_hs = (phi_eq + math.pi) % (2 * math.pi)
                r_outer = out_r * (theta_outer / (math.pi / 2))
                u_outer = out_r + r_outer * math.sin(phi_hs)
                v_outer = out_r + r_outer * math.cos(phi_hs)
                r_inner = out_r * (theta_inner / (math.pi / 2))
                u_inner = out_r + r_inner * math.sin(phi_hs)
                v_inner = out_r + r_inner * math.cos(phi_hs)
                pygame.draw.line(screen, w_label_color, (LEFT_PANEL_WIDTH + int(u_outer), int(v_outer)), (LEFT_PANEL_WIDTH + int(u_inner), int(v_inner)), 4)
                pygame.draw.circle(screen, w_label_color, (LEFT_PANEL_WIDTH + int(u_inner), int(v_inner)), 10)
                font = pygame.font.Font(MODERN_FONT_NAME, 18)
                label_surf = font.render(label, True, w_label_color)
                label_rect = label_surf.get_rect(center=(LEFT_PANEL_WIDTH + int(u_inner), int(v_inner) - 18))
                screen.blit(label_surf, label_rect)

    # --- Refactored: Each point is its own mask prompt ---
    class PointManager:
        def __init__(self):
            self.points = []  # List of (x, y, type_idx)
        def add_point(self, x, y, type_idx):
            self.points.append((x, y, type_idx))
        def remove_nearest(self, sx, sy, scale_factor, offset_x, offset_y):
            if not self.points:
                return
            min_dist = float('inf')
            min_idx = None
            for idx, (px, py, type_idx) in enumerate(self.points):
                ux = int(px * scale_factor) - offset_x
                uy = int(py * scale_factor) - offset_y
                dist = (ux - sx) ** 2 + (uy - sy) ** 2
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
            if min_dist < (12 ** 2) and min_idx is not None:
                del self.points[min_idx]
        def clear(self):
            self.points = []
        def get_all(self):
            return self.points
        def __len__(self):
            return len(self.points)

    point_manager = PointManager()
    masks = []
    esc_count = 0
    mask = None
    masked_surf = None

    clock = pygame.time.Clock()
    running = True
    processing = False
    process_message = "Processing..."

    # --- Letterbox crop value ---
    # Automatically calculate letterbox crop for equirectangular images
    # For a perfect equirectangular: width = 2 * height, so crop = 0
    # If image is taller: crop = (h_img - w_img // 2) // 2
    letterbox_crop = max(0, (h_img - w_img // 2) // 2)
    # Mouse hover coordinates (image space)
    mouse_img_x = None
    mouse_img_y = None

    def compute_svf(mask, crop, h):
        """
        Compute Sky View Factor (SVF) from a weighted mask and crop value using solid‐angle integration
        on an equirectangular image.

        This function implements two equivalent, physically correct methods for SVF calculation over a hemisphere:

        1. Continuous solid‐angle integration (default):
        - For an equirectangular panorama of height h:
            each pixel row y corresponds to a zenith angle θ = π·y/h.
        - For row y (from crop up to h//2):
                θ0 = π * y     / h      (top of row)
                θ1 = π * (y+1) / h      (bottom of row)
            solid angle of that ring:
                dΩ_row = 2π * [ cos(θ0) - cos(θ1) ]
        - Accumulate
                numerator   = ∑ (dΩ_row * sky_fraction_row)
                denominator = ∑ dΩ_row
            where sky_fraction_row = mean(mask[y, :]) ∈ [0,1].
        - SVF = numerator / denominator

        2. Discrete “ring” approximation using the paper’s closed‐form weights:
        - Divide the hemisphere into n = (h//2 - crop) equal‐angle θ‐rings.
        - For i = 1…n, each ring’s normalized weight is
                w_i = (1 / (2π))
                    · sin(π / (2n))
                    · sin[π (2i - 1) / (2n)]
            which satisfies ∑ w_i = 1.
        - SVF = ∑_{i=1}^n w_i · mean(mask[crop + i - 1, :])

        Args:
            mask : 2D numpy array of floats in [0,1].  
                0 = fully blocked, 1 = fully sky.
            crop : int, pixels removed from top (zenith) of image.
            h    : int, total image height in pixels.

        Returns:
            float in [0,1]: 1.0 = open sky, 0.0 = fully blocked.

        Notes:
        - Both methods assume an equirectangular vertical mapping (linear θ vs y).
        - Method (1) integrates “exactly” over the pixel rows.
        - Method (2) matches the paper’s closed‐form spherical‐area discretization.
        """
        y0 = int(crop)
        y1 = h // 2
        if y1 <= y0:
            return 0.0

        svf_num = 0.0
        svf_den = 0.0
        for y in range(y0, y1):
            theta0 = math.pi * y     / h
            theta1 = math.pi * (y + 1) / h
            d_omega = 2 * math.pi * (math.cos(theta0) - math.cos(theta1))
            sky_frac = np.mean(mask[y, :])
            svf_num += d_omega * sky_frac
            svf_den += d_omega

        return svf_num / svf_den if svf_den > 0 else 0.0

    svf_value = 0  # Store last SVF value, initialized to 0
    show_math_viz = False  # Toggle for SVF math visualization
    show_horizon = False  # Toggle for always showing the horizon
    show_spherical_view = False  # Toggle for spherical (fisheye) projection
    show_points = True  # Toggle for showing points
    show_masks = True   # Toggle for showing masks

    # --- Spherical/Fisheye projection helpers ---
    def equi_to_fisheye_coords(x, y, w, h, out_r):
        """
        Map equirectangular (x, y) to fisheye (u, v) coordinates.
        Only zenith angles 0..pi/2 (northern hemisphere) are mapped.
        Returns (u, v) in fisheye image (centered at (out_r, out_r), radius out_r).
        Azimuth is inverted so that right in EQ maps to right in HS (photographic convention).
        """
        theta = math.pi * y / h  # zenith angle [0, pi]
        phi = 2 * math.pi * (1 - x / w)  # azimuth [0, 2pi], INVERTED
        if theta > math.pi / 2:
            return None
        r = out_r * theta / (math.pi / 2)
        u = out_r + r * math.sin(phi)
        v = out_r - r * math.cos(phi)
        return u, v

    def fisheye_to_equi_coords(u, v, w, h, out_r):
        """
        Map fisheye (u, v) to equirectangular (x, y).
        Returns (x, y) in equirectangular image, or None if outside hemisphere.
        Azimuth is inverted so that right in HS maps to right in EQ (photographic convention).
        """
        dx = u - out_r
        dy = out_r - v
        r = math.sqrt(dx*dx + dy*dy)
        if r > out_r:
            return None
        theta = (r / out_r) * (math.pi / 2)
        phi = math.atan2(dx, dy)
        if phi < 0:
            phi += 2 * math.pi
        phi = (2 * math.pi - phi) % (2 * math.pi)  # Invert azimuth
        x = int((phi / (2 * math.pi)) * w)
        y = int((theta / math.pi) * h)
        return x, y

    def project_equi_to_fisheye(img, out_size):
        """
        Project equirectangular image (h, w, 3) to fisheye (out_size, out_size, 3).
        """
        h, w = img.shape[:2]
        out_r = out_size // 2
        out_img = np.zeros((out_size, out_size, img.shape[2]), dtype=img.dtype)
        for v in range(out_size):
            for u in range(out_size):
                res = fisheye_to_equi_coords(u, v, w, h, out_r)
                if res is not None:
                    x, y = res
                    out_img[v, u] = img[y % h, x % w]
        return out_img

    def project_mask_equi_to_fisheye(mask, out_size):
        h, w = mask.shape
        out_r = out_size // 2
        out_mask = np.zeros((out_size, out_size), dtype=mask.dtype)
        for v in range(out_size):
            for u in range(out_size):
                res = fisheye_to_equi_coords(u, v, w, h, out_r)
                if res is not None:
                    x, y = res
                    out_mask[v, u] = mask[y % h, x % w]
        return out_mask

    def project_point_equi_to_fisheye(px, py, w, h, out_r):
        return equi_to_fisheye_coords(px, py, w, h, out_r)

    def project_point_fisheye_to_equi(u, v, w, h, out_r):
        return fisheye_to_equi_coords(u, v, w, h, out_r)

    # --- UI drawing functions ---
    def draw_left_panel():
        # Draw left panel background (flat grey)
        pygame.draw.rect(screen, PANEL_BG, (0, 0, LEFT_PANEL_WIDTH, window_h))
        # Panel border (right edge)
        pygame.draw.line(screen, PANEL_BORDER, (LEFT_PANEL_WIDTH - 1, 0), (LEFT_PANEL_WIDTH - 1, window_h), 2)
        # Heading
        heading_margin = 32
        heading_surf = panel_heading_font.render("SAM SVF Calculation", MODERN_FONT_ANTIALIAS, PANEL_HEADING)
        screen.blit(heading_surf, (heading_margin, 28))
        # Separator below heading
        sep_y = 28 + panel_heading_font.get_height() + 15
        pygame.draw.line(screen, PANEL_DIVIDER, (heading_margin, sep_y), (LEFT_PANEL_WIDTH - heading_margin, sep_y), 2)
        # --- Click type selector (4 in a row, left of SVF text) ---
        dot_radius = 18  # Slightly larger than original, but not huge
        selector_y = sep_y + 32
        n_dots = len(CLICK_TYPES)
        total_dots = n_dots + 1  # 4 percentages + 1 W
        dots_area_width = 220  # Enough for 5 dots with spacing
        selector_x_start = heading_margin + 28  # Respect left padding, but not too much
        spacing_x = dots_area_width // (total_dots - 1) if total_dots > 1 else 0
        dot_centers = []
        for i, t in enumerate(CLICK_TYPES):
            cx = selector_x_start + i * spacing_x
            cy = selector_y
            dot_centers.append((cx, cy))
            # Draw background highlight if selected
            if i == active_click_type:
                pygame.draw.circle(screen, (255, 255, 255), (cx, cy), dot_radius + 4)
            pygame.draw.circle(screen, t["color"], (cx, cy), dot_radius)
            # Draw border
            pygame.draw.circle(screen, (40, 40, 40), (cx, cy), dot_radius, 3)
        # Draw the pink 'W' direction tool, spaced as 5th dot
        w_cx = selector_x_start + n_dots * spacing_x
        w_cy = selector_y
        if active_click_type == len(CLICK_TYPES):
            pygame.draw.circle(screen, (255, 255, 255), (w_cx, w_cy), dot_radius + 4)
        pygame.draw.circle(screen, DIRECTION_TOOL["color"], (w_cx, w_cy), dot_radius)
        pygame.draw.circle(screen, (40, 40, 40), (w_cx, w_cy), dot_radius, 3)
        # Draw 'W' label below the pink dot
        percent_font = pygame.font.Font(MODERN_FONT_NAME, 16)
        w_label = percent_font.render("W", MODERN_FONT_ANTIALIAS, (255,255,255))
        w_label_rect = w_label.get_rect(center=(w_cx, w_cy + dot_radius + 12))
        screen.blit(w_label, w_label_rect)
        # Draw percentages below each dot
        percent_font = pygame.font.Font(MODERN_FONT_NAME, 16)
        for i, t in enumerate(CLICK_TYPES):
            cx, cy = dot_centers[i]
            percent_label = f"{int(t['weight']*100)}%"
            percent_surf = percent_font.render(percent_label, MODERN_FONT_ANTIALIAS, (255,255,255))
            percent_rect = percent_surf.get_rect(center=(cx, cy + dot_radius + 12))
            screen.blit(percent_surf, percent_rect)
        # Move second separator here, above SVF heading/value
        dots_bottom_y = selector_y + dot_radius + 24  # Add a bit of space below dots/labels
        pygame.draw.line(screen, PANEL_DIVIDER, (heading_margin, dots_bottom_y), (LEFT_PANEL_WIDTH - heading_margin, dots_bottom_y), 2)
        # --- SVF (%) label and value below the new separator ---
        svf_label_font = pygame.font.Font(MODERN_FONT_NAME, 24)
        svf_label = svf_label_font.render("SVF (%)", True, (220, 220, 230))
        svf_label_x = heading_margin + 10  # left padding
        svf_label_y = dots_bottom_y + 18  # below new separator
        screen.blit(svf_label, (svf_label_x, svf_label_y))
        # --- SVF value below label, blue, bold, much bigger ---
        if svf_value is not None:
            svf_val_font = pygame.font.Font(MODERN_FONT_NAME, 48)
            svf_val_font.set_bold(True)
            svf_val_str = f"{svf_value*100:.3f}"
            svf_val_surf = svf_val_font.render(svf_val_str, True, (80, 200, 255))
            svf_val_x = svf_label_x
            svf_val_y = svf_label_y + svf_label.get_height() + 4
            screen.blit(svf_val_surf, (svf_val_x, svf_val_y))
            row_bottom = svf_val_y + svf_val_surf.get_height() + 16
        else:
            row_bottom = svf_label_y + svf_label.get_height() + 16
        # Divider below SVF value
        pygame.draw.line(screen, PANEL_DIVIDER, (heading_margin, row_bottom), (LEFT_PANEL_WIDTH - heading_margin, row_bottom), 2)
        # --- Show Points toggle ---
        toggle_font = pygame.font.Font(MODERN_FONT_NAME, 20)
        show_points_label = "Show Points"
        show_points_x = heading_margin
        show_points_y = row_bottom + 24
        show_points_w = LEFT_PANEL_WIDTH - 2 * heading_margin
        show_points_h = 32
        show_points_rect = pygame.Rect(show_points_x, show_points_y, show_points_w, show_points_h)
        pygame.draw.rect(screen, (60, 60, 80), show_points_rect, border_radius=8)
        if show_points:
            pygame.draw.rect(screen, (80, 200, 255), show_points_rect, 0, border_radius=8)
        show_points_surf = toggle_font.render(show_points_label, True, (255,255,255))
        screen.blit(show_points_surf, (show_points_rect.x + 12, show_points_rect.y + 5))
        show_points_ind_color = (80, 200, 255) if show_points else (120, 120, 120)
        pygame.draw.circle(screen, show_points_ind_color, (show_points_rect.right - 18, show_points_rect.centery), 10)
        # --- Show Masks toggle ---
        show_masks_label = "Show Masks"
        show_masks_x = heading_margin
        show_masks_y = show_points_y + show_points_h + 12
        show_masks_w = LEFT_PANEL_WIDTH - 2 * heading_margin
        show_masks_h = 32
        show_masks_rect = pygame.Rect(show_masks_x, show_masks_y, show_masks_w, show_masks_h)
        pygame.draw.rect(screen, (60, 60, 80), show_masks_rect, border_radius=8)
        if show_masks:
            pygame.draw.rect(screen, (80, 200, 255), show_masks_rect, 0, border_radius=8)
        show_masks_surf = toggle_font.render(show_masks_label, True, (255,255,255))
        screen.blit(show_masks_surf, (show_masks_rect.x + 12, show_masks_rect.y + 5))
        show_masks_ind_color = (80, 200, 255) if show_masks else (120, 120, 120)
        pygame.draw.circle(screen, show_masks_ind_color, (show_masks_rect.right - 18, show_masks_rect.centery), 10)
        # --- Show Horizon toggle below show_masks toggle ---
        horizon_toggle_label = "Show Horizon"
        horizon_toggle_x = heading_margin
        horizon_toggle_y = show_masks_y + show_masks_h + 12
        horizon_toggle_w = LEFT_PANEL_WIDTH - 2 * heading_margin
        horizon_toggle_h = 32
        horizon_toggle_rect = pygame.Rect(horizon_toggle_x, horizon_toggle_y, horizon_toggle_w, horizon_toggle_h)
        pygame.draw.rect(screen, (60, 60, 80), horizon_toggle_rect, border_radius=8)
        if show_horizon:
            pygame.draw.rect(screen, (80, 200, 255), horizon_toggle_rect, 0, border_radius=8)
        horizon_label_surf = toggle_font.render(horizon_toggle_label, True, (255,255,255))
        screen.blit(horizon_label_surf, (horizon_toggle_rect.x + 12, horizon_toggle_rect.y + 5))
        horizon_ind_color = (80, 200, 255) if show_horizon else (120, 120, 120)
        pygame.draw.circle(screen, horizon_ind_color, (horizon_toggle_rect.right - 18, horizon_toggle_rect.centery), 10)
        # --- Spherical view toggle below horizon toggle ---
        spherical_toggle_label = "Show Spherical View (Slow)"
        spherical_toggle_x = heading_margin
        spherical_toggle_y = horizon_toggle_y + horizon_toggle_h + 12
        spherical_toggle_w = LEFT_PANEL_WIDTH - 2 * heading_margin
        spherical_toggle_h = 32
        spherical_toggle_rect = pygame.Rect(spherical_toggle_x, spherical_toggle_y, spherical_toggle_w, spherical_toggle_h)
        pygame.draw.rect(screen, (60, 60, 80), spherical_toggle_rect, border_radius=8)
        if show_spherical_view:
            pygame.draw.rect(screen, (80, 200, 255), spherical_toggle_rect, 0, border_radius=8)
        spherical_label_surf = toggle_font.render(spherical_toggle_label, True, (255,255,255))
        screen.blit(spherical_label_surf, (spherical_toggle_rect.x + 12, spherical_toggle_rect.y + 5))
        spherical_ind_color = (80, 200, 255) if show_spherical_view else (120, 120, 120)
        pygame.draw.circle(screen, spherical_ind_color, (spherical_toggle_rect.right - 18, spherical_toggle_rect.centery), 10)
        # --- Letterbox crop info (automatic) ---
        info_font = pygame.font.Font(MODERN_FONT_NAME, 20)
        info_str = f"Letterbox crop (auto): {letterbox_crop}px top/bottom"
        info_y = spherical_toggle_y + spherical_toggle_h + 18
        info_surf = info_font.render(info_str, MODERN_FONT_ANTIALIAS, PANEL_SUBTLE)
        screen.blit(info_surf, (heading_margin, info_y))
        # --- Save/Prev/Next buttons ---
        nav_font = pygame.font.Font(MODERN_FONT_NAME, 20)
        nav_btn_w = LEFT_PANEL_WIDTH - 2 * heading_margin
        nav_btn_h = 32
        nav_btn_y = info_y + 36
        nav_btn_gap = 12

        # Save SVF button
        save_btn_rect = pygame.Rect(heading_margin, nav_btn_y, nav_btn_w, nav_btn_h)
        pygame.draw.rect(screen, (60, 60, 80), save_btn_rect, border_radius=8)
        save_label = nav_font.render("Save image SVF [Enter]", True, (255,255,255))
        screen.blit(save_label, (save_btn_rect.x + 12, save_btn_rect.y + 5))

        # Prev/Next buttons
        prev_btn_rect = pygame.Rect(heading_margin, nav_btn_y + nav_btn_h + nav_btn_gap, (nav_btn_w-8)//2, nav_btn_h)
        next_btn_rect = pygame.Rect(prev_btn_rect.right + 8, prev_btn_rect.y, (nav_btn_w-8)//2, nav_btn_h)
        pygame.draw.rect(screen, (60, 60, 80), prev_btn_rect, border_radius=8)
        pygame.draw.rect(screen, (60, 60, 80), next_btn_rect, border_radius=8)
        prev_label = nav_font.render("Prev [B]", True, (255,255,255))
        next_label = nav_font.render("Next [N]", True, (255,255,255))
        screen.blit(prev_label, (prev_btn_rect.x + 12, prev_btn_rect.y + 5))
        screen.blit(next_label, (next_btn_rect.x + 12, next_btn_rect.y + 5))

        # Move instructions to bottom, subtle
        y = window_h - 28 * len(left_panel_texts) - 24
        for text in left_panel_texts:
            text_surf = panel_font_subtle.render(text, MODERN_FONT_ANTIALIAS, PANEL_SUBTLE)
            screen.blit(text_surf, (heading_margin, y))
            y += 22
        return show_points_rect, show_masks_rect, horizon_toggle_rect, spherical_toggle_rect, save_btn_rect, prev_btn_rect, next_btn_rect, (w_cx, w_cy, dot_radius)

    def draw_status_bar():
        # Draw status bar at the bottom (flat grey, less height, text left)
        pygame.draw.rect(screen, STATUS_BG, (0, window_h - STATUS_BAR_HEIGHT, window_w, STATUS_BAR_HEIGHT))
        # Show model loading status if not done
        if not model_loading_status['done']:
            status_surf = status_font.render(model_loading_status['status'], MODERN_FONT_ANTIALIAS, (255, 220, 80))
        elif model_loading_status['done'] and not model_loading_status['success']:
            status_surf = status_font.render(model_loading_status['status'], MODERN_FONT_ANTIALIAS, (255, 80, 80))
        else:
            status_surf = status_font.render(status_text, MODERN_FONT_ANTIALIAS, STATUS_TEXT)
        # Align status text with left panel margin
        screen.blit(status_surf, (32, window_h - STATUS_BAR_HEIGHT + 6))
        # Draw mouse hover x/y in bottom right
        if mouse_img_x is not None and mouse_img_y is not None:
            mouse_font = pygame.font.Font(MODERN_FONT_NAME, 16)
            mouse_str = f"x={mouse_img_x}  y={mouse_img_y}"
            mouse_surf = mouse_font.render(mouse_str, True, (120, 120, 120))
            mouse_rect = mouse_surf.get_rect()
            mouse_rect.bottomright = (window_w - 18, window_h - 6)
            screen.blit(mouse_surf, mouse_rect)

    def draw_click_type_selector():
        # Floating section in top-left of image area
        selector_x = LEFT_PANEL_WIDTH + 16
        selector_y = 16
        dot_radius = 18
        spacing = 54
        for i, t in enumerate(CLICK_TYPES):
            cx = selector_x
            cy = selector_y + i * spacing
            # Draw background highlight if selected
            if i == active_click_type:
                pygame.draw.circle(screen, (255, 255, 255), (cx, cy), dot_radius + 6)
            pygame.draw.circle(screen, t["color"], (cx, cy), dot_radius)
            # Draw border
            pygame.draw.circle(screen, (40, 40, 40), (cx, cy), dot_radius, 3)
            # Draw label
            label = panel_font.render(f"{int(t['weight']*100)}%", MODERN_FONT_ANTIALIAS, (255,255,255))
            screen.blit(label, (cx + dot_radius + 12, cy - 16))

    def crop_letterbox(img_or_mask, crop_n):
        # Crop top and bottom crop_n pixels
        if crop_n > 0:
            if img_or_mask.ndim == 3:
                return img_or_mask[crop_n:-crop_n, :, :]
            else:
                return img_or_mask[crop_n:-crop_n, :]
        return img_or_mask

    def draw_image_area():
        if show_spherical_view:
            # Spherical (fisheye) projection
            out_size = min(img_area_w, img_area_h)
            crop_n = int(letterbox_crop)
            # Downscale factor for performance
            fisheye_render_scale = 0.5  # Render at 50% size, then scale up
            render_size = max(1, int(out_size * fisheye_render_scale))
            # Crop letterbox from image before projection
            img_cropped = crop_letterbox(img, crop_n)
            fisheye_img_small = project_equi_to_fisheye(img_cropped, render_size)
            # Overlay: fully black out equirectangular area before drawing spherical
            overlay = pygame.Surface((img_area_w, img_area_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 255))  # Fully opaque
            screen.blit(overlay, (LEFT_PANEL_WIDTH, 0))
            # Stretch up to fit viewport
            fisheye_img_big = np.array(Image.fromarray(fisheye_img_small).resize((out_size, out_size), Image.BILINEAR))
            surf = pygame.surfarray.make_surface(fisheye_img_big.swapaxes(0, 1))
            screen.blit(surf, (LEFT_PANEL_WIDTH, 0))
            # Mask overlay
            if mask is not None and show_masks:
                mask_cropped = crop_letterbox((mask * 255).astype(np.uint8), crop_n)
                fisheye_mask_small = project_mask_equi_to_fisheye(mask_cropped, render_size)
                fisheye_mask = np.array(Image.fromarray(fisheye_mask_small).resize((out_size, out_size), Image.NEAREST))
                color_mask_ui = np.zeros((out_size, out_size, 4), dtype=np.uint8)
                for i, t in enumerate(CLICK_TYPES):
                    lower = t["weight"] - 0.13
                    upper = t["weight"] + 0.13
                    mask_range = ((fisheye_mask / 255.0) >= lower) & ((fisheye_mask / 255.0) < upper)
                    color = t["color"]
                    color_mask_ui[..., 0][mask_range] = color[0]
                    color_mask_ui[..., 1][mask_range] = color[1]
                    color_mask_ui[..., 2][mask_range] = color[2]
                    color_mask_ui[..., 3][mask_range] = (fisheye_mask[mask_range] * 0.6).astype(np.uint8)
                color_surf = pygame.image.frombuffer(color_mask_ui.tobytes(), (out_size, out_size), 'RGBA').convert_alpha()
                screen.blit(color_surf, (LEFT_PANEL_WIDTH, 0))
            # Draw points (use full out_size for mapping, not render_size)
            if show_points:
                point_radius = 7
                for px, py, type_idx in point_manager.get_all():
                    crop_n = int(letterbox_crop)
                    py_cropped = py - crop_n
                    h_cropped = h - 2 * crop_n
                    if 0 <= py_cropped < h_cropped:
                        res = project_point_equi_to_fisheye(px, py_cropped, w, h_cropped, out_size // 2)
                        if res is not None:
                            u, v = res
                            cx = LEFT_PANEL_WIDTH + int(u)
                            cy = int(v)
                            color = CLICK_TYPES[type_idx]["color"]
                            pygame.draw.circle(screen, color, (cx, cy), point_radius)
            # Draw direction points overlay if w_point is set
            if w_point is not None:
                draw_direction_points_hemisphere(screen, w_point, out_size, crop_n)
        else:
            # Equirectangular view
            # Draw checkerboard background for transparency debug (in image area)
            checker_size = 20
            for y in range(0, img_area_h, checker_size):
                for x in range(0, img_area_w, checker_size):
                    color = (180, 180, 180) if (x // checker_size + y // checker_size) % 2 == 0 else (100, 100, 100)
                    pygame.draw.rect(screen, color, (LEFT_PANEL_WIDTH + x, y, checker_size, checker_size))
            img_rect = pygame.Rect(offset_x, offset_y, img_area_w, img_area_h)
            surf = pygame.surfarray.make_surface(img_ui.swapaxes(0, 1))
            screen.blit(surf, (LEFT_PANEL_WIDTH, 0), img_rect)
            # Overlay improved mask as purple with 0.6 opacity (downscale for UI, scrolled)
            if mask is not None and show_masks:
                mask_ui = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((w_ui, h_ui), Image.NEAREST))
                color_mask_ui = np.zeros((h_ui, w_ui, 4), dtype=np.uint8)
                # Use purple for 100%, red for 75%, orange for 50%, yellow for 25%, blend by value
                for i, t in enumerate(CLICK_TYPES):
                    # Create a mask for this weight range
                    lower = t["weight"] - 0.13
                    upper = t["weight"] + 0.13
                    mask_range = ((mask_ui / 255.0) >= lower) & ((mask_ui / 255.0) < upper)
                    color = t["color"]
                    color_mask_ui[..., 0][mask_range] = color[0]
                    color_mask_ui[..., 1][mask_range] = color[1]
                    color_mask_ui[..., 2][mask_range] = color[2]
                    color_mask_ui[..., 3][mask_range] = (mask_ui[mask_range] * 0.6).astype(np.uint8)
                # Clamp crop coordinates to valid range
                crop_x0 = max(0, offset_x)
                crop_x1 = min(w_ui, offset_x + img_area_w)
                crop_y0 = max(0, offset_y)
                crop_y1 = min(h_ui, offset_y + img_area_h)
                crop_w = max(0, crop_x1 - crop_x0)
                crop_h = max(0, crop_y1 - crop_y0)
                if crop_w > 0 and crop_h > 0:
                    color_crop = color_mask_ui[crop_y0:crop_y1, crop_x0:crop_x1]
                    pad_w = img_area_w - crop_w
                    pad_h = img_area_h - crop_h
                    if pad_w > 0 or pad_h > 0:
                        color_crop = np.pad(color_crop, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                    color_surf = pygame.image.frombuffer(color_crop.tobytes(), (img_area_w, img_area_h), 'RGBA').convert_alpha()
                    screen.blit(color_surf, (LEFT_PANEL_WIDTH, 0))
            # Draw points with color by type
            if show_points:
                point_radius = 7
                for px, py, type_idx in point_manager.get_all():
                    sx = int(px * scale_factor) - offset_x
                    sy = int(py * scale_factor) - offset_y
                    if 0 <= sx < img_area_w and 0 <= sy < img_area_h:
                        color = CLICK_TYPES[type_idx]["color"]
                        pygame.draw.circle(screen, color, (LEFT_PANEL_WIDTH + sx, sy), point_radius)
            # Draw direction points overlay if w_point is not None:
            if w_point is not None:
                draw_direction_points_func(screen, w_point)
        # --- SVF Math Visualization ---
        if show_math_viz or show_horizon:
            if show_spherical_view:
                out_size = min(img_area_w, img_area_h)
                cx = LEFT_PANEL_WIDTH + out_size // 2
                cy = out_size // 2
                radius = out_size // 2
                pygame.gfxdraw.aacircle(screen, cx, cy, radius, (0, 255, 0))
            else:
                horizon_y = int(h_ui // 2) - offset_y
                if 0 <= horizon_y < img_area_h:
                    pygame.draw.line(screen, (0, 255, 0), (LEFT_PANEL_WIDTH, horizon_y), (LEFT_PANEL_WIDTH + img_area_w, horizon_y), 3)

    esc_count = 0
    save_mask = False
    clock = pygame.time.Clock()
    running = True
    processing = False
    process_message = "Processing..."

    # --- Ensure all button rects are initialized before use ---
    save_btn_rect = pygame.Rect(0, 0, 0, 0)
    prev_btn_rect = pygame.Rect(0, 0, 0, 0)
    next_btn_rect = pygame.Rect(0, 0, 0, 0)
    show_points_rect = None
    show_masks_rect = None
    horizon_toggle_rect = None
    spherical_toggle_rect = None

    def redraw_all():
        # draw_nav_bar()  # REMOVE this line
        show_points_rect, show_masks_rect, horizon_toggle_rect, spherical_toggle_rect, save_btn_rect, prev_btn_rect, next_btn_rect = draw_left_panel()
        draw_image_area()
        draw_status_bar()
        pygame.display.flip()
        return show_points_rect, show_masks_rect, horizon_toggle_rect, spherical_toggle_rect, save_btn_rect, prev_btn_rect, next_btn_rect

    def redraw_image_area():
        screen.fill((0, 0, 0), rect=image_area_rect)
        draw_image_area()
        draw_status_bar()
        pygame.display.update([image_area_rect, status_bar_rect])

    def redraw_left_panel():
        draw_left_panel()
        draw_status_bar()
        pygame.display.update([left_panel_rect, status_bar_rect])

    def redraw_status_bar():
        draw_status_bar()
        pygame.display.update(status_bar_rect)

    # --- Image navigation state ---
    image_idx = files.index(image_file)
    def reload_files():
        nonlocal files
        files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]
        files.sort()
    def load_image_by_idx(idx):
        nonlocal img_pil, img, h_img, w_img, h, w, img_ui, w_ui, h_ui, scale_factor, base_scale, image_file, image_path, svf_value, mask, masked_surf, mouse_img_x, mouse_img_y, image_idx, predictor
        reload_files()
        idx = idx % len(files)
        image_file = files[idx]
        image_idx = idx
        image_path = os.path.join(IMAGE_DIR, image_file)
        img_pil = Image.open(image_path).convert("RGB")
        img = np.array(img_pil)
        h_img, w_img, _ = img.shape
        h = int(h_img)
        w = int(w_img)
        base_scale = float(max(img_area_w / w, img_area_h / h, 0.2))
        scale_factor = base_scale
        w_ui, h_ui = int(w * scale_factor), int(h * scale_factor)
        img_ui = np.array(Image.fromarray(img).resize((w_ui, h_ui), Image.LANCZOS))
        svf_value = 0  # Initialize SVF value to 0 when loading a new image
        mask = None
        masked_surf = None
        mouse_img_x = None
        mouse_img_y = None
        pygame.display
        # Ensure predictor is updated with new image
        if model_loaded[0] is not None:
            sam, predictor_local = model_loaded[0]
            predictor = predictor_local
            predictor.set_image(img)

    def save_image_svf():
        if svf_value is None:
            logger.warning("SVF value not computed, not saving image.")
            return
        # Show "Saving..." message in gold/yellow
        nonlocal status_text
        status_text = "Saving..."
        draw_status_bar()
        pygame.display.update(status_bar_rect)
        pygame.event.pump()
        base, ext = os.path.splitext(image_file)

        # Always use PNG for export
        eq_name = f"{base}_SVF{svf_value*100:.3f}_EQ.png"
        hs_name = f"{base}_SVF{svf_value*100:.3f}_HS.png"
        eq_mask_name = f"{base}_SVF{svf_value*100:.3f}_EQ_mask.png"
        hs_mask_name = f"{base}_SVF{svf_value*100:.3f}_HS_mask.png"
        eq_path = os.path.join(SVF_DIR, eq_name)
        hs_path = os.path.join(SVF_DIR, hs_name)
        eq_mask_path = os.path.join(SVF_DIR, eq_mask_name)
        hs_mask_path = os.path.join(SVF_DIR, hs_mask_name)
        # Save equirectangular (original) image as PNG
        img_pil.save(eq_path)
        logger.info(f"Saved equirectangular SVF image to {eq_path}")
        # Save equirectangular mask as PNG (quantized)
        if mask is not None:
            bins = [0.13, 0.38, 0.63, 0.88, 1.01]
            quantized = np.digitize(mask, bins)
            quantized_5level = (quantized * 64).clip(0, 255).astype(np.uint8)
            Image.fromarray(quantized_5level, mode='L').save(eq_mask_path)
            logger.info(f"Saved equirectangular mask to {eq_mask_path}")
        # Save hemispherical (projected) image as PNG
        out_size = min(img.shape[1], img.shape[0])  # Use min(w, h) for 1:1
        crop_n = int(letterbox_crop)
        img_cropped = img[crop_n:img.shape[0]-crop_n, :, :] if crop_n > 0 else img
        fisheye_img = project_equi_to_fisheye(img_cropped, out_size)
        Image.fromarray(fisheye_img).save(hs_path)
        logger.info(f"Saved hemispherical SVF image to {hs_path}")
        # --- Additional exports: Rotated hemisphere (north up) ---
        fisheye_north = rotate_north_up(fisheye_img, w_point, w_img)
        hs_north_name = f"{base}_SVF{svf_value*100:.3f}_HS_N.png"
        hs_north_path = os.path.join(SVF_DIR, hs_north_name)
        Image.fromarray(fisheye_north.astype(np.uint8)).save(hs_north_path)
        logger.info(f"Saved rotated hemisphere (north up) to {hs_north_path}")
        # --- Additional exports: Rotated hemisphere with annotations ---
        fisheye_north_annot = annotate_directions_on_hemisphere(fisheye_north, w_point=w_point, w_img=w_img)
        hs_north_annot_name = f"{base}_SVF{svf_value*100:.3f}_HS_N_annotated.png"
        hs_north_annot_path = os.path.join(SVF_DIR, hs_north_annot_name)
        Image.fromarray(fisheye_north_annot.astype(np.uint8)).save(hs_north_annot_path)
        logger.info(f"Saved rotated hemisphere with annotations to {hs_north_annot_path}")
        # --- CSV export ---
        csv_path = os.path.join(SVF_DIR, "svf_results.csv")
        Station_ID = base
        base_path = image_file  # includes extension
        svf_val = float(svf_value)
        eq_orig = eq_name
        eq_mask = eq_mask_name
        hs_proj = hs_name
        hs_mask = hs_mask_name
        hs_north = hs_north_name  # now contains the filename of the HS_N image
        hs_north_annot = hs_north_annot_name  # filename for annotated image
        # Compute north pixel (EQ x value) if w_point is set
        if w_point is not None:
            px, py = w_point
            px_north = int((px - w_img // 4) % w_img)
        else:
            px_north = ""
        # --- Compute SVF for each percentage level (100%, 75%, 50%, 25%) ---
        svf_100 = svf_75 = svf_50 = svf_25 = None
        if mask is not None:
            # For each level, create a binary mask and compute SVF
            bins = [0.13, 0.38, 0.63, 0.88, 1.01]
            quantized = np.digitize(mask, bins)
            # 4: 100%, 3: 75%, 2: 50%, 1: 25%
            mask_100 = (quantized == 4).astype(np.float32)
            mask_75 = (quantized == 3).astype(np.float32)
            mask_50 = (quantized == 2).astype(np.float32)
            mask_25 = (quantized == 1).astype(np.float32)
            svf_100 = compute_svf(mask_100, crop=letterbox_crop, h=h)
            svf_75 = compute_svf(mask_75, crop=letterbox_crop, h=h)
            svf_50 = compute_svf(mask_50, crop=letterbox_crop, h=h)
            svf_25 = compute_svf(mask_25, crop=letterbox_crop, h=h)
        # Write header if file does not exist
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow([
                    "Station_ID", "base_path", "svf", "svf_100", "svf_75", "svf_50", "svf_25",
                    "eq_orig", "eq_mask", "hs_proj", "hs_mask", "hs_north", "hs_north_annot", "px_north"
                ])
            writer.writerow([
                Station_ID, base_path, svf_val, svf_100, svf_75, svf_50, svf_25,
                eq_orig, eq_mask, hs_proj, hs_mask, hs_north, hs_north_annot, px_north
            ])
        logger.info(f"Wrote CSV row to {csv_path}")
        # Update file list so user can go back to this file
        reload_files()
        # Show "Saved images." message in green
        status_text = "Saved images."
        draw_status_bar()
        # Draw green status bar for saved message
        status_bar_surface = pygame.Surface((status_bar_rect.width, status_bar_rect.height))
        status_bar_surface.fill((40, 120, 40))  # green
        screen.blit(status_bar_surface, (status_bar_rect.x, status_bar_rect.y))
        status_font = pygame.font.Font(MODERN_FONT_NAME, 20)
        status_surf = status_font.render(status_text, MODERN_FONT_ANTIALIAS, (255, 255, 255))
        screen.blit(status_surf, (32, status_bar_rect.y + 6))
        pygame.display.update(status_bar_rect)
        pygame.event.pump()
        # ...existing code...

    def next_image():
        nonlocal image_idx
        image_idx = (image_idx + 1) % len(files)
        load_image_by_idx(image_idx)
        point_manager.clear()
        masks.clear()
        nonlocal mask, svf_value, status_text
        mask = None
        svf_value = 0  # Set SVF to 0 on image change
        status_text = f"Loaded {image_file}. Add points to segment."

    def prev_image():
        nonlocal image_idx
        image_idx = (image_idx - 1) % len(files)
        load_image_by_idx(image_idx)
        point_manager.clear()
        masks.clear()
        nonlocal mask, svf_value, status_text
        mask = None
        svf_value = 0  # Set SVF to 0 on image change
        status_text = f"Loaded {image_file}. Add points to segment."

    # --- Main event loop ---
    while running:
        # Only allow interaction after model is loaded
        if not model_loading_status['done']:
            draw_left_panel()
            draw_image_area()
            draw_status_bar()
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logger.info('Quit event received')
                    running = False
            clock.tick(30)
            continue
        if model_loading_status['done'] and not model_loading_status['success']:
            draw_left_panel()
            draw_image_area()
            draw_status_bar()
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            continue
        # On first successful load, set up predictor and image
        if model_loaded[0] is not None and 'predictor' not in locals():
            sam, predictor = model_loaded[0]
            predictor.set_image(img)
            logger.info(model_loading_status['status'])
        # --- Always update toggle rects at start of event loop ---
        show_points_rect, show_masks_rect, horizon_toggle_rect, spherical_toggle_rect, save_btn_rect, prev_btn_rect, next_btn_rect, w_dot_info = draw_left_panel()
        draw_image_area()
        draw_status_bar()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logger.info('Quit event received')
                running = False
            elif event.type == pygame.KEYDOWN:
                logger.debug(f'Keydown: {event.key}')
                if event.key == pygame.K_ESCAPE:
                    esc_count += 1
                    logger.debug(f'ESC pressed {esc_count} times')
                    if esc_count < 3:
                        status_text = f"Press ESC {3 - esc_count} more time(s) to exit. (Current action: ESC)"
                    if esc_count >= 3:
                        logger.info('ESC pressed 3 times, exiting')
                        running = False
                elif event.key == pygame.K_r:
                    logger.info('Resetting all points and masks')
                    point_manager.clear()
                    masks.clear()
                    esc_count = 0
                    status_text = "Reset. Add points to segment."
                    mask = None  # Clear the mask overlay as well
                    svf_value = 0  # Reset SVF value to 0
                    w_point = None  # Clear W-Markers as well
                elif event.key == pygame.K_LEFT:
                    offset_x -= 50
                    clamp_offsets()
                    logger.info(f'Scroll left: offset_x={offset_x}')
                elif event.key == pygame.K_RIGHT:
                    offset_x += 50
                    clamp_offsets()
                    logger.info(f'Scroll right: offset_x={offset_x}')
                elif event.key == pygame.K_UP:
                    offset_y -= 50
                    clamp_offsets()
                    logger.info(f'Scroll up: offset_y={offset_y}')
                elif event.key == pygame.K_DOWN:
                    offset_y += 50
                    clamp_offsets()
                    logger.info(f'Scroll down: offset_y={offset_y}')
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    # Zoom in
                    old_scale = scale_factor
                    scale_factor = min(MAX_SCALE, scale_factor * 1.2)
                    scale_factor = max(MIN_SCALE, scale_factor)
                    logger.info(f'Zoom in: scale_factor={scale_factor}')
                    update_ui_image()
                    mx, my = zoom_center
                    offset_x = int((offset_x + mx) * scale_factor / old_scale - mx)
                    offset_y = int((offset_y + my - 60) * scale_factor / old_scale - (my - 60))
                    clamp_offsets()
                elif event.key == pygame.K_MINUS:
                    # Zoom out
                    old_scale = scale_factor
                    scale_factor = max(MIN_SCALE, scale_factor / 1.2)
                    scale_factor = min(MAX_SCALE, scale_factor)
                    logger.info(f'Zoom out: scale_factor={scale_factor}')
                    update_ui_image()
                    mx, my = zoom_center
                    offset_x = int((offset_x + mx) * scale_factor / old_scale - mx)
                    offset_y = int((offset_y + my - 60) * scale_factor / old_scale - (my - 60))
                    clamp_offsets()
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                    idx = event.key - pygame.K_1
                    if 0 <= idx < len(CLICK_TYPES):
                        active_click_type = idx
                        status_text = f"Selected {CLICK_TYPES[idx]['name']} ({int(CLICK_TYPES[idx]['weight']*100)}%)"
                        redraw_left_panel()
                        redraw_status_bar()
                elif event.key == pygame.K_RETURN:
                    save_image_svf()
                    redraw_left_panel()
                    redraw_status_bar()
                elif event.key == pygame.K_n:
                    next_image()
                    redraw_left_panel()
                    redraw_image_area()
                    redraw_status_bar()
                elif event.key == pygame.K_b:
                    prev_image()
                    redraw_left_panel()
                    redraw_image_area()
                    redraw_status_bar()
                elif event.key == pygame.K_5:
                    active_click_type = len(CLICK_TYPES)
                    status_text = "Selected W (Set West direction)"
                    redraw_left_panel()
                    redraw_status_bar()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                # --- Left panel navigation buttons ---
                if save_btn_rect.collidepoint(mx, my):
                    save_image_svf()
                    redraw_left_panel()
                    redraw_status_bar()
                    continue
                if prev_btn_rect.collidepoint(mx, my):
                    prev_image()
                    redraw_left_panel()
                    redraw_image_area()
                    redraw_status_bar()
                    continue
                if next_btn_rect.collidepoint(mx, my):
                    next_image()
                    redraw_left_panel()
                    redraw_image_area()
                    redraw_status_bar()
                    continue
                # --- Left panel toggles ---
                if show_points_rect and show_points_rect.collidepoint(mx, my):
                    show_points = not show_points
                    status_text = f"Show Points: {'ON' if show_points else 'OFF'}"
                    redraw_left_panel()
                    redraw_image_area()
                    redraw_status_bar()
                    continue
                if show_masks_rect and show_masks_rect.collidepoint(mx, my):
                    show_masks = not show_masks
                    status_text = f"Show Masks: {'ON' if show_masks else 'OFF'}"
                    redraw_left_panel()
                    redraw_image_area()
                    redraw_status_bar()
                    continue
                if horizon_toggle_rect and horizon_toggle_rect.collidepoint(mx, my):
                    show_horizon = not show_horizon
                    status_text = f"Show Horizon: {'ON' if show_horizon else 'OFF'}"
                    redraw_left_panel()
                    redraw_image_area()
                    redraw_status_bar()
                    continue
                if spherical_toggle_rect and spherical_toggle_rect.collidepoint(mx, my):
                    show_spherical_view = not show_spherical_view
                    status_text = f"Spherical View: {'ON' if show_spherical_view else 'OFF'}"
                    redraw_left_panel()
                    redraw_image_area()
                    redraw_status_bar()
                    continue
                # Check if click is in click type selector (row layout, left panel, centered)
                dot_radius = 18  # Match the new radius
                sep_y = 28 + panel_heading_font.get_height() + 15
                selector_y = sep_y + 32
                heading_margin = 32
                n_dots = len(CLICK_TYPES)
                total_dots = n_dots + 1
                dots_area_width = 220
                selector_x_start = heading_margin + 28
                spacing_x = dots_area_width // (total_dots - 1) if total_dots > 1 else 0
                for i in range(n_dots):
                    cx = selector_x_start + i * spacing_x
                    cy = selector_y
                    if (mx - cx) ** 2 + (my - cy) ** 2 <= (dot_radius + 7) ** 2:
                        active_click_type = i
                        status_text = f"Selected {CLICK_TYPES[i]['name']} ({int(CLICK_TYPES[i]['weight']*100)}%)"
                        break
                else:
                    # Check if click is in the pink 'W' dot
                    w_cx = selector_x_start + n_dots * spacing_x
                    w_cy = selector_y
                    w_dot_radius = dot_radius
                    if (mx - w_cx) ** 2 + (my - w_cy) ** 2 <= (w_dot_radius + 7) ** 2:
                        active_click_type = len(CLICK_TYPES)
                        status_text = "Selected W (Set West direction)"
                        break
                    # --- DRAGGING LOGIC ---
                    # Only start dragging if:
                    # - Middle mouse button (button 2)
                    # - OR left mouse button (button 1) with Ctrl held
                    # - AND click is in image area (not left panel)
                    dragging = False
                    if (event.button == 2 or (event.button == 1 and (pygame.key.get_mods() & pygame.KMOD_CTRL))) and mx >= LEFT_PANEL_WIDTH and my < img_area_h:
                        dragging = True
                        logger.info('Begin drag')
                        continue  # Don't process as marker click
                    # --- Marker logic below ---
                    img_x = img_y = None
                    if show_spherical_view:
                        out_size = min(img_area_w, img_area_h)
                        x = mx - LEFT_PANEL_WIDTH
                        y = my
                        crop_n = int(letterbox_crop)
                        res = project_point_fisheye_to_equi(x, y, w, h, out_size // 2)
                        if res is not None:
                            img_x, img_y = res
                            img_y += crop_n
                    else:
                        if mx < LEFT_PANEL_WIDTH or my >= img_area_h:
                            continue
                        x = mx - LEFT_PANEL_WIDTH
                        y = my
                        img_x = int((x + offset_x) / scale_factor)
                        img_y = int((y + offset_y) / scale_factor)
                    if img_x is None or img_y is None:
                        continue
                    # Only set W marker if left mouse button (not drag/scroll)
                    if active_click_type == len(CLICK_TYPES) and event.button == 1 and not (pygame.key.get_mods() & pygame.KMOD_CTRL):
                        # Set W-point and show all four direction points
                        w_point = (img_x, img_y)
                        status_text = f"Set W-point at x={img_x}, y={img_y}"
                        # Do not redraw/continue here; let normal flow handle UI
                        # continue  # <-- removed to prevent image disappearing
                    elif event.button == 1:
                        point_manager.add_point(img_x, img_y, active_click_type)
                        status_text = f"Added point at x={img_x}, y={img_y} ({CLICK_TYPES[active_click_type]['name']})"
                    elif event.button == 3:
                        point_manager.remove_nearest(x, y, scale_factor, offset_x, offset_y)
                        status_text = f"Removed nearest point."
                    else:
                        continue
                # --- UI update before mask prediction ---
                prev_status_text = status_text
                status_text = process_message  # Show 'Processing...' in yellow
                draw_status_bar()
                pygame.display.update(pygame.Rect(0, window_h - STATUS_BAR_HEIGHT, window_w, STATUS_BAR_HEIGHT))
                pygame.event.pump()  # Ensure UI is responsive
                # --- End status bar update ---
                # Predict masks for all points and combine (weighted)
                all_points = point_manager.get_all()
                h_int = int(h)
                w_int = int(w)
                combined_mask = np.zeros((h_int, w_int), dtype=np.float32)
                for px, py, type_idx in all_points:
                    coords = np.array([[px, py]], dtype=np.float32)
                    labels = np.ones(1, dtype=np.int32)
                    weight = CLICK_TYPES[type_idx]["weight"]
                    try:
                        area_masks, _, _ = predictor.predict(
                            point_coords=coords,
                            point_labels=labels,
                            multimask_output=False,
                        )
                    except Exception as e:
                        logger.error(f'Predictor error: {e}')
                        continue
                    if area_masks is not None and len(area_masks) > 0:
                        area_mask = area_masks[0].astype(np.float32)
                        area_mask = np.squeeze(area_mask)
                        if area_mask.shape != (h_int, w_int):
                            try:
                                area_mask = area_mask.reshape((h_int, w_int))
                            except Exception as e:
                                logger.error(f"[MASK] Could not reshape area_mask: {e}")
                                continue
                        # --- Multipolygon support: keep all connected components ---
                        num_labels, labels_im = cv2.connectedComponents(area_mask.astype(np.uint8))
                        if num_labels > 1:
                            # Create a mask that is the union of all non-background components
                            multi_mask = (labels_im > 0).astype(np.float32)
                        else:
                            multi_mask = area_mask
                        weighted_mask = multi_mask * weight
                        combined_mask = np.maximum(combined_mask, weighted_mask)
                    else:
                        logger.warning('No mask returned for point')
                # Normalize to [0,1] (clip)
                combined_mask = np.clip(combined_mask, 0, 1)
                # --- Letterbox crop: set top and bottom N pixels to black ---
                crop_n = int(letterbox_crop)
                if crop_n > 0 and crop_n * 2 < h_int:
                    combined_mask[:crop_n, :] = 0.0
                    combined_mask[-crop_n:, :] = 0.0
                mask = combined_mask if np.any(combined_mask > 0) else None
                # Save quantized mask as PNG (4 levels: 0,1,2,3,4)
                if mask is not None:
                    # Quantize: 0=bg, 1=yellow, 2=orange, 3=red, 4=purple
                    bins = [0.13, 0.38, 0.63, 0.88, 1.01]  # right edges
                    quantized = np.digitize(mask, bins)  # 0-4
                    logger.debug(f"Quantized mask unique values: {np.unique(quantized)}")
                    # Save 5-level mask (0, 64, 128, 192, 255) as 'mask_YYYYMMDD_HHMMSS.png'
                    quantized_5level = (quantized * 64).clip(0, 255).astype(np.uint8)
                    out_path_mask = os.path.join(RESULTS_DIR, f"mask_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S').replace('-', '')}.png")
                    img_out_mask = Image.fromarray(quantized_5level, mode='L')
                    img_out_mask.save(out_path_mask)
                    logger.info(f"Saved mask PNG to {out_path_mask}")
                mask = combined_mask if np.any(combined_mask > 0) else None
                # Save quantized mask as PNG (4 levels: 0,1,2,3,4)
                if mask is not None:
                    # Quantize: 0=bg, 1=yellow, 2=orange, 3=red, 4=purple
                    bins = [0.13, 0.38, 0.63, 0.88, 1.01]  # right edges
                    quantized = np.digitize(mask, bins)  # 0-4
                    logger.debug(f"Quantized mask unique values: {np.unique(quantized)}")
                    # Save 5-level mask (0, 64, 128, 192, 255) as 'mask_YYYYMMDD_HHMMSS.png'
                    quantized_5level = (quantized * 64).clip(0, 255).astype(np.uint8)
                    out_path_mask = os.path.join(RESULTS_DIR, f"mask_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S').replace('-', '')}.png")
                    img_out_mask = Image.fromarray(quantized_5level, mode='L')
                    img_out_mask.save(out_path_mask)
                    logger.info(f"Saved mask PNG to {out_path_mask}")
                    # --- SVF calculation ---
                    svf_value = compute_svf(mask, crop=letterbox_crop, h=h)
                    logger.info(f"SVF: {svf_value*100:.1f}%")
                processing = False
                status_text = prev_status_text

                # --- UI update after mask prediction ---
                # Only clear and redraw the image area, not the status bar
                screen.fill((0, 0, 0), rect=image_area_rect)
                draw_image_area()
                draw_status_bar()  # Always last
                pygame.display.update([image_area_rect, status_bar_rect])
                pygame.event.pump()
                # --- End UI update ---
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2 or (event.button == 1 and (pygame.key.get_mods() & pygame.KMOD_CTRL)):
                    dragging = False
                    logger.info('End drag')
            elif event.type == pygame.MOUSEWHEEL:
                # Enable zoom with Ctrl + mouse wheel
                if pygame.key.get_mods() & pygame.KMOD_CTRL:
                    old_scale = scale_factor
                    if event.y > 0:
                        # Zoom in
                        scale_factor = min(MAX_SCALE, scale_factor * 1.2)
                        scale_factor = max(MIN_SCALE, scale_factor)
                    elif event.y < 0:
                        # Zoom out
                        scale_factor = max(MIN_SCALE, scale_factor / 1.2)
                        scale_factor = min(MAX_SCALE, scale_factor)
                    update_ui_image()
                    mx, my = zoom_center
                    offset_x = int((offset_x + mx) * scale_factor / old_scale - mx)
                    offset_y = int((offset_y + my - 60) * scale_factor / old_scale - (my - 60))
                    clamp_offsets()
                    redraw_image_area()
                    redraw_status_bar()
            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                if dragging and not show_spherical_view:
                    dx, dy = event.rel
                    offset_x -= dx
                    offset_y -= dy
                    clamp_offsets()
                    # Fast drag: only blit the cached image, no overlays/masks/points
                    screen.fill((0, 0, 0), rect=image_area_rect)
                    img_rect = pygame.Rect(offset_x, offset_y, img_area_w, img_area_h)
                    screen.blit(img_ui_surf, (LEFT_PANEL_WIDTH, 0), img_rect)
                    pygame.display.update(image_area_rect)
                    continue  # Skip rest of MOUSEMOTION for performance
                if show_spherical_view:
                    out_size = min(img_area_w, img_area_h)
                    x = mx - LEFT_PANEL_WIDTH
                    y = my
                    res = project_point_fisheye_to_equi(x, y, w, h, out_size // 2)
                    if res is not None:
                        mouse_img_x, mouse_img_y = res
                        mouse_img_x = max(0, min(mouse_img_x, w - 1))
                        mouse_img_y = max(0, min(mouse_img_y, h - 1))
                    else:
                        mouse_img_x = None
                        mouse_img_y = None
                else:
                    if LEFT_PANEL_WIDTH <= mx < window_w and 0 <= my < img_area_h:
                        x = mx - LEFT_PANEL_WIDTH
                        y = my
                        img_x = int((x + offset_x) / scale_factor)
                        img_y = int((y + offset_y) / scale_factor)
                        mouse_img_x = max(0, min(img_x, w - 1))
                        mouse_img_y = max(0, min(img_y, h - 1))
                    else:
                        mouse_img_x = None
                        mouse_img_y = None

    logger.info('Exiting SAM Segmenter')
    pygame.quit()
    sys.exit()

# --- Hemisphere rotation and annotation helpers ---
def rotate_north_up(fisheye_img, w_point, w_img):
    """
    Rotate the hemispherical (fisheye) image so that north is up, using the W point as reference.
    North is 90° *counterclockwise* from west (i.e., -π/2 azimuth from W, since view is up).
    If w_point is None, returns the image unchanged.
    """
    if w_point is None or w_img is None:
        return fisheye_img
    import numpy as np
    from PIL import Image
    px, _ = w_point
    # North is 90° counterclockwise from West (viewport up)
    north_x = (px - w_img // 4) % w_img
    north_angle = 360.0 * north_x / w_img
    img_pil = Image.fromarray(fisheye_img)
    rotated = img_pil.rotate(-north_angle, resample=Image.BILINEAR)
    # --- Fix: rotate by 180° to match annotated HS image orientation ---
    rotated = rotated.rotate(180, resample=Image.BILINEAR)
    return np.array(rotated)

def annotate_directions_on_hemisphere(fisheye_img, w_point=None, w_img=None):
    """
    Overlay dashes and labels for W, N, E, S directions on the hemisphere image.
    The image and label positions are rotated 180°, but the label text is not rotated.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    # --- Remove: do NOT rotate the image here, it is already rotated ---
    pil_img = Image.fromarray(fisheye_img)
    out_img = np.array(pil_img)
    h, w = out_img.shape[:2]
    cx, cy = w // 2, h // 2
    r = min(cx, cy)
    draw = ImageDraw.Draw(pil_img, 'RGBA')
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 72)
    except Exception:
        font = ImageFont.load_default()
    # Compute direction azimuths: W, N, E, S (counterclockwise from W)
    if w_point is not None and w_img is not None:
        px, _ = w_point
        base = px % w_img
        directions = [
            (base, 'E'),
            ((base - w_img//4) % w_img, 'S'),
            ((base + w_img//2) % w_img, 'W'),
            ((base + w_img//4) % w_img, 'N'),
        ]
        north_x = (px - w_img // 4) % w_img
        north_angle = 360.0 * north_x / w_img
    else:
        directions = [
            (0, 'W'),
            (3*w//4, 'N'),
            (w//2, 'E'),
            (w//4, 'S'),
        ]
        north_angle = 0.0
    angles_labels = []
    for x, label in directions:
        phi = 2 * np.pi * (x / (w_img if w_img else w))
        angle = np.degrees(phi)
        angle = (angle - 90) % 360  # rotate so that 0° is up (N)
        angle = (angle - north_angle) % 360
        # --- Shift all positions by 180° so S is down, N is up ---
        angle = (angle + 180) % 360
        angles_labels.append((angle, label))
    dash_len = 120
    dash_color = (255, 255, 255, 220)
    label_color = (255, 255, 255, 255)
    outline_color = (0, 0, 0, 255)
    for angle, label in angles_labels:
        theta = np.deg2rad(angle)
        # Make lines thicker and longer
        x0 = cx + int((r - dash_len) * np.cos(theta))
        y0 = cy + int((r - dash_len) * np.sin(theta))
        x1 = cx + int((r - 24) * np.cos(theta))
        y1 = cy + int((r - 24) * np.sin(theta))
        draw.line([(x0, y0), (x1, y1)], fill=dash_color, width=12)
        # Move label further inwards
        label_r = r - 100
        lx = cx + int(label_r * np.cos(theta))
        ly = cy + int(label_r * np.sin(theta))
        try:
            w_label, h_label = font.getsize(label)
        except AttributeError:
            w_label, h_label = font.getbbox(label)[2:4] if hasattr(font, 'getbbox') else (48, 48)
        # Draw black outline by drawing text at offsets
        for ox in [-3, 0, 3]:
            for oy in [-3, 0, 3]:
                if ox == 0 and oy == 0:
                    continue
                draw.text((lx - w_label//2 + ox, ly - h_label//2 + oy), label, font=font, fill=outline_color)
        # Draw white text
        draw.text((lx - w_label//2, ly - h_label//2), label, font=font, fill=label_color)
    return np.array(pil_img)

if __name__ == '__main__':
    main()