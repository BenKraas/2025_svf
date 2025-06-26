import sys
import os
import pygame
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamPredictor
import datetime
import logging
import cv2
import pygame.gfxdraw
import math

# Configuration
IMAGE_DIR = "images"  # directory with equirectangular images
SAM_MODEL_TYPE = "vit_h"  # model type: vit_h, vit_l, vit_b
SAM_WEIGHTS_PATH = "sam_vit_h_4b8939.pth"  # SAM weights file
RESULTS_DIR = "results"
CACHE_DIR = "cache"

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
    logger = init_logging('svf', console_level=logging.DEBUG, file_level=logging.DEBUG)
    logger.info('Starting SAM Segmenter')
    pygame.init()

    # Temporary loading window
    temp_screen = pygame.display.set_mode((600, 200))
    pygame.display.set_caption("SAM Segmenter - Loading...")
    pygame.font.init()
    temp_font = pygame.font.SysFont(None, 48)
    temp_screen.fill((30, 30, 30))
    loading_text = temp_font.render("Loading image and model...", True, (255, 255, 255))
    temp_screen.blit(loading_text, (50, 80))
    pygame.display.flip()
    for _ in range(10):
        pygame.event.pump()
        pygame.time.wait(10)

    # Prepare cache dir
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # Get first image in directory, prefer test.jpg if present
    files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]
    if not files:
        logger.error(f"No images found in {IMAGE_DIR}")
        print(f"No images found in {IMAGE_DIR}")
        return
    # Prefer 'test.jpg' (case-insensitive), else first file
    preferred = next((f for f in files if f.lower() == "test.jpg"), None)
    image_file = preferred if preferred else files[0]
    image_path = os.path.join(IMAGE_DIR, image_file)
    logger.info(f"Loading image: {image_path}")
    img_pil = Image.open(image_path).convert("RGB")
    img = np.array(img_pil)
    h_img, w_img, _ = img.shape
    logger.debug(f"Loaded image shape: h_img={h_img}, w_img={w_img}")
    h = int(h_img)
    w = int(w_img)
    logger.debug(f"Set h={h}, w={w} (image dimensions)")

    # --- Model loading with timing and UI counter ---
    import time
    from threading import Thread
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_text = f"Loading model to {'GPU' if device == 'cuda' else 'CPU'}..."
    temp_screen.fill((30, 30, 30))
    loading_text = temp_font.render(model_text, True, (255, 255, 255))
    temp_screen.blit(loading_text, (50, 80))
    pygame.display.flip()
    pygame.event.pump()

    start_time = time.time()
    model_loaded = [None]
    def load_model():
        sam_local = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_WEIGHTS_PATH)
        sam_local.to(device)
        predictor_local = SamPredictor(sam_local)
        model_loaded[0] = (sam_local, predictor_local)
    t = Thread(target=load_model)
    t.start()
    approx_time = 10  # Approximate expected load time in seconds
    while model_loaded[0] is None:
        elapsed = int(time.time() - start_time)
        temp_screen.fill((30, 30, 30))
        loading_text = temp_font.render(model_text, True, (255, 255, 255))
        temp_screen.blit(loading_text, (50, 70))
        counter_text = temp_font.render(f"{elapsed} sec / ~{approx_time} sec", True, (180, 180, 180))
        temp_screen.blit(counter_text, (50, 130))
        pygame.display.flip()
        pygame.event.pump()
        pygame.time.wait(200)
    sam, predictor = model_loaded[0]
    total_time = time.time() - start_time
    logger.info(f"Model loaded to {device} in {total_time:.1f} seconds.")
    predictor.set_image(img)

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

    def update_ui_image():
        nonlocal img_ui, w_ui, h_ui
        w_ui, h_ui = int(w * scale_factor), int(h * scale_factor)
        img_ui = np.array(Image.fromarray(img).resize((w_ui, h_ui), Image.LANCZOS))
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
    active_click_type = 0  # Index into CLICK_TYPES

    class PointAreasManager:
        def __init__(self):
            self.areas = [[]]  # List of lists of (x, y, type_idx)
            self.current = 0
        def add_point(self, x, y, type_idx):
            self.areas[self.current].append((x, y, type_idx))
        def remove_last(self):
            if self.areas[self.current]:
                self.areas[self.current].pop()
        def new_area(self):
            self.areas.append([])
            self.current = len(self.areas) - 1
        def clear(self):
            self.areas = [[]]
            self.current = 0
        def get_all(self):
            return self.areas
        def get_current(self):
            return self.areas[self.current]
        def set_current(self, idx):
            if 0 <= idx < len(self.areas):
                self.current = idx
        def __len__(self):
            return sum(len(a) for a in self.areas)

    area_manager = PointAreasManager()
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

    svf_value = None  # Store last SVF value
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
        """
        theta = math.pi * y / h  # zenith angle [0, pi]
        phi = 2 * math.pi * x / w  # azimuth [0, 2pi]
        if theta > math.pi / 2:
            return None  # Only northern hemisphere
        r = out_r * theta / (math.pi / 2)
        u = out_r + r * math.sin(phi)
        v = out_r - r * math.cos(phi)
        return int(u), int(v)

    def fisheye_to_equi_coords(u, v, w, h, out_r):
        """
        Map fisheye (u, v) to equirectangular (x, y).
        Returns (x, y) in equirectangular image, or None if outside hemisphere.
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
        dot_radius = 14
        selector_y = sep_y + 32
        n_dots = len(CLICK_TYPES)
        dots_area_width = 180
        selector_x_start = heading_margin + 10  # Respect left padding
        spacing_x = dots_area_width // (n_dots - 1) if n_dots > 1 else 0
        dot_centers = []
        for i, t in enumerate(CLICK_TYPES):
            cx = selector_x_start + i * spacing_x
            cy = selector_y
            dot_centers.append((cx, cy))
            # Draw background highlight if selected
            if i == active_click_type:
                pygame.draw.circle(screen, (255, 255, 255), (cx, cy), dot_radius + 6)
            pygame.draw.circle(screen, t["color"], (cx, cy), dot_radius)
            # Draw border
            pygame.draw.circle(screen, (40, 40, 40), (cx, cy), dot_radius, 3)
        # Draw percentages below each dot
        percent_font = pygame.font.Font(MODERN_FONT_NAME, 16)
        for i, t in enumerate(CLICK_TYPES):
            cx, cy = dot_centers[i]
            percent_label = f"{int(t['weight']*100)}%"
            percent_surf = percent_font.render(percent_label, MODERN_FONT_ANTIALIAS, (255,255,255))
            percent_rect = percent_surf.get_rect(center=(cx, cy + dot_radius + 14))
            screen.blit(percent_surf, percent_rect)
        # --- SVF (%) label to the right of the dots, vertically centered, larger and bolder ---
        svf_label_font = pygame.font.Font(MODERN_FONT_NAME, 24)
        svf_label = svf_label_font.render("SVF (%)", True, (220, 220, 230))
        right_padding = 32
        svf_label_x = LEFT_PANEL_WIDTH - right_padding - svf_label.get_width() // 2
        label_rect = svf_label.get_rect(center=(svf_label_x, selector_y))
        screen.blit(svf_label, label_rect)
        # --- SVF value below label ---
        if svf_value is not None:
            svf_val_font = pygame.font.Font(MODERN_FONT_NAME, 28)
            svf_val_str = f"{svf_value*100:.3f}"
            svf_val_surf = svf_val_font.render(svf_val_str, True, (80, 200, 255))
            svf_val_x = LEFT_PANEL_WIDTH - right_padding - svf_val_surf.get_width() // 2
            row_bottom = selector_y + dot_radius + 32
            center_y = (selector_y + row_bottom) // 2
            val_rect = svf_val_surf.get_rect(center=(svf_val_x, center_y))
            screen.blit(svf_val_surf, val_rect)
        # Divider below SVF value
        row_bottom = selector_y + dot_radius + 32
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
        # Move instructions to bottom, subtle
        y = window_h - 28 * len(left_panel_texts) - 24
        for text in left_panel_texts:
            text_surf = panel_font_subtle.render(text, MODERN_FONT_ANTIALIAS, PANEL_SUBTLE)
            screen.blit(text_surf, (heading_margin, y))
            y += 22
        return show_points_rect, show_masks_rect, horizon_toggle_rect, spherical_toggle_rect

    def draw_status_bar():
        # Draw status bar at the bottom (flat grey, less height, text left)
        pygame.draw.rect(screen, STATUS_BG, (0, window_h - STATUS_BAR_HEIGHT, window_w, STATUS_BAR_HEIGHT))
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
                for area in area_manager.get_all():
                    for px, py, type_idx in area:
                        crop_n = int(letterbox_crop)
                        py_cropped = py - crop_n
                        h_cropped = h - 2 * crop_n
                        if 0 <= py_cropped < h_cropped:
                            res = project_point_equi_to_fisheye(px, py_cropped, w, h_cropped, out_size // 2)
                            if res is not None:
                                sx, sy = res
                                pygame.draw.circle(screen, (0, 0, 0), (LEFT_PANEL_WIDTH + sx, sy), point_radius + 2)
                                pygame.draw.circle(screen, CLICK_TYPES[type_idx]["color"], (LEFT_PANEL_WIDTH + sx, sy), point_radius)
        else:
            # Equirectangular view
            # Draw checkerboard background for transparency debug (in image area)
            checker_size = 20
            for y in range(0, img_area_h, checker_size):
                for x in range(0, img_area_w, checker_size):
                    color = (180, 180, 180) if (x // checker_size + y // checker_size) % 2 == 0 else (100, 100, 100)
                    pygame.draw.rect(screen, color, (LEFT_PANEL_WIDTH + x, y, checker_size, checker_size))
            # Draw base image (UI, scrolled) as the lowest layer
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
                for area in area_manager.get_all():
                    for px, py, type_idx in area:
                        sx = int(px * scale_factor) - offset_x
                        sy = int(py * scale_factor) - offset_y
                        if 0 <= sx < img_area_w and 0 <= sy < img_area_h:
                            pygame.draw.circle(screen, (0, 0, 0), (LEFT_PANEL_WIDTH + sx, sy), point_radius + 2)
                            pygame.draw.circle(screen, CLICK_TYPES[type_idx]["color"], (LEFT_PANEL_WIDTH + sx, sy), point_radius)
        # --- SVF Math Visualization ---
        if show_math_viz or show_horizon:
            # Draw horizon (green)
            if show_spherical_view:
                # Draw green circle for horizon in spherical view
                out_size = min(img_area_w, img_area_h)
                cx = LEFT_PANEL_WIDTH + out_size // 2
                cy = out_size // 2
                radius = out_size // 2
                pygame.gfxdraw.aacircle(screen, cx, cy, radius, (0, 255, 0))
            else:
                # Draw green line for horizon in equirectangular view
                horizon_y = int(h_ui // 2) - offset_y
                if 0 <= horizon_y < img_area_h:
                    pygame.draw.line(screen, (0, 255, 0), (LEFT_PANEL_WIDTH, horizon_y), (LEFT_PANEL_WIDTH + img_area_w, horizon_y), 3)
        # ...existing code for math viz overlays (if show_math_viz)...

    esc_count = 0
    save_mask = False
    clock = pygame.time.Clock()
    running = True
    processing = False
    process_message = "Processing..."

    # --- Cached rects for partial redraws ---
    left_panel_rect = pygame.Rect(0, 0, LEFT_PANEL_WIDTH, window_h)
    image_area_rect = pygame.Rect(LEFT_PANEL_WIDTH, 0, img_area_w, img_area_h)
    status_bar_rect = pygame.Rect(0, window_h - STATUS_BAR_HEIGHT, window_w, STATUS_BAR_HEIGHT)

    def redraw_all():
        draw_left_panel()
        draw_image_area()
        draw_status_bar()  # Always last
        pygame.display.flip()
        return draw_left_panel(), None, None, None

    def redraw_image_area():
        # Only clear and redraw the image area, not the status bar
        screen.fill((0, 0, 0), rect=image_area_rect)
        draw_image_area()
        draw_status_bar()  # Always draw status bar last
        pygame.display.update([image_area_rect, status_bar_rect])

    def redraw_left_panel():
        draw_left_panel()
        draw_status_bar()  # Always draw status bar last
        pygame.display.update([left_panel_rect, status_bar_rect])

    def redraw_status_bar():
        draw_status_bar()
        pygame.display.update(status_bar_rect)

    while running:
        # Only redraw everything if dragging or continuous interaction
        needs_full_redraw = dragging
        if needs_full_redraw:
            show_points_rect, show_masks_rect, horizon_toggle_rect, spherical_toggle_rect = redraw_all()
        else:
            redraw_left_panel()
            redraw_image_area()
            redraw_status_bar()
            show_points_rect, show_masks_rect, horizon_toggle_rect, spherical_toggle_rect = draw_left_panel()
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
                    logger.info('Resetting all areas and masks')
                    area_manager.clear()
                    masks.clear()
                    esc_count = 0
                    status_text = "Reset. Add points to segment."
                    mask = None  # Clear the mask overlay as well
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
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
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
                dot_radius = 14
                sep_y = 28 + panel_heading_font.get_height() + 15
                selector_y = sep_y + 32
                heading_margin = 32
                n_dots = len(CLICK_TYPES)
                dots_area_width = 180
                selector_x_start = heading_margin + 10  # Match draw_left_panel
                spacing_x = dots_area_width // (n_dots - 1) if n_dots > 1 else 0
                for i in range(n_dots):
                    cx = selector_x_start + i * spacing_x
                    cy = selector_y
                    if (mx - cx) ** 2 + (my - cy) ** 2 <= (dot_radius + 8) ** 2:
                        active_click_type = i
                        status_text = f"Selected {CLICK_TYPES[i]['name']} ({int(CLICK_TYPES[i]['weight']*100)}%)"
                        break
                else:
                    # Only handle events in image area
                    img_x = img_y = None
                    if show_spherical_view:
                        out_size = min(img_area_w, img_area_h)
                        x = mx - LEFT_PANEL_WIDTH
                        y = my
                        crop_n = int(letterbox_crop)
                        res = project_point_fisheye_to_equi(x, y, w, h - 2 * crop_n, out_size // 2)
                        if res is not None:
                            img_x, img_y = res
                            img_y += crop_n  # adjust back to full image coordinates
                    else:
                        if mx < LEFT_PANEL_WIDTH or my >= img_area_h:
                            continue
                        x = mx - LEFT_PANEL_WIDTH
                        y = my
                        img_x = int((x + offset_x) / scale_factor)
                        img_y = int((y + offset_y) / scale_factor)
                    if img_x is None or img_y is None:
                        continue
                    if event.button == 1:  # left-click: add point as new area
                        area_manager.new_area()
                        area_manager.add_point(img_x, img_y, active_click_type)
                        logger.debug(f'Current area points: {area_manager.get_current()}')
                        save_mask = True
                        logger.debug(f'Added point: {(img_x, img_y)} to area {area_manager.current}')
                        status_text = f"Added point at ({img_x}, {img_y}) to area {area_manager.current + 1} ({CLICK_TYPES[active_click_type]['name']})"
                    elif event.button == 3:  # right-click: remove nearest point from any area
                        min_dist = float('inf')
                        min_idx = None
                        min_area = None
                        click_sx = x
                        click_sy = y
                        for area_idx, area in enumerate(area_manager.get_all()):
                            for pt_idx, (px, py, type_idx) in enumerate(area):
                                sx = int(px * scale_factor) - offset_x
                                sy = int(py * scale_factor) - offset_y
                                dist = (sx - click_sx) ** 2 + (sy - click_sy) ** 2
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = pt_idx
                                    min_area = area_idx
                        if min_dist < (12 ** 2) and min_area is not None and min_idx is not None:
                            removed = area_manager.areas[min_area][min_idx]
                            del area_manager.areas[min_area][min_idx]
                            logger.debug(f'Removed point: {removed} from area {min_area}')
                            status_text = f"Removed point from area {min_area + 1}."
                    else:
                        save_mask = False
                # --- UI update before mask prediction ---
                prev_status_text = status_text
                status_text = process_message  # Show 'Processing...' in yellow
                draw_status_bar()
                pygame.display.update(pygame.Rect(0, window_h - STATUS_BAR_HEIGHT, window_w, STATUS_BAR_HEIGHT))
                pygame.event.pump()  # Ensure UI is responsive
                # --- End status bar update ---
                # Predict masks for all areas and combine (weighted)
                all_areas = area_manager.get_all()
                # --- Debug: log h, w types and values before mask creation ---
                logger.debug(f"Preparing to create mask array: h={h} (type {type(h)}), w={w} (type {type(w)})")
                h_int = int(h)
                w_int = int(w)
                logger.debug(f"Casted h={h_int} (type {type(h_int)}), w={w_int} (type {type(w_int)})")
                combined_mask = np.zeros((h_int, w_int), dtype=np.float32)
                logger.debug(f"Created combined_mask array with shape {combined_mask.shape}, dtype {combined_mask.dtype}")
                for area in all_areas:
                    if area and len(area) > 0:
                        coords = np.array([(x, y) for x, y, t in area], dtype=np.float32)
                        labels = np.ones(len(area), dtype=np.int32)
                        weights = [CLICK_TYPES[t]["weight"] for _, _, t in area]
                        logger.info(f"Predicting mask for area with {len(area)} points")
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
                            logger.debug(f"[MASK] area_mask.shape before squeeze: {area_mask.shape}")
                            area_mask = np.squeeze(area_mask)
                            logger.debug(f"[MASK] area_mask.shape after squeeze: {area_mask.shape}")
                            if area_mask.shape != (h_int, w_int):
                                logger.warning(f"[MASK] area_mask shape {area_mask.shape} does not match (h, w)=({h_int}, {w_int}), attempting reshape.")
                                try:
                                    area_mask = area_mask.reshape((h_int, w_int))
                                    logger.debug(f"[MASK] area_mask reshaped to {area_mask.shape}")
                                except Exception as e:
                                    logger.error(f"[MASK] Could not reshape area_mask: {e}")
                                    continue
                            num_labels, labels_im = cv2.connectedComponents(area_mask.astype(np.uint8))
                            if num_labels > 1:
                                max_label = 1 + np.argmax([
                                    np.sum(labels_im == i) for i in range(1, num_labels)
                                ])
                                area_mask = (labels_im == max_label).astype(np.float32)
                            # For each point, apply the weight and keep the maximum per pixel
                            # Create a weighted mask for this area (use the max weight in the area)
                            max_weight = max(weights) if weights else 0.0
                            weighted_mask = area_mask * max_weight
                            combined_mask = np.maximum(combined_mask, weighted_mask)
                        else:
                            logger.warning('No mask returned for area')
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
                if event.button == 2:
                    dragging = False
                    logger.info('End drag')
            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
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
                        mouse_img_x = int((x + offset_x) / scale_factor)
                        mouse_img_y = int((y + offset_y) / scale_factor)
                        mouse_img_x = max(0, min(mouse_img_x, w - 1))
                        mouse_img_y = max(0, min(mouse_img_y, h - 1))
                    else:
                        mouse_img_x = None
                        mouse_img_y = None
                if dragging and not show_spherical_view:
                    dx, dy = event.rel
                    offset_x -= dx
                    offset_y -= dy
                    clamp_offsets()
                    redraw_image_area()
        clock.tick(30)

    logger.info('Exiting SAM Segmenter')
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()