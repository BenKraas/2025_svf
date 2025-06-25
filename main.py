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

    # Get first image in directory
    files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]
    if not files:
        logger.error(f"No images found in {IMAGE_DIR}")
        print(f"No images found in {IMAGE_DIR}")
        return
    image_path = os.path.join(IMAGE_DIR, files[0])
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
        # --- Draw click type selector in a single row, centered in left half ---
        dot_radius = 14
        selector_y = sep_y + 32
        panel_half = LEFT_PANEL_WIDTH // 2
        dots_area_width = panel_half - 2 * heading_margin
        n_dots = len(CLICK_TYPES)
        if n_dots > 1:
            spacing_x = dots_area_width // (n_dots - 1)
        else:
            spacing_x = 0
        selector_x_start = heading_margin + dot_radius  # shift right by one dot width
        dots_row_center = selector_x_start + (dots_area_width // 2)
        # Center the row of dots in the left half
        dot_centers = []
        for i, t in enumerate(CLICK_TYPES):
            cx = selector_x_start + i * spacing_x
            cy = selector_y
            dot_centers.append((cx, cy))
            # Draw background highlight if selected
            if i == active_click_type:
                pygame.gfxdraw.filled_circle(screen, cx, cy, dot_radius + 5, (255, 255, 255))
            # Antialiased color dot
            pygame.gfxdraw.aacircle(screen, cx, cy, dot_radius, t["color"])
            pygame.gfxdraw.filled_circle(screen, cx, cy, dot_radius, t["color"])
            # Draw border
            pygame.gfxdraw.aacircle(screen, cx, cy, dot_radius, (40, 40, 40))
            pygame.gfxdraw.aacircle(screen, cx, cy, dot_radius-1, (40, 40, 40))
            # Percentage label below
            label = panel_font_subtle.render(f"{int(t['weight']*100)}%", MODERN_FONT_ANTIALIAS, (255,255,255))
            label_rect = label.get_rect(center=(cx, cy + dot_radius + 16))
            screen.blit(label, label_rect)
        # --- SVF (%) label to the right of the dots, vertically centered, larger and bolder ---
        svf_label_font = pygame.font.Font(MODERN_FONT_NAME, 24)
        svf_label = svf_label_font.render("SVF (%)", True, (220, 220, 230))
        label_x = panel_half + (LEFT_PANEL_WIDTH - panel_half) // 2
        label_rect = svf_label.get_rect(center=(label_x, selector_y))
        screen.blit(svf_label, label_rect)
        # Divider
        row_bottom = selector_y + dot_radius + 32
        pygame.draw.line(screen, PANEL_DIVIDER, (heading_margin, row_bottom), (LEFT_PANEL_WIDTH - heading_margin, row_bottom), 2)
        # Move instructions to bottom, subtle
        y = window_h - 28 * len(left_panel_texts) - 24
        for text in left_panel_texts:
            text_surf = panel_font_subtle.render(text, MODERN_FONT_ANTIALIAS, PANEL_SUBTLE)
            screen.blit(text_surf, (heading_margin, y))
            y += 22

    def draw_status_bar():
        # Draw status bar at the bottom (flat grey, less height, text left)
        pygame.draw.rect(screen, STATUS_BG, (0, window_h - STATUS_BAR_HEIGHT, window_w, STATUS_BAR_HEIGHT))
        status_surf = status_font.render(status_text, MODERN_FONT_ANTIALIAS, STATUS_TEXT)
        # Align status text with left panel margin
        screen.blit(status_surf, (32, window_h - STATUS_BAR_HEIGHT + 6))

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

    def draw_image_area():
        # Draw checkerboard background for transparency debug (in image area)
        checker_size = 20
        for y in range(0, img_area_h, checker_size):
            for x in range(0, img_area_w, checker_size):
                color = (180, 180, 180) if (x // checker_size + y // checker_size) % 2 == 0 else (100, 100, 100)
                pygame.draw.rect(screen, color, (LEFT_PANEL_WIDTH + x, y, checker_size, checker_size))
        # Draw base image (UI image, scrolled) as the lowest layer
        img_rect = pygame.Rect(offset_x, offset_y, img_area_w, img_area_h)
        surf = pygame.surfarray.make_surface(img_ui.swapaxes(0, 1))
        screen.blit(surf, (LEFT_PANEL_WIDTH, 0), img_rect)
        # Overlay improved mask as purple with 0.6 opacity (downscale for UI, scrolled)
        if mask is not None:
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
        point_radius = 7
        for area in area_manager.get_all():
            for px, py, type_idx in area:
                sx = int(px * scale_factor) - offset_x
                sy = int(py * scale_factor) - offset_y
                if 0 <= sx < img_area_w and 0 <= sy < img_area_h:
                    pygame.draw.circle(screen, (0, 0, 0), (LEFT_PANEL_WIDTH + sx, sy), point_radius + 2)
                    pygame.draw.circle(screen, CLICK_TYPES[type_idx]["color"], (LEFT_PANEL_WIDTH + sx, sy), point_radius)

    esc_count = 0
    save_mask = False
    clock = pygame.time.Clock()
    running = True
    processing = False
    process_message = "Processing..."

    while running:
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
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                # Check if click is in click type selector (row layout, left half, centered)
                dot_radius = 14
                sep_y = 28 + panel_heading_font.get_height() + 15
                selector_y = sep_y + 32
                heading_margin = 32
                panel_half = LEFT_PANEL_WIDTH // 2
                dots_area_width = panel_half - 2 * heading_margin
                n_dots = len(CLICK_TYPES)
                if n_dots > 1:
                    spacing_x = dots_area_width // (n_dots - 1)
                else:
                    spacing_x = 0
                selector_x_start = heading_margin + dot_radius  # shift right by one dot width
                for i in range(len(CLICK_TYPES)):
                    cx = selector_x_start + i * spacing_x
                    cy = selector_y
                    if (mx - cx) ** 2 + (my - cy) ** 2 <= (dot_radius + 8) ** 2 and mx < panel_half:
                        active_click_type = i
                        status_text = f"Selected {CLICK_TYPES[i]['name']} ({int(CLICK_TYPES[i]['weight']*100)}%)"
                        break
                else:
                    # Only handle events in image area
                    if mx < LEFT_PANEL_WIDTH or my >= img_area_h:
                        continue
                    x = mx - LEFT_PANEL_WIDTH
                    y = my
                    if event.button == 2:  # Only middle mouse for dragging
                        dragging = True
                        logger.info(f'Start drag at {event.pos}')
                    elif event.button == 4:  # Mouse wheel up
                        if pygame.key.get_mods() & pygame.KMOD_CTRL:
                            old_scale = scale_factor
                            scale_factor = min(MAX_SCALE, scale_factor * 1.2)
                            scale_factor = max(MIN_SCALE, scale_factor)
                            logger.info(f'Zoom in (ctrl+wheel): scale_factor={scale_factor}')
                            update_ui_image()
                            offset_x = int((offset_x + x) * scale_factor / old_scale - x)
                            offset_y = int((offset_y + y) * scale_factor / old_scale - y)
                            clamp_offsets()
                    elif event.button == 5:  # Mouse wheel down
                        if pygame.key.get_mods() & pygame.KMOD_CTRL:
                            old_scale = scale_factor
                            scale_factor = max(MIN_SCALE, scale_factor / 1.2)
                            scale_factor = min(MAX_SCALE, scale_factor)
                            logger.info(f'Zoom out (ctrl+wheel): scale_factor={scale_factor}')
                            update_ui_image()
                            offset_x = int((offset_x + x) * scale_factor / old_scale - x)
                            offset_y = int((offset_y + y) * scale_factor / old_scale - y)
                            clamp_offsets()
                    elif event.button == 1 or event.button == 3:
                        esc_count = 0
                        img_x = int((x + offset_x) / scale_factor)
                        img_y = int((y + offset_y) / scale_factor)
                        logger.info(f'Mouse click at ({mx}, {my}) [img: ({img_x}, {img_y})], button {event.button}')
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
                            save_mask = False
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
                        processing = False
                        status_text = prev_status_text
                        # --- UI update after mask prediction ---
                        # Only clear and redraw the area below the status bar
                        screen.fill((0, 0, 0), rect=pygame.Rect(0, STATUS_BAR_HEIGHT, window_w, window_h - STATUS_BAR_HEIGHT))
                        draw_image_area()
                        draw_status_bar()
                        pygame.display.flip()
                        pygame.event.pump()
                        # --- End UI update ---
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:  # Only middle mouse for dragging
                    dragging = False
                    logger.info('End drag')
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx, dy = event.rel
                offset_x -= dx
                offset_y -= dy
                clamp_offsets()
        # --- Always redraw the UI every frame for smooth dragging ---
        draw_left_panel()
        draw_image_area()
        draw_status_bar()
        pygame.display.flip()
        clock.tick(30)

    logger.info('Exiting SAM Segmenter')
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()