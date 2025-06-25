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

# Configuration
IMAGE_DIR = "images"  # directory with equirectangular images
SAM_MODEL_TYPE = "vit_h"  # model type: vit_h, vit_l, vit_b
SAM_WEIGHTS_PATH = "sam_vit_h_4b8939.pth"  # SAM weights file
RESULTS_DIR = "results"
CACHE_DIR = "cache"

# UI Layout constants
LEFT_PANEL_WIDTH = 320
STATUS_BAR_HEIGHT = 48
PANEL_BG = (40, 40, 60)
PANEL_TEXT = (255, 255, 255)
PANEL_HEADING = (180, 220, 255)

# Logging initialization
def init_logging(name, console_level=logging.INFO, file_level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

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
    logger = init_logging('svf', console_level=logging.INFO, file_level=logging.DEBUG)
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
    # Allow window to update
    for _ in range(10):
        pygame.event.pump()
        pygame.time.wait(10)

    # Prepare cache dir
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # Get first image in directory
    files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]
    logger.debug(f"Found files: {files}")
    if not files:
        logger.error(f"No images found in {IMAGE_DIR}")
        print(f"No images found in {IMAGE_DIR}")
        return
    image_path = os.path.join(IMAGE_DIR, files[0])
    logger.info(f"Loading image: {image_path}")
    img_pil = Image.open(image_path).convert("RGB")
    img = np.array(img_pil)
    h, w, _ = img.shape

    # Set up window for half-size image, but now with left panel and status bar
    base_scale = 0.5
    scale_factor = base_scale
    w_ui, h_ui = int(w * scale_factor), int(h * scale_factor)
    # Window size: left panel + image area, status bar at bottom
    window_w = LEFT_PANEL_WIDTH + min(w_ui, 1200)
    window_h = min(h_ui, 900) + STATUS_BAR_HEIGHT
    img_area_w = window_w - LEFT_PANEL_WIDTH
    img_area_h = window_h - STATUS_BAR_HEIGHT
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

    # SAM setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_WEIGHTS_PATH)
    sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(img)

    pygame.font.init()
    font = pygame.font.SysFont(None, 28)
    panel_font = pygame.font.SysFont(None, 24)
    panel_heading_font = pygame.font.SysFont(None, 32, bold=True)
    status_font = pygame.font.SysFont(None, 26)
    # All non-status texts for the left panel
    left_panel_texts = [
        "Left click: add point (new area)",
        "Right click: remove point",
        "R: reset",
        "ESC x3: quit",
        "Pan: middle mouse or Ctrl+left drag",
        "Arrow keys: scroll",
        "+/- or Ctrl+wheel: zoom"
    ]
    status_text = "Add points to segment."

    class PointAreasManager:
        def __init__(self):
            self.areas = [[]]  # List of lists of (x, y)
            self.current = 0
        def add_point(self, x, y):
            self.areas[self.current].append((x, y))
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
        # Draw left panel background
        pygame.draw.rect(screen, PANEL_BG, (0, 0, LEFT_PANEL_WIDTH, window_h))
        # Heading
        heading_surf = panel_heading_font.render("SAM SVF Calculation", True, PANEL_HEADING)
        screen.blit(heading_surf, (20, 24))
        # Divider
        pygame.draw.line(screen, (80, 100, 140), (20, 70), (LEFT_PANEL_WIDTH - 20, 70), 2)
        # Instructions
        y = 90
        for text in left_panel_texts:
            text_surf = panel_font.render(text, True, PANEL_TEXT)
            screen.blit(text_surf, (24, y))
            y += 32

    def draw_status_bar():
        # Draw status bar at the bottom
        pygame.draw.rect(screen, (30, 30, 30), (0, window_h - STATUS_BAR_HEIGHT, window_w, STATUS_BAR_HEIGHT))
        status_surf = status_font.render(status_text, True, (255, 255, 0))
        screen.blit(status_surf, (LEFT_PANEL_WIDTH + 16, window_h - STATUS_BAR_HEIGHT + 12))

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
            mask_bin = (mask > 0.5).astype(np.uint8)
            mask_ui = np.array(Image.fromarray(mask_bin * 255).resize((w_ui, h_ui), Image.NEAREST))
            purple_mask_ui = np.zeros((h_ui, w_ui, 4), dtype=np.uint8)
            purple_mask_ui[..., 0] = 128
            purple_mask_ui[..., 1] = 0
            purple_mask_ui[..., 2] = 128
            purple_mask_ui[..., 3] = (mask_ui > 0) * int(0.6 * 255)
            crop_x0, crop_x1 = offset_x, offset_x + img_area_w
            crop_y0, crop_y1 = offset_y, offset_y + img_area_h
            purple_crop = purple_mask_ui[crop_y0:crop_y1, crop_x0:crop_x1]
            purple_surf = pygame.image.frombuffer(purple_crop.tobytes(), (img_area_w, img_area_h), 'RGBA').convert_alpha()
            screen.blit(purple_surf, (LEFT_PANEL_WIDTH, 0))
        # Draw green points with black outline for all areas (map to UI scale, scrolled) ABOVE the mask
        point_radius = 7
        for area in area_manager.get_all():
            for px, py in area:
                sx = int(px * scale_factor) - offset_x
                sy = int(py * scale_factor) - offset_y
                if 0 <= sx < img_area_w and 0 <= sy < img_area_h:
                    pygame.draw.circle(screen, (0, 0, 0), (LEFT_PANEL_WIDTH + sx, sy), point_radius + 2)
                    pygame.draw.circle(screen, (0, 255, 0), (LEFT_PANEL_WIDTH + sx, sy), point_radius)

    # Show loading while model loads
    screen.fill((30, 30, 30))
    loading_text = temp_font.render("Loading model...", True, (255, 255, 255))
    screen.blit(loading_text, (LEFT_PANEL_WIDTH + img_area_w // 2 - 180, img_area_h // 2 - 24))
    pygame.display.flip()
    for _ in range(10):
        pygame.event.pump()
        pygame.time.wait(10)

    # SAM setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_WEIGHTS_PATH)
    sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(img)

    pygame.font.init()
    font = pygame.font.SysFont(None, 28)
    panel_font = pygame.font.SysFont(None, 24)
    panel_heading_font = pygame.font.SysFont(None, 32, bold=True)
    status_font = pygame.font.SysFont(None, 26)
    # All non-status texts for the left panel
    left_panel_texts = [
        "Left click: add point (new area)",
        "Right click: remove point",
        "R: reset",
        "ESC x3: quit",
        "Pan: middle mouse or Ctrl+left drag",
        "Arrow keys: scroll",
        "+/- or Ctrl+wheel: zoom"
    ]
    status_text = "Add points to segment."

    class PointAreasManager:
        def __init__(self):
            self.areas = [[]]  # List of lists of (x, y)
            self.current = 0
        def add_point(self, x, y):
            self.areas[self.current].append((x, y))
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
        # Draw left panel background
        pygame.draw.rect(screen, PANEL_BG, (0, 0, LEFT_PANEL_WIDTH, window_h))
        # Heading
        heading_surf = panel_heading_font.render("SAM SVF Calculation", True, PANEL_HEADING)
        screen.blit(heading_surf, (20, 24))
        # Divider
        pygame.draw.line(screen, (80, 100, 140), (20, 70), (LEFT_PANEL_WIDTH - 20, 70), 2)
        # Instructions
        y = 90
        for text in left_panel_texts:
            text_surf = panel_font.render(text, True, PANEL_TEXT)
            screen.blit(text_surf, (24, y))
            y += 32

    def draw_status_bar():
        # Draw status bar at the bottom
        pygame.draw.rect(screen, (30, 30, 30), (0, window_h - STATUS_BAR_HEIGHT, window_w, STATUS_BAR_HEIGHT))
        status_surf = status_font.render(status_text, True, (255, 255, 0))
        screen.blit(status_surf, (LEFT_PANEL_WIDTH + 16, window_h - STATUS_BAR_HEIGHT + 12))

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
            mask_bin = (mask > 0.5).astype(np.uint8)
            mask_ui = np.array(Image.fromarray(mask_bin * 255).resize((w_ui, h_ui), Image.NEAREST))
            purple_mask_ui = np.zeros((h_ui, w_ui, 4), dtype=np.uint8)
            purple_mask_ui[..., 0] = 128
            purple_mask_ui[..., 1] = 0
            purple_mask_ui[..., 2] = 128
            purple_mask_ui[..., 3] = (mask_ui > 0) * int(0.6 * 255)
            crop_x0, crop_x1 = offset_x, offset_x + img_area_w
            crop_y0, crop_y1 = offset_y, offset_y + img_area_h
            purple_crop = purple_mask_ui[crop_y0:crop_y1, crop_x0:crop_x1]
            purple_surf = pygame.image.frombuffer(purple_crop.tobytes(), (img_area_w, img_area_h), 'RGBA').convert_alpha()
            screen.blit(purple_surf, (LEFT_PANEL_WIDTH, 0))
        # Draw green points with black outline for all areas (map to UI scale, scrolled) ABOVE the mask
        point_radius = 7
        for area in area_manager.get_all():
            for px, py in area:
                sx = int(px * scale_factor) - offset_x
                sy = int(py * scale_factor) - offset_y
                if 0 <= sx < img_area_w and 0 <= sy < img_area_h:
                    pygame.draw.circle(screen, (0, 0, 0), (LEFT_PANEL_WIDTH + sx, sy), point_radius + 2)
                    pygame.draw.circle(screen, (0, 255, 0), (LEFT_PANEL_WIDTH + sx, sy), point_radius)

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
                    if esc_count >= 3:
                        logger.info('ESC pressed 3 times, exiting')
                        running = False
                elif event.key == pygame.K_r:
                    logger.info('Resetting all areas and masks')
                    area_manager.clear()
                    masks.clear()
                    esc_count = 0
                    status_text = "Reset. Add points to segment."
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
                    scale_factor = min(2.0, scale_factor * 1.2)
                    logger.info(f'Zoom in: scale_factor={scale_factor}')
                    update_ui_image()
                    # Adjust offset to keep zoom center
                    mx, my = zoom_center
                    offset_x = int((offset_x + mx) * scale_factor / old_scale - mx)
                    offset_y = int((offset_y + my - 60) * scale_factor / old_scale - (my - 60))
                    clamp_offsets()
                elif event.key == pygame.K_MINUS:
                    # Zoom out
                    old_scale = scale_factor
                    scale_factor = max(0.1, scale_factor / 1.2)
                    logger.info(f'Zoom out: scale_factor={scale_factor}')
                    update_ui_image()
                    mx, my = zoom_center
                    offset_x = int((offset_x + mx) * scale_factor / old_scale - mx)
                    offset_y = int((offset_y + my - 60) * scale_factor / old_scale - (my - 60))
                    clamp_offsets()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
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
                        scale_factor = min(2.0, scale_factor * 1.2)
                        logger.info(f'Zoom in (ctrl+wheel): scale_factor={scale_factor}')
                        update_ui_image()
                        offset_x = int((offset_x + x) * scale_factor / old_scale - x)
                        offset_y = int((offset_y + y) * scale_factor / old_scale - y)
                        clamp_offsets()
                elif event.button == 5:  # Mouse wheel down
                    if pygame.key.get_mods() & pygame.KMOD_CTRL:
                        old_scale = scale_factor
                        scale_factor = max(0.1, scale_factor / 1.2)
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
                        area_manager.add_point(img_x, img_y)
                        logger.debug(f'Current area points: {area_manager.get_current()}')
                        save_mask = True
                        logger.debug(f'Added point: {(img_x, img_y)} to area {area_manager.current}')
                        status_text = f"Added point at ({img_x}, {img_y}) to area {area_manager.current + 1}"
                    elif event.button == 3:  # right-click: remove nearest point from any area
                        min_dist = float('inf')
                        min_idx = None
                        min_area = None
                        click_sx = x
                        click_sy = y
                        for area_idx, area in enumerate(area_manager.get_all()):
                            for pt_idx, (px, py) in enumerate(area):
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
                    # Predict masks for all areas and combine
                    all_areas = area_manager.get_all()
                    logger.debug(f'All areas: {all_areas}')
                    combined_mask = np.zeros((h, w), dtype=bool)
                    for area in all_areas:
                        if area and len(area) > 0:
                            coords = np.array(area, dtype=np.float32)
                            logger.debug(f'Predictor input coords: {coords}')
                            labels = np.ones(len(area), dtype=np.int32)
                            logger.info(f'Predicting mask for area with {len(area)} points')
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
                                area_mask = area_masks[0]
                                num_labels, labels_im = cv2.connectedComponents(area_mask.astype(np.uint8))
                                if num_labels > 1:
                                    max_label = 1 + np.argmax([
                                        np.sum(labels_im == i) for i in range(1, num_labels)
                                    ])
                                    area_mask = (labels_im == max_label)
                                combined_mask = np.logical_or(combined_mask, area_mask)
                            else:
                                logger.warning('No mask returned for area')
                    mask = combined_mask if np.any(combined_mask) else None
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