import sys
import os
import pygame
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamPredictor

# Configuration
IMAGE_DIR = "images"  # directory with equirectangular images
SAM_MODEL_TYPE = "vit_h"  # model type: vit_h, vit_l, vit_b
SAM_WEIGHTS_PATH = "sam_vit_h_4b8939.pth"  # SAM weights file

# Initialize Pygame
def main():
    pygame.init()

    # Get first image in directory
    files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]
    if not files:
        print(f"No images found in {IMAGE_DIR}")
        return
    image_path = os.path.join(IMAGE_DIR, files[0])
    img_pil = Image.open(image_path).convert("RGB")
    img = np.array(img_pil)
    h, w, _ = img.shape

    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption(f"SAM Segmenter - {files[0]}")

    # SAM setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_WEIGHTS_PATH)
    sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(img)

    points = []
    masks = []
    esc_count = 0

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    esc_count += 1
                    if esc_count >= 3:
                        running = False
                elif event.key == pygame.K_r:
                    points.clear()
                    masks.clear()
                    esc_count = 0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                esc_count = 0
                x, y = event.pos
                if event.button == 1:  # left-click: add point
                    points.append([x, y])
                elif event.button == 3:  # right-click: remove last point
                    if points:
                        points.pop()
                # Predict masks with current points
                if points:
                    coords = np.array(points)
                    labels = np.ones(len(points))
                    masks, _, _ = predictor.predict(
                        point_coords=coords,
                        point_labels=labels,
                        multimask_output=False,
                    )
                else:
                    masks = []

        # Draw base image
        screen.fill((0, 0, 0))
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        screen.blit(surf, (0, 0))

        # Overlay each mask
        for mask in masks:
            mask_surf = pygame.surfarray.make_surface(
                np.dstack([mask * 255] * 3).swapaxes(0, 1)
            )
            mask_surf.set_alpha(128)
            screen.blit(mask_surf, (0, 0))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()