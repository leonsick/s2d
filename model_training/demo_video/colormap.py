# Copyright (c) Meta Platforms, Inc. and affiliates.
# copied from https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/colormap.py

"""
An awesome colormap for really neat visualizations.
Copied from Detectron, and removed gray colors.
"""

import numpy as np
import random

__all__ = ["colormap", "random_color", "random_colors"]


_COLORS = np.array(
    [
        # --- YOUR INITIAL 3 COLORS ---
        0.000, 0.671, 0.557,  # Teal / Dark Cyan (New)
        0.831, 0.094, 0.463,  # Dark Pink / Raspberry (New)
        0.216, 0.294, 0.737,  # Indigo / Dark Slate Blue (New)
        # --- NEW COHESIVE PALETTE (76 Colors) ---
        0.700, 0.300, 0.000,  # Burnt Orange
        0.900, 0.500, 0.000,  # Deep Gold
        0.200, 0.600, 0.200,  # Forest Green
        0.100, 0.400, 0.500,  # Deep Sea Green
        0.500, 0.100, 0.700,  # Violet / Amethyst
        0.800, 0.600, 0.200,  # Mustard Yellow
        0.600, 0.200, 0.200,  # Deep Maroon
        0.100, 0.700, 0.800,  # Turquoise
        0.400, 0.400, 0.000,  # Dark Olive
        0.900, 0.200, 0.400,  # Ruby Red
        0.000, 0.800, 0.200,  # Bright Emerald
        0.300, 0.100, 0.500,  # Plum
        0.700, 0.400, 0.100,  # Copper
        0.500, 0.500, 0.500,  # Mid Gray
        0.100, 0.100, 0.100,  # Near Black
        0.900, 0.700, 0.100,  # Sunny Yellow
        0.100, 0.600, 0.600,  # Light Teal
        0.600, 0.000, 0.800,  # Dark Lavender
        0.400, 0.700, 0.100,  # Lime Green
        0.000, 0.300, 0.900,  # Royal Blue
        0.800, 0.400, 0.600,  # Mauve
        0.200, 0.900, 0.900,  # Light Aqua
        0.700, 0.100, 0.000,  # Dark Red
        0.300, 0.500, 0.700,  # Cadet Blue
        1.000, 0.800, 0.300,  # Peach
        0.500, 0.200, 0.900,  # Deep Purple
        0.000, 0.500, 0.100,  # Dark Mint
        0.800, 0.800, 0.800,  # Light Gray
        0.600, 0.500, 0.100,  # Brownish Yellow
        0.400, 0.100, 0.200,  # Rusty Brown
        0.900, 0.600, 0.800,  # Pale Pink
        0.200, 0.300, 0.400,  # Dark Slate
        0.700, 0.800, 0.000,  # Bright Olive
        0.300, 0.000, 0.600,  # Deep Indigo
        0.100, 0.900, 0.500,  # Spring Green
        0.600, 0.300, 0.500,  # Dusty Rose
        0.800, 0.200, 0.900,  # Fuchsia
        0.200, 0.700, 0.300,  # Moss Green
        0.500, 0.400, 0.200,  # Tan
        0.900, 0.000, 0.700,  # Vibrant Magenta
        0.100, 0.500, 0.900,  # Deep Cerulean
        0.400, 0.200, 0.800,  # Deep Violet
        0.700, 0.700, 0.100,  # Golden Rod
        0.000, 0.900, 0.700,  # Persian Green
        0.600, 0.100, 0.300,  # Dark Cranberry
        0.300, 0.800, 0.600,  # Medium Aquamarine
        0.500, 0.700, 0.900,  # Light Steel Blue
        0.900, 0.400, 0.200,  # Vermillion
        0.200, 0.000, 0.700,  # Deep Blue
        0.800, 0.900, 0.100,  # Chartreuse
        0.400, 0.500, 0.600,  # Steel Gray
        0.100, 0.200, 0.300,  # Very Dark Slate
        0.700, 0.900, 0.500,  # Pale Green
        0.300, 0.300, 0.700,  # Blue-Gray
        0.900, 0.100, 0.300,  # Hot Pink
        0.000, 0.400, 0.000,  # Darker Green (40%)
        0.000, 0.600, 0.000,  # Medium Green (60%)
        0.000, 0.800, 0.000,  # Bright Green (80%)
        0.000, 0.000, 0.400,  # Darker Blue (40%)
        0.000, 0.000, 0.600,  # Medium Blue (60%)
        0.000, 0.000, 0.800,  # Bright Blue (80%)
        0.400, 0.000, 0.000,  # Darker Red (40%)
        0.600, 0.000, 0.000,  # Medium Red (60%)
        0.800, 0.000, 0.000,  # Bright Red (80%)
        0.400, 0.400, 0.400,  # Dark Gray (40%)
        0.600, 0.600, 0.600,  # Medium Gray (60%)
        0.800, 0.800, 0.800,  # Light Gray (80%)
        0.950, 0.950, 0.950,  # Off White
        0.050, 0.050, 0.050,  # Off Black
        1.000, 0.000, 0.500,  # Pure Pink
        0.500, 1.000, 0.000,  # Pure Yellow-Green
        0.000, 0.500, 1.000,  # Pure Cyan-Blue
        1.000, 0.500, 0.500,  # Salmon
        0.500, 1.000, 0.500,  # Mint
        0.500, 0.500, 1.000,  # Light Blue
        0.100, 0.800, 0.400,  # Jade
        0.800, 0.100, 0.600,  # Deep Rose
        0.300, 0.600, 0.900,  # Sky Blue
    ]
).astype(np.float32).reshape(-1, 3)

# fmt: off
# RGB:
_STANDARD_COLORS = np.array(
    [
        0.000, 0.447, 0.741,  # Cerulean / Deep Sky Blue
        0.850, 0.325, 0.098,  # Rust Orange
        0.929, 0.694, 0.125,  # Goldenrod / Yellow-Orange
        0.494, 0.184, 0.556,  # Dark Magenta / Purple
        0.466, 0.674, 0.188,  # Lime Green / Chartreuse
        0.301, 0.745, 0.933,  # Sky Blue / Cyan
        0.635, 0.078, 0.184,  # Dark Red / Crimson
        0.300, 0.300, 0.300,  # Dark Gray
        0.600, 0.600, 0.600,  # Medium Gray
        1.000, 0.000, 0.000,  # Red (Primary)
        1.000, 0.500, 0.000,  # Orange (Secondary)
        0.749, 0.749, 0.000,  # Olive / Yellow-Green (Close to Yellow)
        0.000, 1.000, 0.000,  # Green (Primary)
        0.000, 0.000, 1.000,  # Blue (Primary)
        0.667, 0.000, 1.000,  # Purple / Magenta
        0.333, 0.333, 0.000,  # Dark Olive
        0.333, 0.667, 0.000,  # Dark Lime Green
        0.333, 1.000, 0.000,  # Bright Green
        0.667, 0.333, 0.000,  # Burnt Orange
        0.667, 0.667, 0.000,  # Dark Yellow
        0.667, 1.000, 0.000,  # Yellow-Green
        1.000, 0.333, 0.000,  # Bright Orange-Red
        1.000, 0.667, 0.000,  # Vivid Orange
        1.000, 1.000, 0.000,  # Yellow (Secondary)
        0.000, 0.333, 0.500,  # Dark Teal / Deep Cyan
        0.000, 0.667, 0.500,  # Medium Spring Green
        0.000, 1.000, 0.500,  # Vivid Spring Green
        0.333, 0.000, 0.500,  # Dark Fuchsia / Deep Purple
        0.333, 0.333, 0.500,  # Slate Blue / Dark Mauve
        0.333, 0.667, 0.500,  # Sage Green / Dusky Teal
        0.333, 1.000, 0.500,  # Light Mint Green
        0.667, 0.000, 0.500,  # Red-Purple / Cerise
        0.667, 0.333, 0.500,  # Dusty Rose
        0.667, 0.667, 0.500,  # Beige / Olive Tan
        0.667, 1.000, 0.500,  # Pale Yellow-Green
        1.000, 0.000, 0.500,  # Hot Pink / Fluorescent Magenta
        1.000, 0.333, 0.500,  # Coral / Salmon Pink
        1.000, 0.667, 0.500,  # Light Coral / Peach
        1.000, 1.000, 0.500,  # Pastel Yellow
        0.000, 0.333, 1.000,  # Bright Blue / Azure
        0.000, 0.667, 1.000,  # Deep Cyan / Brilliant Turquoise
        0.000, 1.000, 1.000,  # Cyan (Secondary)
        0.333, 0.000, 1.000,  # Medium Blue-Violet
        0.333, 0.333, 1.000,  # Periwinkle / Light Royal Blue
        0.333, 0.667, 1.000,  # Sky Blue / Pale Cerulean
        0.333, 1.000, 1.000,  # Pale Cyan / Aqua
        0.667, 0.000, 1.000,  # Vivid Violet / Purple
        0.667, 0.333, 1.000,  # Lavender / Light Purple
        0.667, 0.667, 1.000,  # Lilac / Pale Blue-Violet
        0.667, 1.000, 1.000,  # Very Pale Cyan
        1.000, 0.000, 1.000,  # Magenta (Secondary)
        1.000, 0.333, 1.000,  # Vivid Pink
        1.000, 0.667, 1.000,  # Orchid / Light Magenta
        0.333, 0.000, 0.000,  # Dark Red (33%)
        0.500, 0.000, 0.000,  # Half Red / Maroon (50%)
        0.667, 0.000, 0.000,  # Medium Red (67%)
        0.833, 0.000, 0.000,  # Bright Red (83%)
        1.000, 0.000, 0.000,  # Red (100%)
        0.000, 0.167, 0.000,  # Very Dark Green (17%)
        0.000, 0.333, 0.000,  # Dark Green (33%)
        0.000, 0.500, 0.000,  # Half Green / Medium Green (50%)
        0.000, 0.667, 0.000,  # Medium Light Green (67%)
        0.000, 0.833, 0.000,  # Bright Green (83%)
        0.000, 1.000, 0.000,  # Green (100%)
        0.000, 0.000, 0.167,  # Very Dark Blue (17%)
        0.000, 0.000, 0.333,  # Dark Blue (33%)
        0.000, 0.000, 0.500,  # Half Blue / Medium Blue (50%)
        0.000, 0.000, 0.667,  # Medium Light Blue (67%)
        0.000, 0.000, 0.833,  # Bright Blue (83%)
        0.000, 0.000, 1.000,  # Blue (100%)
        0.000, 0.000, 0.000,  # Black
        0.143, 0.143, 0.143,  # Very Dark Gray
        0.857, 0.857, 0.857,  # Very Light Gray
        1.000, 1.000, 1.000   # White
    ]
).astype(np.float32).reshape(-1, 3)
# fmt: on


def colormap(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1
    Returns:
        ndarray: a float32 array of Nx3 colors, in range [0, 255] or [0, 1]
    """
    assert maximum in [255, 1], maximum
    c = _COLORS * maximum
    if not rgb:
        c = c[:, ::-1]
    return c


def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1
    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def random_colors(N, rgb=False, maximum=255):
    """
    Args:
        N (int): number of unique colors needed
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1
    Returns:
        ndarray: a list of random_color
    """
    indices = random.sample(range(len(_COLORS)), N)
    ret = [_COLORS[i] * maximum for i in indices]
    if not rgb:
        ret = [x[::-1] for x in ret]
    return ret

def select_colors(rgb=False, maximum=255, indices=[0]):
    """
    Args:
        N (int): number of unique colors needed
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1
    Returns:
        ndarray: a list of random_color
    """
    # indices = random.sample(range(len(_COLORS)), N)
    ret = [_COLORS[i] * maximum for i in indices]
    if not rgb:
        ret = [x[::-1] for x in ret]
    return ret

if __name__ == "__main__":
    import cv2

    size = 100
    H, W = 10, 10
    canvas = np.random.rand(H * size, W * size, 3).astype("float32")
    for h in range(H):
        for w in range(W):
            idx = h * W + w
            if idx >= len(_COLORS):
                break
            canvas[h * size : (h + 1) * size, w * size : (w + 1) * size] = _COLORS[idx]
    cv2.imshow("a", canvas)
    cv2.waitKey(0)