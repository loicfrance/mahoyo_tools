from importlib.util import find_spec
from typing import cast
import numpy as np

def quantize(pixels: np.ndarray, dithering_level: float = 1,
             max_colors: int = 256, min_quality: int = 0, max_quality: int = 100) :

    assert 0 <= dithering_level <= 1, \
        "dithering_level must be a float between 0.0 and 1.0"
    assert 1 <= max_colors <= 256, \
        "max_colors must be an integer between 1 and 256"
    assert 0 <= min_quality <= 100, \
        "min_quality must be an integer between 0 and 100"
    assert 0 <= max_quality <= 100, \
        "max_quality must be an integer between 0 and 100"
    assert min_quality <= max_quality, \
        "min_quality must be lower or equal to max_quality"
    shape = pixels.shape
    assert len(shape) == 3 and shape[2] == 4, \
        "pixels array must have the shape (width, height, 4)"

    width, height, _ = shape

    # try to use imagequant if it is available
    if find_spec("imagequant") is not None :
        import imagequant
        indices, palette = imagequant.quantize_raw_rgba_bytes(
            pixels.tobytes(), width, height,
            dithering_level, max_colors,
            min_quality, max_quality,
        )
        indices = np.frombuffer(indices, dtype = np.uint8)
        palette = np.fromiter(palette, dtype = np.uint8)
        indices.shape = (width, height, 1)
        palette.shape = (palette.size//4, 4)
        return indices, palette
    
    # temporary reshape the image 3D array to a 2D array
    pixels = pixels.reshape((width * height, 4))

    # retrieve initial palette and quantized image
    palette, indices = np.unique(pixels, axis=0, return_inverse=True)
    # if the original image already has sufficiently few colors, return the palette and quantized image
    palette_len = palette.shape[0]
    if palette_len <= max_colors :
        indices = indices.reshape((width, height, 1))
        return indices, palette
    
    # attempt to reduce colors by merging fully-transparent pixels
    first_transparent = -1
    duplicates = []
    for i, alpha in enumerate(palette[:, 3]) :
        if alpha == 0 :
            if first_transparent == -1 :
                first_transparent = i
                palette[i, :3] = np.zeros(3, np.uint8)
            else :
                duplicates.append(i)
    if palette_len - len(duplicates) <= max_colors :
        # TODO remove duplicate colors, re-compute quantized image
        raise NotImplementedError(
            "Multiple fully-transparent pixels with different RGB values.")
    else :
        raise ImportError(
            "Image Quantization not implemented. install `imagequant` module")
    