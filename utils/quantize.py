from importlib.util import find_spec
from typing import Literal, Optional, Dict, cast
import numpy as np
import cffi

def _quantize_libimagequant(input_data, width: int, height: int,
    dithering_level: float = 1.0, max_colors: int = 256,
    min_quality: int = 0, max_quality: int = 100,
    edge_flags: Optional[Dict[str, bool]] = None) :

    from imagequant._libimagequant import lib, ffi

    def _get_error_msg(code) :
        """Get error string from libimagequant return codes.

        :param int code: The code returned by libimagequant.

        :rtype: str
        """
        if code == lib.LIQ_QUALITY_TOO_LOW :
            return "Quality too low"
        elif code == lib.LIQ_VALUE_OUT_OF_RANGE :
            return "Value out of range"
        elif code == lib.LIQ_OUT_OF_MEMORY :
            return "Out of memory"
        elif code == lib.LIQ_ABORTED :
            return "Aborted"
        elif code == lib.LIQ_BITMAP_NOT_AVAILABLE :
            return "Bitmap not available"
        elif code == lib.LIQ_BUFFER_TOO_SMALL :
            return "Buffer too small"
        elif code == lib.LIQ_INVALID_POINTER :
            return "Invalid pointer"
        elif code == lib.LIQ_UNSUPPORTED :
            return "Unsupported"
        else :
            return "Unknown error"

    # Create quantization attributes
    liq_attr = lib.liq_attr_create()
    lib.liq_set_max_colors(liq_attr, max_colors)
    lib.liq_set_quality(liq_attr, min_quality, max_quality)

    # Create image
    liq_image = lib.liq_image_create_rgba(liq_attr, input_data,
                                          width, height, 0)
    if not liq_image :
        lib.liq_attr_destroy(liq_attr)
        raise RuntimeError("Failed to create liq_image")

    # Add a transparent black color (0, 0, 0, 0)
    # to avoid replacing it with 71, 112, 76, 0 (hex: 47, 70, 4C, 00)
    color = ffi.new("liq_color *", (0, 0, 0, 0))
    code = lib.liq_image_add_fixed_color(liq_image, color[0])
    if code != lib.LIQ_OK :
        lib.liq_image_destroy(liq_image)
        lib.liq_attr_destroy(liq_attr)
        raise RuntimeError(_get_error_msg(code))
    ffi.release(color)

    # Create importance map if edge_flags is provided
    if edge_flags is not None :
        # Default edge_flags
        edge_flags = edge_flags or {"left": False, "right": False,
                                    "top": False, "bottom": False}

        # Set weights:
        # - Central zone (1 pixels inset): 255
        # - Outer edge: 1
        inner_weight = 255
        outer_weight = 1

        # Initialize importance map with minimal weight
        importance_map = np.full((height, width), outer_weight, dtype=np.uint8)

        # Central zone
        importance_map[1:-1, 1:-1] = inner_weight

        # Override weights for edges on image boundary
        if edge_flags.get("left", False) :
            importance_map[:, 0] = inner_weight  # Outer left edge
        if edge_flags.get("right", False) :
            importance_map[:, -1] = inner_weight  # Outer right edge
        if edge_flags.get("top", False) :
            importance_map[0, :] = inner_weight  # Outer top edge
        if edge_flags.get("bottom", False) :
            importance_map[-1, :] = inner_weight  # Outer bottom edge

        # Convert importance map to contiguous C-compatible buffer
        importance_map = np.ascontiguousarray(importance_map, dtype=np.uint8)
        if importance_map.size != width * height :
            lib.liq_image_destroy(liq_image)
            lib.liq_attr_destroy(liq_attr)
            raise ValueError(f"Importance map size {importance_map.size} "
                                f"does not match {width * height}")

        importance_buffer = ffi.from_buffer("unsigned char[]", importance_map)
        if not importance_buffer :
            lib.liq_image_destroy(liq_image)
            lib.liq_attr_destroy(liq_attr)
            raise RuntimeError("Failed to allocate importance buffer")

        code = lib.liq_image_set_importance_map(liq_image, importance_buffer,
                                                width * height,
                                                lib.LIQ_COPY_PIXELS)
        if code != lib.LIQ_OK :
            lib.liq_image_destroy(liq_image)
            lib.liq_attr_destroy(liq_attr)
            ffi.release(importance_buffer)
            raise RuntimeError(_get_error_msg(code))
        ffi.release(importance_buffer)

    # Quantize image
    liq_result_p = ffi.new("liq_result**")
    code = lib.liq_image_quantize(liq_image, liq_attr, liq_result_p)
    if code != lib.LIQ_OK :
        lib.liq_result_destroy(liq_result_p[0])
        lib.liq_image_destroy(liq_image)
        lib.liq_attr_destroy(liq_attr)
        raise RuntimeError(_get_error_msg(code))

    code = lib.liq_set_dithering_level(liq_result_p[0], dithering_level)
    if code != lib.LIQ_OK :
        lib.liq_result_destroy(liq_result_p[0])
        lib.liq_image_destroy(liq_image)
        lib.liq_attr_destroy(liq_attr)
        raise RuntimeError(_get_error_msg(code))

    raw_8bit_pixels = ffi.new("char[]", width * height)
    code = lib.liq_write_remapped_image(
        liq_result_p[0], liq_image, raw_8bit_pixels, width * height
    )
    if code != lib.LIQ_OK :
        lib.liq_result_destroy(liq_result_p[0])
        lib.liq_image_destroy(liq_image)
        lib.liq_attr_destroy(liq_attr)
        ffi.release(raw_8bit_pixels)
        raise RuntimeError(_get_error_msg(code))

    # Get palette
    pal = lib.liq_get_palette(liq_result_p[0])

    output_image_data = ffi.unpack(raw_8bit_pixels, width * height)
    output_palette = sum([[color.r, color.g, color.b, color.a]
                          for color in pal.entries], [])

    # Cleanup
    lib.liq_result_destroy(liq_result_p[0])
    lib.liq_image_destroy(liq_image)
    lib.liq_attr_destroy(liq_attr)
    ffi.release(raw_8bit_pixels)

    return output_image_data, output_palette

def _quantize_imagequant(pixels: np.ndarray, dithering_level: float = 1,
             max_colors: int = 256, min_quality: int = 0, max_quality: int = 100,
             edge_flags: Optional[Dict[str, bool]] = None) :
             
    # test if imagequant is available
    if find_spec("imagequant") is not None :
        
        width, height, _ = pixels.shape
        indices, palette = _quantize_libimagequant(
            pixels.tobytes(), width, height,
            dithering_level, max_colors,
            min_quality, max_quality, edge_flags
        )
        indices = np.frombuffer(indices, dtype = np.uint8)
        palette = np.fromiter(palette, dtype = np.uint8)
        indices.shape = (width, height, 1)
        palette.shape = (palette.size//4, 4)
        return indices, palette
    else :
        return None

def _quantize_numpy(pixels: np.ndarray, max_colors: int = 256) :
    
    width, height, _ = pixels.shape
    # temporary reshape the image 3D array to a 2D array
    pixels = pixels.reshape((width * height, 4))

    # retrieve initial palette and quantized image
    palette, indices = np.unique(pixels, axis=0, return_inverse=True)
    # if the original image already has sufficiently few colors, return the palette and quantized image
    palette_len = palette.shape[0]
    if palette_len <= max_colors :
        indices = indices.reshape((width, height, 1))
        if max_colors <= 256 :
            indices = indices.astype(np.uint8)
        if palette_len < max_colors :
            # add black pixels at the beginning of the palette (apparently important for some formats)
            pad_size = max_colors - palette_len
            pad = np.repeat(np.array([[0,0,0,255]], dtype=np.uint8), pad_size, axis=0)
            palette = np.vstack((pad, palette))
            indices += pad_size

        
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
    return None

def quantize(pixels: np.ndarray, dithering_level: float = 1,
             max_colors: int = 256, min_quality: int = 0, max_quality: int = 100,
             priority: list[Literal['imagequant', 'numpy']] = ['imagequant', 'numpy'],
             edge_flags: Optional[Dict[str, bool]] = None) :

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

    for method in priority :
        match method :
            case 'imagequant' :
                result = _quantize_imagequant(pixels, dithering_level,
                                              max_colors, min_quality,
                                              max_quality, edge_flags)
            case 'numpy' :
                result = _quantize_numpy(pixels, max_colors)
        if result :
            indices, palette = result
            return indices, palette
    if 'imagequant' in priority :
        raise ImportError(
                "Install `imagequant` module to quantize this image")
    else :
        raise RuntimeError("Could not quantize image. Retry with 'imagequant' "\
            "in `priority` argument")
    
    # otherwise, use home-made implementation of imagequant


    

    target_mse = quality_to_mse(max_quality)
    max_mse = quality_to_mse(min_quality)
    gamma = 0.4545
    speed = 4
    iterations = (max(8-speed, 0) * 3) // 2
    kmeans_iterations = iterations
    kmeans_iterations_limit = 1 / (1 << 23-speed)
    feedback_loop_trials = max(56-9*speed, 0)
    max_histogram_entries = (1 << 17) + (1 << 18)* (10-speed)
    min_posterization_input = 1 if speed >= 8 else 0
    match speed :
        case s if s < 3 : use_dither_map = 2
        case s if s < 5 : use_dither_map = 1
        case _ : use_dither_map = 0
    use_contrat_map = (speed <= 7) or (use_dither_map != 0)
    progress_stage1 = 20 if use_contrat_map else 8
    if feedback_loop_trials < 2 :
        progress_stage1 += 30
    progress_stage3 = 50 // (1 + speed)
    progress_stage2 = 100 - progress_stage1 - progress_stage3

    ignore_bits = max(0, min_posterization_input)

"""
    liq_error err = liq_histogram_add_image(hist, attr, img);
    if (LIQ_OK != err) {
        liq_histogram_destroy(hist);
        return err;
    }

    err = liq_histogram_quantize_internal(hist, attr, false, result_output);
    liq_histogram_destroy(hist);

    return err;
    # -----
    code = lib.liq_image_quantize(liq_image, liq_attr, liq_result_p)
    if code != lib.LIQ_OK:
        raise RuntimeError(_get_error_msg(code))
    lib.liq_set_dithering_level(liq_result_p[0], dithering_level)

    raw_8bit_pixels = ffi.new("char[]", width * height)
    lib.liq_write_remapped_image(
        liq_result_p[0], liq_image, raw_8bit_pixels, width * height
    )

    pal = lib.liq_get_palette(liq_result_p[0])

    output_image_data = ffi.unpack(raw_8bit_pixels, width * height)
    output_palette = _liq_palette_to_raw_palette(pal)

    lib.liq_result_destroy(liq_result_p[0])
    lib.liq_image_destroy(liq_image)
    lib.liq_attr_destroy(liq_attr)
    ffi.release(raw_8bit_pixels)

    return output_image_data, output_palette


    return pixels, palette




def quality_to_mse(quality: int) :
    match quality :
        case 0 : return 1e20
        case 100 : return 0
        case q :
            # curve fudged to be roughly similar to quality of libjpeg
            # except lowest 10 for really low number of colors
            extra_low_quality_fudge = max(0, 0.016 / (0.001 + q) - 0.001)
            return extra_low_quality_fudge + \
                2.5/pow(210.0 + q, 1.2) * (100.1 - q) / 100.0

def alloc_a_color_hash(max_colors: int, surface: int, ignore_bits: int) -> dict :
    if surface > (512*512) :
        estimated_colors = min(max_colors, surface / (ignore_bits + 6))
    else :
        estimated_colors = min(max_colors, surface / (ignore_bits + 5))
    match estimated_colors :
        case c if c < 66_000 : hash_size = 6673
        case c if c < 200_000 : hash_size = 12011
        case _ : hash_size = 24019
    return {
        'hash_size': hash_size,
        'buckets': { }
    }

def pam_add_to_hash(acht: dict, hash: int, boost: int, px: np.ndarray,
                    px_u32: int, row: int, rows: int) -> bool :
    pass

    # head of the hash function stores first 2 colors inline (achl->used = 1..2),
    # to reduce number of allocations of achl->other_items.
    # 
    achl = cast(dict, acht['buckets']).get(px_u32)
    if achl is not None :
        achl['weight'] += boost
        return True
    else :
        
    if (achl->inline1.color.l == px.l && achl->used) {
        achl->inline1.perceptual_weight += boost;
        return true;
    }
    if (achl->used) {
        if (achl->used > 1) {
            if (achl->inline2.color.l == px.l) {
                achl->inline2.perceptual_weight += boost;
                return true;
            }
            // other items are stored as an array (which gets reallocated if needed)
            struct acolorhist_arr_item *other_items = achl->other_items;
            unsigned int i = 0;
            for (; i < achl->used-2; i++) {
                if (other_items[i].color.l == px.l) {
                    other_items[i].perceptual_weight += boost;
                    return true;
                }
            }

            // the array was allocated with spare items
            if (i < achl->capacity) {
                other_items[i] = (struct acolorhist_arr_item){
                    .color = px,
                    .perceptual_weight = boost,
                };
                achl->used++;
                ++acht->colors;
                return true;
            }

            if (++acht->colors > acht->maxcolors) {
                return false;
            }

            struct acolorhist_arr_item *new_items;
            unsigned int capacity;
            if (!other_items) { // there was no array previously, alloc "small" array
                capacity = 8;
                if (acht->freestackp <= 0) {
                    // estimate how many colors are going to be + headroom
                    const size_t mempool_size = ((acht->rows + rows-row) * 2 * acht->colors / (acht->rows + row + 1) + 1024) * sizeof(struct acolorhist_arr_item);
                    new_items = mempool_alloc(&acht->mempool, sizeof(struct acolorhist_arr_item)*capacity, mempool_size);
                } else {
                    // freestack stores previously freed (reallocated) arrays that can be reused
                    // (all pesimistically assumed to be capacity = 8)
                    new_items = acht->freestack[--acht->freestackp];
                }
            } else {
                const unsigned int stacksize = sizeof(acht->freestack)/sizeof(acht->freestack[0]);

                // simply reallocs and copies array to larger capacity
                capacity = achl->capacity*2 + 16;
                if (acht->freestackp < stacksize-1) {
                    acht->freestack[acht->freestackp++] = other_items;
                }
                const size_t mempool_size = ((acht->rows + rows-row) * 2 * acht->colors / (acht->rows + row + 1) + 32*capacity) * sizeof(struct acolorhist_arr_item);
                new_items = mempool_alloc(&acht->mempool, sizeof(struct acolorhist_arr_item)*capacity, mempool_size);
                if (!new_items) return false;
                memcpy(new_items, other_items, sizeof(other_items[0])*achl->capacity);
            }

            achl->other_items = new_items;
            achl->capacity = capacity;
            new_items[i] = (struct acolorhist_arr_item){
                .color = px,
                .perceptual_weight = boost,
            };
            achl->used++;
        } else {
            // these are elses for first checks whether first and second inline-stored colors are used
            achl->inline2.color.l = px.l;
            achl->inline2.perceptual_weight = boost;
            achl->used = 2;
            ++acht->colors;
        }
    } else {
        achl->inline1.color.l = px.l;
        achl->inline1.perceptual_weight = boost;
        achl->used = 1;
        ++acht->colors;
    }
    return true;

def pam_computeacolorhash(acht: dict, pixels: np.ndarray, cols: int, rows: int,
                          importance_map: np.ndarray | None, ignore_bits: int,
                          hash_size: int) -> bool :

    channel_mask: int = (255 >> ignore_bits) << ignore_bits
    channel_hmask: int = (255 >> ignore_bits) ^ 0xFF
    posterize_mask: int = channel_mask * 0x01010101 # repeat byte 4 times
    posterize_high_mask: int = channel_hmask *0x01010101

    # Go through the entire image, building a hash table of colors. */
    for row in range(0, rows) :
        for col in range(0, cols) :
            boost: int

            # RGBA color is casted to long for easier hasing/comparisons
            px: np.ndarray = pixels[row][col]
            px_u32 = int.from_bytes(px.tobytes(), 'little')
            hash: int
            if (px[3] == 0) :
                # "dirty alpha" has different RGBA values that end up being the
                # same fully transparent color
                px_u32 = 0
                hash = 0
                boost = 2000
                if (importance_map) :
                    importance_map++
                
            else :
                # mask posterizes all 4 channels in one go
                px_u32 = (px_u32 & posterize_mask) | ((px_u32 & posterize_high_mask) >> (8 - ignore_bits))
                # fancier hashing algorithms didn't improve much
                hash = px_u32 % hash_size;

                if (importance_map) :
                    boost = *importance_map++;
                else :
                    boost = 255

            if (!pam_add_to_hash(acht, hash, boost, px, row, rows)) {
                return false;
            }
        }

    }
    acht->cols = cols;
    acht->rows += rows;
    return true;
}

def histogram_add_image(histogram, image: np.ndarray, gamma: float, ignore_bits, max_histogram_entries: int) :
    rows, cols, _ = image.shape

    for(int i = 0; i < input_image->fixed_colors_count; i++) {
        liq_error res = liq_histogram_add_fixed_color_f(input_hist, input_image->fixed_colors[i]);
        if (res != LIQ_OK) {
            return res;
        }
    }

    # Step 2: attempt to make a histogram of the colors, unclustered.
    # If at first we don't succeed, increase ignorebits to increase color
    # coherence and try again.

    # Usual solution is to start from scratch when limit is exceeded, but that's
    # not possible if it's not the first image added
    max_histogram_entries = max_histogram_entries;
    acht = None
    while acht is None :
        acht = alloc_a_color_hash(max_histogram_entries, rows * cols, ignore_bits)
        # histogram uses noise contrast map for importance. Color accuracy in noisy areas is not very important.
        # noise map does not include edges to avoid ruining anti-aliasing
        added_ok = pam_computeacolorhash(input_hist->acht, (const rgba_pixel *const *)input_image->rows, cols, rows, input_image->importance_map);
        if added_ok :
            break
        else :
            ignore_bits += 1;
            acht = None
    } while(!input_hist->acht);

    input_hist->had_image_added = true;

    liq_image_free_importance_map(input_image);

    if (input_image->free_pixels && input_image->f_pixels) {
        liq_image_free_rgba_source(input_image); // bow can free the RGBA source if copy has been made in f_pixels
    }

    return LIQ_OK;


typedef struct liq_attr {
    const char *magic_header;
    void* (*malloc)(size_t);
    void (*free)(void*);

    double target_mse, max_mse, kmeans_iteration_limit;
    unsigned int max_colors, max_histogram_entries;
    unsigned int min_posterization_output /* user setting */, min_posterization_input /* speed setting */;
    unsigned int kmeans_iterations, feedback_loop_trials;
    bool last_index_transparent, use_contrast_maps;
    unsigned char use_dither_map;
    unsigned char speed;

    unsigned char progress_stage1, progress_stage2, progress_stage3;
    void *progress_callback;
    void *progress_callback_user_info;

    void *log_callback;
    void *log_callback_user_info;
    void *log_flush_callback;
    void *log_flush_callback_user_info;
} liq_attr;

typedef struct liq_image liq_image;
typedef struct liq_result liq_result;
typedef struct liq_histogram liq_histogram;

typedef struct liq_color {
    unsigned char r, g, b, a;
} liq_color;

typedef struct liq_palette {
    unsigned int count;
    liq_color entries[256];
} liq_palette;

typedef enum liq_error {
    LIQ_OK = 0,
    LIQ_QUALITY_TOO_LOW = 99,
    LIQ_VALUE_OUT_OF_RANGE = 100,
    LIQ_OUT_OF_MEMORY,
    LIQ_ABORTED,
    LIQ_BITMAP_NOT_AVAILABLE,
    LIQ_BUFFER_TOO_SMALL,
    LIQ_INVALID_POINTER,
    LIQ_UNSUPPORTED,
} liq_error;

enum liq_ownership {
    LIQ_OWN_ROWS=4,
    LIQ_OWN_PIXELS=8,
    LIQ_COPY_PIXELS=16,
};

typedef struct liq_histogram_entry {
    liq_color color;
    unsigned int count;
} liq_histogram_entry;

//----------------------

static double quality_to_mse(long quality)
{
    if (quality == 0) {
        return MAX_DIFF;
    }
    if (quality == 100) {
        return 0;
    }

    // curve fudged to be roughly similar to quality of libjpeg
    // except lowest 10 for really low number of colors
    const double extra_low_quality_fudge = MAX(0,0.016/(0.001+quality) - 0.001);
    return extra_low_quality_fudge + 2.5/pow(210.0 + quality, 1.2) * (100.1-quality)/100.0;
}

static unsigned int mse_to_quality(double mse)
{
    for(int i=100; i > 0; i--) {
        if (mse <= quality_to_mse(i) + 0.000001) { // + epsilon for floating point errors
            return i;
        }
    }
    return 0;
}

liq_error liq_set_quality(liq_attr* attr, int minimum, int target)
{
    if (!CHECK_STRUCT_TYPE(attr, liq_attr)) return LIQ_INVALID_POINTER;
    if (target < 0 || target > 100 || target < minimum || minimum < 0) return LIQ_VALUE_OUT_OF_RANGE;

    attr->target_mse = quality_to_mse(target);
    attr->max_mse = quality_to_mse(minimum);
    return LIQ_OK;
}

"""