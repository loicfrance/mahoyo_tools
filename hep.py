"""
header : 20 bytes :

0x00:0x03   magic bytes : HEP\0
0x04:0x07   file size (little endian, from start of header to last byte of file)
0x08:0x0F   ? (e.g., 00 00 30 1D 06 00 00 00 / 00 00 30 25 06 00 00 00), seems to depend on dimensions
0x10:0x13   ? (e.g., 10 00 00 00)
0x14:0x17   width
0x18:0x1B   height
0x1C:0x1F   ? (e.g., 02 00 00 00)
{width*height}
n*4 bytes : ? palette (RGBA)
"""

from io import BufferedReader, BytesIO
import struct
from typing import TYPE_CHECKING

import numpy as np

from .mzx import mzx_decompress
if TYPE_CHECKING :
    from .mzp import MzpImage

def fix_alpha(a) :
    if a & 0x80 == 0 :
        return np.uint8(((a << 1) | (a >> 6)) & 0xFF)
    else :
        return np.uint8(0xFF)
np_fix_alpha = np.vectorize(fix_alpha)

HEP_MAGIC = int.from_bytes(b'HEP\0', 'little')
HEP_HEADER_SIZE = 0x20
HEP_PALETTE_SIZE = 0x400

def hep_extract_tile(mzp: "MzpImage", tile_index: int) :
    decomp = mzx_decompress(mzp[tile_index+1].to_file())

    (   magic, file_size, _, _, _, width, height, _
    ) = struct.unpack("<IIIIIIII", decomp.read(HEP_HEADER_SIZE))
    nb_pixels = width*height
    assert magic == HEP_MAGIC
    assert width == mzp.tile_width
    assert height == mzp.tile_height
    assert file_size == HEP_HEADER_SIZE + nb_pixels + HEP_PALETTE_SIZE

    decomp.seek(HEP_HEADER_SIZE + nb_pixels)
    palette = np.frombuffer(decomp.read(HEP_PALETTE_SIZE), dtype=np.uint8)
    assert palette.size == HEP_PALETTE_SIZE
    palette.shape = (256, 4)
    palette = np.hstack((palette[:, :3], np_fix_alpha(palette[:, 3:])), dtype=np.uint8)

    decomp.seek(HEP_HEADER_SIZE)
    pixels = palette[np.frombuffer(decomp.read(nb_pixels), dtype=np.uint8)]

    return pixels.tobytes()

def insert_tile(mzp: "MzpImage", tile_index: int, src: str|BufferedReader,
                keepPalette: bool = True) :
    # TODO
    pass