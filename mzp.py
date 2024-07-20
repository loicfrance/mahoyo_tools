from __future__ import annotations
from io import BufferedReader, BufferedWriter, BytesIO
from math import ceil, floor
import os
import struct
from typing import cast, overload
import zlib
from PIL import Image, ImagePalette
import numpy as np
from numpy.typing import NDArray

from .mzx import mzx_compress, mzx_decompress
from .hep import hep_extract_tile, hep_insert_tile
from .utils.io import BytesReader, BytesWriter, save
from .utils.quantize import quantize

MZP_FILE_MAGIC = b'mrgd00'
MZP_HEADER_SIZE = len(MZP_FILE_MAGIC) + 2
MZP_ENTRY_HEADER_SIZE = 4*2
MZP_DEFAULT_ALIGNMENT = 8

MZP_SECTOR_SIZE = 0x800

def rgb565_unpack(pq: np.uint16, offsets_byte: np.uint8) :
    r = ((pq & 0xF800) >> 8) + (offsets_byte >> 5)
    g = ((pq & 0x07E0) >> 3) + ((offsets_byte >> 3) & 0x03)
    b = ((pq & 0x001F) << 3) + (offsets_byte & 0x7)
    return np.array([r, g, b], dtype=np.uint8)
np_rgb565_unpack = np.vectorize(rgb565_unpack)

def rgb565_pack(r: np.uint8, g: np.uint8, b: np.uint8) :
    offset = ((r & 0x07) << 5) \
           | ((g & 0x03) << 3) \
           | ((b & 0x07))
    pq = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | ((b & 0xF8) >> 3)
    return np.array([pq, offset], dtype="<u2")
np_rgb565_pack = np.vectorize(rgb565_pack)

def fix_alpha(a: np.uint8) :
    if a & 0x80 == 0 :
        return np.uint8(((a << 1) | (a >> 6)) & 0xFF)
    else :
        return np.uint8(0xFF)
np_fix_alpha = np.vectorize(fix_alpha)

def unfix_alpha(a: np.uint8) :
    if a == 0xFF :
        return a
    else :
        return np.uint8(a >> 1)
np_unfix_alpha = np.vectorize(unfix_alpha)

class MzpArchiveEntry :

    def __init__(self, mzp: MzpArchive, index: int) -> None:
        self._mzp = mzp
        header_offset = MZP_HEADER_SIZE + index * MZP_ENTRY_HEADER_SIZE
        file = mzp._file
        position = file.tell()
        file.seek(header_offset)
        sector_offset = int.from_bytes(file.read(2), 'little', signed=False)
        byte_offset = int.from_bytes(file.read(2), 'little', signed=False)
        size_sectors = int.from_bytes(file.read(2), 'little', signed=False)
        size_bytes = int.from_bytes(file.read(2), 'little', signed=False)
        file.seek(position)
        self.offset = sector_offset * MZP_SECTOR_SIZE + byte_offset

        if True :
            self._size = size_bytes
            while self.size_sectors < size_sectors :
                self._size += MZP_SECTOR_SIZE
        else :
            # sector_start = offset - offset % SIZE,
            # sector_end = sector_start + size_sectors * SIZE,
            # size = (sector_end - offset) & ~0xFFFF | size_bytes
            size = size_sectors * MZP_SECTOR_SIZE - self.offset % MZP_SECTOR_SIZE
            size = (size & ~0xFFFF) | size_bytes

            self._size = size
        assert self.size_sectors == size_sectors
        assert self.size_bytes == size_bytes
        assert self.sector_offset == sector_offset
        assert self.byte_offset == byte_offset
        self._data: bytes|None = None
    
    @property
    def index(self) -> int :
        """Index of the entry in the mzp file"""
        return self._mzp._entries.index(self)
        
    @property
    def header_start(self) -> int :
        """Address of the entry header in the mzp file"""
        return MZP_HEADER_SIZE + self.index * MZP_ENTRY_HEADER_SIZE
    
    @property
    def header(self) -> bytes :
        return self.sector_offset.to_bytes(2, 'little') + \
               self.byte_offset.to_bytes(2, 'little') + \
               self.size_sectors.to_bytes(2, 'little') + \
               self.size_bytes.to_bytes(2, 'little')

    @property
    def sector_offset(self) :
        """Start location of this entry in whole sectors"""
        return int(self.offset / MZP_SECTOR_SIZE)
    
    @property
    def byte_offset(self) :
        """Start location of this entry (sub-sector positioning)"""
        return int(self.offset % MZP_SECTOR_SIZE)
    
    @property
    def size_sectors(self) :
        """The upper bound on the number of sectors that this data exists on"""
        sector_start = floor(self.offset / MZP_SECTOR_SIZE)
        sector_end = ceil((self.offset + self.size) / MZP_SECTOR_SIZE)
        return sector_end - sector_start
    
    @property
    def size_bytes(self) :
        """The raw low 16 bits of the full size of the archive data"""
        return self.size & 0xFFFF

    @property
    def size(self) -> int :
        """Size (in bytes) of the entry data"""
        return self._size
        
    @property
    def data_start(self) -> int :
        """Address of the entry data in the mzp file"""
        return self._mzp.data_start_offset() + self.offset
    
    @property
    def data(self) :
        if self._data is None :
            file = self._mzp._file
            pos = file.tell()
            file.seek(self.data_start)
            data = file.read(self.size)
            file.seek(pos)
            return data
        else :
            return self._data

    @data.setter
    def data(self, data: bytes|BytesReader) :
        if not isinstance(data, bytes) :
            assert not isinstance(data, (memoryview, bytearray))
            data = data.read()
        self._data = data
        self._size = len(self._data)
    
    def to_archive(self, file: BytesWriter, alignment: int = MZP_DEFAULT_ALIGNMENT) :
        position = file.tell()
        file.seek(self.header_start)
        file.write(self.header)
        file.seek(self.data_start)
        file.write(self.data)
        file.write(b'\xFF' * (alignment - len(self.data) % alignment))
        file.seek(position)
    
    @overload
    def to_file(self, dest: str | BytesIO) -> None : ...
    @overload
    def to_file(self, dest: None = None) -> BytesIO : ...

    def to_file(self, dest: BytesWriter | str | None = None) -> BytesIO | None :
        return save(self.data, dest)
    
    def from_file(self, src: BytesReader | str) :
        if isinstance(src, str) :
            file = open(src, "rb")
            self.data = file.read()
            file.close()
        else :
            self.data = src.read()
    
    def __repr__(self) :
        return f"MZP-entry {self.index:03}: " + \
            f"offset = 0x{self.offset:08x}[0x{self.sector_offset:04x},0x{self.byte_offset:04x}], " + \
            f"size = 0x{len(self.data):08x}[0x{self.size_sectors:04x},0x{self.size_bytes:04x}]"

class MzpArchive :

    def __init__(self, src: str | bytes) -> None:
        if isinstance(src, str) :
            self._file = open(src, "rb+")
        else :
            self._file = BytesIO(src)
            
        header = self._file.read(MZP_HEADER_SIZE)
        assert header.startswith(MZP_FILE_MAGIC)
        nbEntries = int.from_bytes(header[-2:], "little", signed=False)
        
        self._entries: list[MzpArchiveEntry] = [
            MzpArchiveEntry(self, i) for i in range(nbEntries)
        ]
    
    def __getitem__(self, index: int) -> MzpArchiveEntry :
        return self._entries[index]
    
    def __iter__(self) :
        yield from self._entries
    
    @property
    def nb_entries(self) -> int :
        return len(self._entries)

    @property
    def header(self) -> bytes :
        return MZP_FILE_MAGIC + self.nb_entries.to_bytes(2, 'little')

    def data_start_offset(self) :
        return MZP_HEADER_SIZE + self.nb_entries * MZP_ENTRY_HEADER_SIZE
    
    def update_offsets(self, alignment: int = MZP_DEFAULT_ALIGNMENT) :
        """Re-position the entries offsets"""
        offset = 0
        for entry in self._entries :
            entry.offset = offset
            offset += entry.size
            if alignment > 0 :
                offset += alignment - offset % alignment
    
    @overload
    def mzp_write(self, dest: BytesWriter | str,
                  alignment: int = MZP_DEFAULT_ALIGNMENT) -> None: ...
    @overload
    def mzp_write(self, dest: None = None,
                  alignment: int = MZP_DEFAULT_ALIGNMENT) -> BytesIO: ...

    def mzp_write(self, dest: str | BytesWriter | None = None,
                  alignment: int = MZP_DEFAULT_ALIGNMENT) :
        """Write the mzp archive to a file"""
        match dest :
            case str() :
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                file = open(dest, "wb")
            case None : file = BytesIO()
            case _ : file = dest
        self.update_offsets(alignment)
        file.write(self.header)
        for entry in self :
            entry.to_archive(file, alignment)
        
        match dest :
            case str() : cast(BufferedWriter, file).close()
            case None :
                file.seek(0)
                return file

class MzpImage(MzpArchive) :

    def __init__(self, src: str | bytes) -> None:
        super().__init__(src)
        self.update_img_info()
    
    def update_img_info(self) :
        (self.width, self.height, self.tile_width, self.tile_height,
         self.tile_x_count, self.tile_y_count, self.bmp_type, self.bmp_depth,
         self.tile_crop) = struct.unpack("<HHHHHHHBB", self[0].data[:16])
    
    @property
    def tile_size(self) : return self.tile_width * self.tile_height

    @property
    def nb_channels(self) :
        match self.bmp_type, self.bits_per_px :
            case 0x01, 4|8 : return 1
            case 0x0C, bpp : return 4
            case t   , 24  : return 3
            case t   , 32  : return 4
            case t, bpp : raise ValueError(
                f"Unexpected bmp type - bpp pair 0x{t:02x}, {bpp}")

    @property
    def bits_per_px(self) :
        match (self.bmp_type, self.bmp_depth) :
            case (0x01, 0x00|0x10) : return 4
            case (0x01, 0x01|0x11|0x91) : return 8
            case (0x08, 0x14) : return 24
            case (0x0B, 0x14) : return 32
            case (0x0C, 0x11) : return 32
            case (0x03, depth) : # 'PEH' 8bpp + palette
                raise NotImplementedError("Unsupported bmp type 0x03 (PEH)")
            case (t, d) :
                raise ValueError(f"Unknown bmp type & depth pair 0x{t:02X},0x{d:02X}")
    
    @property
    def bytes_per_px(self) :
        bpp = self.bits_per_px
        if bpp < 8 :
            return 1
        else :
            return floor(bpp / 8)

    def get_palette(self) :
        match (self.bmp_type, self.bmp_depth) :
            case (0x01, 0x00|0x10) : palette_size = 16
            case (0x01, 0x01|0x11|0x91) : palette_size = 256
            case (0x01, unknown_depth) :
                raise ValueError(f"Unknown depth 0x{unknown_depth:02X}")
            case (t, d) : # type != 0x01 (no palette)
                return None
        
        palette = np.frombuffer(self[0].data, dtype = np.uint8, offset = 16,
                                count = palette_size*4).copy()
        palette.shape = (palette_size, 4)
        palette = np.hstack((palette[:, :3], np_fix_alpha(palette[:, 3:])), dtype=np.uint8)
        
        if self.bmp_depth in [0x11, 0x91] :
            # swap palette blocks
            for i in range(0, len(palette), 32) :
                block1 = palette[i+8:i+16].copy()
                palette[i+8:i+16] = palette[i+16:i+24]
                palette[i+16:i+24] = block1

        #palette += np.repeat([(0,0,0,255)] * (0x100 - len(palette))
        filler = np.repeat(np.array([[0,0,0,255]], dtype=np.uint8), 256 - palette_size, axis=0)

        return np.vstack((palette, filler), dtype=np.uint8)
    
    def set_palette(self, palette: np.ndarray | ImagePalette.ImagePalette) :
        match (self.bmp_type, self.bmp_depth) :
            case (0x01, 0x00|0x10) : palette_size = 16
            case (0x01, 0x01|0x11|0x91) : palette_size = 256
            case (0x01, unknown_depth) :
                raise ValueError(f"Unknown depth 0x{unknown_depth:02X}")
            case (t, d) : # type != 0x01 (no palette)
                return
        
        if isinstance(palette, ImagePalette.ImagePalette) :
            mode = palette.mode
            palette = np.fromiter(palette.palette, dtype=np.uint8)
            match mode :
                case "RGB" :
                    palette_size = palette.size // 3
                    palette.shape = (palette_size, 3)
                    alpha = np.frombuffer(b'\xFF'*palette_size, dtype=np.uint8)
                    alpha.shape = (palette_size, 1)
                    palette = np.hstack((palette, alpha))
                case "RGBA" :
                    palette.shape = (palette.size // 4, 4)
                case "L" :
                    palette.shape = (palette.size, 1)
                    alpha = np.frombuffer(b'\xFF'*palette.size, dtype=np.uint8)
                    palette = np.hstack((palette, palette, palette, alpha))
                case _ :
                    raise NotImplementedError(f"Unexpected palette mode {mode}")
        else :
            match palette.shape :
                case (n, 4) : pass
                case (n, 3) : palette = np.hstack((palette, np.full((n, 1), 0xFF, dtype=np.uint8)))
                case (n,) : palette = palette.reshape((n//4, 4))
        assert palette.shape[0] == palette_size
        palette = np.hstack((palette[:, :3], np_unfix_alpha(palette[:, 3:])), dtype=np.uint8)

        if self.bmp_depth in [0x11, 0x91] :
            # swap palette blocks
            for i in range(0, len(palette), 32) :
                block1 = palette[i+8:i+16].copy()
                palette[i+8:i+16] = palette[i+16:i+24]
                palette[i+16:i+24] = block1
        self[0].data = self[0].data[0:16] + palette.tobytes()

    def get_tile(self, index: int) -> np.ndarray :
        tile_file = mzx_decompress(self[index+1].to_file())
        if self.bmp_type == 0x0C :
            return hep_extract_tile(self, index).flatten()\
                .reshape((self.tile_height, self.tile_width, 4))
        match self.bits_per_px :
            case 4 :
                buffer = np.frombuffer(tile_file.getbuffer(), dtype=np.uint8)
                buffer.shape = (buffer.size, 1)
                buffer = np.hstack((buffer & 0x0F, buffer >> 4)).flatten()
                return buffer.reshape((self.tile_height, self.tile_width, 1))
            case 8 :
                buffer = np.frombuffer(tile_file.getbuffer(), dtype=np.uint8)
                buffer.shape = (self.tile_height, self.tile_width, 1)
                #buffer = buffer[:, :, :1]
                return buffer
            case 24 | 32 as bpp: # RGB/RGBA true color for 0x08 and 0x0B bmp type
                assert self.bmp_type in [0x08, 0x0B]
                buffer = tile_file.read()
                rgb565 = np.frombuffer(buffer, dtype='<u2', count=self.tile_size)
                offsets = np.frombuffer(buffer, dtype=np.uint8, offset = self.tile_size*2, count = self.tile_size)
                pixels: NDArray[np.uint8] = np_rgb565_unpack(rgb565, offsets)
                if bpp == 32 :
                    alpha: NDArray[np.uint8] = np.frombuffer(buffer, dtype=np.uint8, offset = self.tile_size*3, count = self.tile_size)
                    pixels = np.hstack((pixels, alpha))
                channels = pixels.shape[1]
                return pixels.flatten().reshape(self.tile_height, self.tile_width, channels)
            case bpp :
                raise ValueError(f"Unexpected {bpp} bpp")
    
    def set_tile(self, index: int, pixels: np.ndarray, compression_level) :
        nb_channels = self.nb_channels
        match pixels.shape :
            case (size,) : pixels = pixels.reshape((size//nb_channels, nb_channels))
            case (nb_px, channels) : pass
            case (w, h, channels) : pixels = pixels.reshape((-1, channels))
            case shape : raise ValueError(f"Unexpected array shape {shape}")
        
        if self.bmp_type == 0x0C :
           hep_insert_tile(self, index, pixels, compression_level)
           return
        
        tile_file = BytesIO()
        bpp = self.bits_per_px
        match bpp :
            case 4 :
                pixels = pixels.reshape((-1)).reshape((pixels.size // 2, 2)) # TODO check if first reshape is necessary
                pixels = (pixels[:, 0] & 0x0F) | ((pixels[:, 1] & 0x0F) << 4)
                tile_file.write(pixels.tobytes())
            case 8 :
                tile_file.write(pixels.tobytes())
            case 24 | 32 as bpp :
                assert self.bmp_type in [0x08, 0x0B]
                if bpp == 32 :
                    alpha = pixels[:, 3]
                    pixels = pixels[:, :3]
                buffer: np.ndarray = np_rgb565_pack(pixels)
                rgb565: np.ndarray = buffer[:, 0]
                offsets: np.ndarray = buffer[:, 1]
                tile_file.write(rgb565.tobytes())
                tile_file.write(offsets.tobytes())
                if bpp == 32 :
                    tile_file.write(alpha.tobytes())
            case _ :
                raise ValueError(f"Unexpected {bpp} bpp")
        tile_file.seek(0)
        self[index+1].from_file(mzx_compress(tile_file, level=compression_level))

    @overload
    def img_write(self, dest: BytesWriter | str) -> None: ...
    @overload
    def img_write(self, dest: None = None) -> BytesIO: ...

    def img_write(self, dest: str | BytesWriter | None = None) :
        crop = self.tile_crop
        height = self.height - self.tile_y_count * crop * 2
        width = self.width - self.tile_x_count * crop * 2
        
        img_pixels = np.zeros((height, width, self.nb_channels), dtype=np.uint8)
        for y in range(self.tile_y_count) :
            start_row = y * (self.tile_height - crop * 2)
            row_count =  min(self.height - start_row, self.tile_height) - crop * 2
            end_row = start_row + row_count
            for x in range(self.tile_x_count) :
                index = self.tile_x_count * y + x
                tile_pixels = self.get_tile(index)
                start_col = x * (self.tile_width - crop * 2)
                col_count = min(self.width - start_col, self.tile_width) - crop * 2
                end_col = start_col + col_count
                img_pixels[start_row:end_row, start_col:end_col] = \
                    tile_pixels[crop:crop+row_count, crop:crop+col_count]
        
        match dest :
            case str() :
                assert dest.lower().endswith(".png")
                file = open(dest, "wb")
            case None : file = BytesIO()
            case _ : file = dest
        
        # PNG SIG
        file.write(b'\x89PNG\x0D\x0A\x1A\x0A')

        match (self.bmp_type, self.bits_per_px) :
            case (0x01, bpp) :
                chunk = struct.pack(">IIBB", width, height, 8, 3) + b'\0\0\0' # 8bpp (PLTE)
                write_pngchunk_withcrc(file, b'IHDR', chunk)
                palette = self.get_palette()
                assert palette is not None
                plte = palette[:, :3].tobytes()
                trns = palette[:, 3].tobytes()
                write_pngchunk_withcrc(file, b'PLTE', plte)
                write_pngchunk_withcrc(file, b'tRNS', trns)
            case (0x0C, bpp) :
                chunk = struct.pack(">IIBB", width, height, 8, 6) + b'\0\0\0'
                write_pngchunk_withcrc(file, b'IHDR', chunk)
            case (t, 24 | 32 as bpp) :
                color_type = 2 if bpp == 24 else 6 # 24bpp RBG or 32bpp RGBA
                chunk = struct.pack(">IIBB", width, height, 8, color_type) + b'\0\0\0'
                write_pngchunk_withcrc(file, b'IHDR', chunk)
            case (t, bpp) :
                raise ValueError(f"Unexpected bmp type - bpp pair 0x{t:02X},{bpp}")

        # split into rows and add png filtering info (mandatory even with no filter)
        padded = np.hstack((
            np.zeros((height, 1), dtype=np.uint8), # add a \x00 at the start of every row
            img_pixels.reshape((height, -1))))
        write_pngchunk_withcrc(file, b"IDAT", zlib.compress(padded.tobytes()))
        write_pngchunk_withcrc(file, b"IEND", b'')
        
        match dest :
            case str() : cast(BufferedWriter, file).close()
            case None :
                file.seek(0)
                return file
            case _ :
                pass
   
    def img_read(self, src: str | BytesReader | bytes, *,
                 compression_level: int = 0) :
        if isinstance(src, bytes) :
            src = BytesIO(src)
        img = Image.open(src)
        
        nb_channels = self.nb_channels
        crop = self.tile_crop
        height = self.height - self.tile_y_count * crop * 2
        width = self.width - self.tile_x_count * crop * 2

        assert img.width == width and img.height == height
        img_pixels = None
        match (self.bmp_type, self.bits_per_px) :
            case (0x01, 4|8 as bpp) :
                if img.mode == "P" and img.palette is not None and \
                     img.palette.mode == "RGBA":
                    self.set_palette(img.palette)
                else :
                    img_pixels = np.array(img.convert("RGBA"))\
                        .reshape((img.height, img.width, 4))
                    palette_size = 16 if bpp == 4 else 256
                    img_pixels, palette = quantize(img_pixels, max_colors = palette_size,
                                            priority=['numpy', 'imagequant'])
                    self.set_palette(palette)
            case (0x0C, bpp) : img = img.convert("RGBA")
            case (_, 24) : img = img.convert("RGB")
            case (_, 32) : img = img.convert("RGBA")
            case (t, bpp) :
                raise ValueError(f"Unexpected bmp type - bpp pair 0x{t:02X},{bpp}")
        if img_pixels is None :
            img_pixels = np.array(img)
        img_pixels.shape = (height, width, nb_channels)
        for y in range(self.tile_y_count) :
            start_row = y * (self.tile_height - crop * 2)
            row_count =  min(self.height - start_row, self.tile_height) - crop * 2
            end_row = start_row + row_count
            for x in range(self.tile_x_count) :
                index = self.tile_x_count * y + x
                start_col = x * (self.tile_width - crop * 2)
                col_count = min(self.width - start_col, self.tile_width) - crop * 2
                end_col = start_col + col_count
                tile_pixels = np.hstack((
                    np.zeros((row_count, crop, nb_channels), dtype = np.uint8),
                    img_pixels[start_row:end_row, start_col:end_col],
                    np.zeros((row_count, self.tile_width - col_count - crop, nb_channels), dtype=np.uint8)
                ))
                tile_pixels = np.vstack((
                    np.zeros((crop, self.tile_width, nb_channels), dtype=np.uint8),
                    tile_pixels,
                    np.zeros((self.tile_height - row_count - crop, self.tile_width, nb_channels), dtype = np.uint8)
                ))
                self.set_tile(index, tile_pixels, compression_level)
        img.close()

def write_pngchunk_withcrc(file: BytesWriter, data_type: bytes, data: bytes):
    file.write(struct.pack(">I", len(data)))
    file.write(data_type)
    file.write(data)
    file.write(struct.pack(">I", zlib.crc32(data_type + data, 0) & 0xffffffff))     
        
"""
HEP (bmp_type = 0x0C) header palette :
1 byte per tile, 0x00 / 0x01 / 0x02 (?)

"""
# TODO : - test img extraction with new implementation (numpy)
#        - implement img injection