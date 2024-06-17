from functools import reduce
from io import BytesIO
from math import ceil, floor
import struct
from typing import Iterable, Mapping, Optional, Union, overload

class Glyph :
    def __init__(self, index: int, char: str, width: int) -> None:
        self.index = index
        self.char = char
        self.width = width
    
    @property
    def page(self) : return floor(self.index / 98)
    @property
    def column(self) : return self.index % 14
    @property
    def row(self) : return floor((self.index % 98) / 14)
    @property
    def x(self) : return ceil(self.column * 73 + (73-self.width)/2),
    @property
    def y(self) : return self.row * 73
    @property
    def height(self) : return 73

    @property
    def ccit_char_index(self) :
        return self.char.encode('utf-8').rjust(4, b'\x00') + \
               self.index.to_bytes(4, 'little')
    
    @property
    def ccit_coords(self) :
        return struct.pack('<5I', self.page, self.x, self.y,
                           self.width, self.height)
    
    @property
    def txt_line(self) :
        return '\t\t'.join([
            f"char={self.char}", f"index={self.index}",
            f"page={self.page}", f"x={self.x}", f"y={self.y}",
            f"width={self.width}", f"height={self.height}"
        ])

class Font :
    def __init__(self, chars: Union[dict[str, int], Iterable[str]]) -> None:
        if isinstance(chars, dict) :
            self.glyphs = {
                char: Glyph(i, char, width) for [i, (char, width)] in enumerate(chars.items())
            }
            self.use_coordinates = True
        else :
            self.glyph = {
                char: Glyph(i, char, 73) for [i, char] in enumerate(chars)
            }
            self.use_coordinates = False
    
    def _refresh_glyphs(self) :
        self.glyphs = {
            g.char: g for g in sorted(self.glyphs.values(), key=lambda g: g.index)
        }
        
    def swap_chars(self, swap_map: Mapping[str, str]) :
        reverse_swap = dict([(v, k) for k, v in swap_map.items()])
        
        for glyph in self.glyphs.values() :
            if glyph.char in swap_map :
                glyph.char = swap_map.get(glyph.char)
            elif glyph.char in reverse_swap :
                glyph.char = reverse_swap.get(glyph.char)
        self._refresh_glyphs()
    
    def string_width(self, text: str) -> int :
        total = 0
        for c in text :
            glyph = self.glyphs.get(c)
            if glyph is None :
                total += 73
            else :
                total += glyph.width
        return total
    
    @overload
    def to_ccit(self, dest: Optional[BytesIO] = None) -> BytesIO : ...
    @overload
    def to_ccit(self, dest: str) -> None : ...
    
    def to_ccit(self, dest: Union[BytesIO, str, None] = None) -> Optional[BytesIO] :
        if isinstance(dest, str) :
            file = open(dest, "wb")
        elif dest is None :
            file = BytesIO()
        file.write(struct.pack('<2I', len(self.glyphs), 8)) # 8 bytes for char and index

        for glyph in self.glyphs.values():
            file.write(glyph.ccit_char_index)

        if self.use_coordinates > 2:
            file.write(struct.pack('<2I', len(self.glyphs), 20)) # 20 bytes for coordinates
            for glyph in self.glyphs.values():
                file.write(glyph.ccit_coords)
        else:
            file.write(0 .to_bytes(4, 'little'))

        if isinstance(dest, str) :
            file.close()
        else :
            return file
    
    def to_txt(self) -> str :
        return "\n".join([g.txt_line for g in self.glyphs.values()])