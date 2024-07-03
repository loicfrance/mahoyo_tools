
from math import floor
from typing import Literal, cast, TYPE_CHECKING
if TYPE_CHECKING :
    from _typeshed import SupportsRead, SupportsWrite

class BitStreamReader :

    def __init__(self, file: 'SupportsRead[bytes]', msb_first: bool = False) -> None:
        self._file = file
        self._shift = -1 if msb_first else 8
        self._curr_byte = 0
        self._msb_first = msb_first
    
    def _readBit_lsb(self) -> Literal[1, 0] :
        if self._shift == 8 :
            self._curr_byte = self._file.read(1)[0]
            self._shift = 1
            return cast(Literal[1, 0], self._curr_byte & 1)
        else :
            val = (self._curr_byte >> self._shift) & 1
            self._shift += 1
            return cast(Literal[1, 0], val)

    def _readBit_msb(self) -> Literal[1, 0] :
        if self._shift == -1 :
            self._curr_byte = self._file.read(1)[0]
            self._shift = 6
            return cast(Literal[1, 0], self._curr_byte >> 7)
        else :
            val = (self._curr_byte >> self._shift) & 1
            self._shift -= 1
            return cast(Literal[1, 0], val)
    
    
    def readBit(self) -> Literal[1, 0] :
        if self._msb_first :
            return self._readBit_msb()
        else :
            return self._readBit_lsb()
    
    def _read_lsb(self, nb_bits: int) -> int :
        if nb_bits == 0 :
            return 0
        nb_new_bytes = floor((self._shift + nb_bits - 1) / 8)
        if nb_new_bytes > 0 :
            self._curr_byte |= (int.from_bytes(self._file.read(nb_new_bytes), 'little')) << 8
        
        val = (self._curr_byte >> self._shift) % (1 << nb_bits)
        self._shift += nb_bits
        if nb_new_bytes > 0 :
            self._curr_byte >>= 8 * nb_new_bytes
            self._shift -= 8 * nb_new_bytes
        return val
    
    def _read_msb(self, nb_bits: int) -> int :
        if nb_bits == 0 :
            return 0
        nb_new_bytes = floor((nb_bits + (6-self._shift))/8)
        if nb_new_bytes > 0 :
            self._curr_byte = (self._curr_byte % (1 << (self._shift+1))) << (8*nb_new_bytes) \
                            | int.from_bytes(self._file.read(nb_new_bytes), 'big')
        else :
            self._curr_byte %= (1 << self._shift+1)
        self._shift = (self._shift - nb_bits) % 8
        if self._shift == 7 :
            self._shift = -1

        val = self._curr_byte >> (self._shift+1)
        self._curr_byte &= 0xFF
        return val

    def read(self, nb_bits: int) -> int :
        if self._msb_first :
            return self._read_msb(nb_bits)
        else :
            return self._read_lsb(nb_bits)

class BitStreamWriter :

    def __init__(self, file: 'SupportsWrite[bytes]', msb_first: bool = False) -> None:
        self._file = file
        self._shift = 0
        self._curr_byte = 0
        self._msb_first = msb_first
    
    def writeBit(self, bit: Literal[1, 0]) :
        # TODO MSB-first case
        self._curr_byte |= bit << self._shift
        if self._shift == 7 :
            self._shift = 0
            self._file.write(self._curr_byte.to_bytes(1))
            self._curr_byte = 0
        else :
            self._shift += 1
    
    def write(self, nb_bits: int, value: int) :
        # TODO MSB-first case
        self._curr_byte |= value << self._shift
        self._shift += nb_bits
        if self._shift >= 8 :
            nb_bytes = floor(self._shift / 8)
            written_bits = nb_bytes * 8
            self._file.write((self._curr_byte % (1 << written_bits)).to_bytes(nb_bytes, 'little'))
            self._curr_byte >>= written_bits
            self._shift %= 8
    
    def flush(self) :
        if self._shift != 0:
            self._file.write(self._curr_byte.to_bytes(1))
            self._shift = 0
            self._curr_byte = 0